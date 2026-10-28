"""Train a small classifier on frozen foundation-model embeddings.

    python train_probe.py --features features/mert.npz            # learned layer mix + MLP
    python train_probe.py --features features/mert.npz --layer_sweep   # also: logistic per layer

The probe learns a softmax weighting over the encoder's layers (rhythm and timbre live in
different layers), then LayerNorm -> MLP. Model selection uses val macro-F1 only; test is
scored once at the end. Saves runs/<name>_probe.pt for classify.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import DATA, RUNS, SEED, STYLE2ID, STYLES
from metrics import evaluate


class LayerMixProbe(nn.Module):
    def __init__(self, n_layers: int, dim: int, n_classes: int, hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(p), nn.Linear(dim, hidden),
                                  nn.GELU(), nn.Dropout(p), nn.Linear(hidden, n_classes))

    def forward(self, x):                          # x: [B, L, D]
        w = torch.softmax(self.layer_logits, 0)
        return self.head((w[None, :, None] * x).sum(1))


def load(features: Path, split_csv: Path):
    z = np.load(features)
    X = z["X"].astype(np.float32)
    pos = {i: k for k, i in enumerate(z["ids"])}
    sp = pd.read_csv(split_csv, dtype={"id": str})
    sp = sp[sp["id"].isin(pos)]
    parts = {}
    for name in ("train", "val", "test"):
        s = sp[sp["split"] == name]
        parts[name] = (X[[pos[i] for i in s["id"]]], s["style"].map(STYLE2ID).to_numpy(), s["id"].to_numpy())
    return parts


def train(parts, epochs, lr, wd, hidden, dropout, batch, dev):
    Xtr, ytr, _ = parts["train"]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-5
    t = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    Xtr_t, Xva_t = t(Xtr), t(parts["val"][0]).to(dev)
    ytr_t = torch.tensor(ytr)
    counts = np.bincount(ytr, minlength=len(STYLES))
    cw = torch.tensor(len(ytr) / (len(STYLES) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)

    torch.manual_seed(SEED)
    model = LayerMixProbe(Xtr.shape[1], Xtr.shape[2], len(STYLES), hidden, dropout).to(dev)
    # layer weights get a larger step and no decay, or they barely move from uniform
    opt = torch.optim.AdamW([{"params": [model.layer_logits], "lr": lr * 20, "weight_decay": 0.0},
                             {"params": model.head.parameters()}], lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    best, best_state, patience = -1.0, None, 0
    from sklearn.metrics import f1_score
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = Xtr_t[idx].to(dev)
            xb = xb + 0.05 * torch.randn_like(xb)   # light feature noise
            loss = loss_fn(model(xb), ytr_t[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).argmax(1).cpu().numpy()
        f1 = f1_score(parts["val"][1], pv, average="macro")
        if f1 > best:
            best, best_state, patience = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 25:
                break
    model.load_state_dict(best_state)
    return model, mu, sd, best, ep + 1


def layer_sweep(parts, out: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for layer in range(parts["train"][0].shape[1]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        clf.fit(parts["train"][0][:, layer], parts["train"][1])
        rows.append({"layer": layer, "val_macro_f1": round(
            f1_score(parts["val"][1], clf.predict(parts["val"][0][:, layer]), average="macro"), 4)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("features/mert.npz"))
    ap.add_argument("--split", type=Path, default=DATA / "split.csv")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--layer_sweep", action="store_true")
    args = ap.parse_args()

    name = args.features.stem
    RUNS.mkdir(parents=True, exist_ok=True)
    parts = load(args.features, args.split)
    print({k: len(v[1]) for k, v in parts.items()}, "features", parts["train"][0].shape[1:])
    if args.layer_sweep:
        layer_sweep(parts, RUNS / f"{name}_layer_sweep.csv")

    dev = torch.device("cpu")  # the probe is tiny; CPU is fast and deterministic
    model, mu, sd, val_f1, epochs = train(parts, args.epochs, args.lr, args.wd, args.hidden,
                                          args.dropout, args.batch, dev)
    model.eval()
    with torch.no_grad():
        Xte = torch.tensor((parts["test"][0] - mu) / sd, dtype=torch.float32)
        proba = torch.softmax(model(Xte), 1).numpy()
    w = torch.softmax(model.layer_logits, 0).detach().numpy()
    evaluate(f"{name}_probe", proba, parts["test"][1], RUNS,
             {"val_macro_f1": round(val_f1, 4), "epochs": epochs,
              "layer_weights": [round(float(v), 3) for v in w]})
    pd.DataFrame(proba, columns=STYLES).assign(id=parts["test"][2], true=[STYLES[i] for i in parts["test"][1]]) \
        .to_csv(RUNS / f"{name}_probe_test_predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "styles": STYLES,
                "shape": parts["train"][0].shape[1:], "hidden": args.hidden, "dropout": args.dropout,
                "features": name}, RUNS / f"{name}_probe.pt")


if __name__ == "__main__":
    main()
