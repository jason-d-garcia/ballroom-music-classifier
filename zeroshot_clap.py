"""Zero-shot CLAP: no training at all, just compare each clip's audio embedding with text
prompts like "ballroom dance music for a rumba". This is the "use a pre-existing model"
baseline. Needs features/clap.npz from `python embed.py --model clap`.

    python zeroshot_clap.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import DATA, RUNS, STYLE2ID, STYLES
from embed import MODELS
from metrics import evaluate

TEMPLATES = ["{} music", "a {} song", "ballroom dance music for a {}", "music to dance the {} to"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("features/clap.npz"))
    ap.add_argument("--split", type=Path, default=DATA / "split.csv")
    args = ap.parse_args()

    from transformers import ClapModel, ClapProcessor
    name = MODELS["clap"]  # must match the checkpoint that produced features/clap.npz
    model, proc = ClapModel.from_pretrained(name).eval(), ClapProcessor.from_pretrained(name)
    with torch.no_grad():
        T = []
        for s in STYLES:
            inp = proc(text=[t.format(s.lower()) for t in TEMPLATES], return_tensors="pt", padding=True)
            e = torch.nn.functional.normalize(model.get_text_features(**inp), dim=-1).mean(0)
            T.append(torch.nn.functional.normalize(e, dim=0))
        T = torch.stack(T).numpy()
        scale = model.logit_scale_a.exp().item()

    z = np.load(args.features)
    A = z["X"][:, 0].astype(np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    pos = {i: k for k, i in enumerate(z["ids"])}
    sp = pd.read_csv(args.split, dtype={"id": str})
    sp = sp[(sp["split"] == "test") & sp["id"].isin(pos)]
    logits = scale * A[[pos[i] for i in sp["id"]]] @ T.T
    proba = torch.softmax(torch.tensor(logits), 1).numpy()
    evaluate("clap_zeroshot", proba, sp["style"].map(STYLE2ID).to_numpy(), RUNS,
             {"templates": TEMPLATES})


if __name__ == "__main__":
    main()
