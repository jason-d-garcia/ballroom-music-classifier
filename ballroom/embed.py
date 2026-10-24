"""Extract frozen foundation-model embeddings for every clip in a manifest.

    python -m ballroom.embed --model mert              # m-a-p/MERT-v1-95M, all 13 layers, mean+std pooled
    python -m ballroom.embed --model clap              # laion/larger_clap_music, joint audio-text space
    python -m ballroom.embed --model mert --manifest my_songs.csv --out features/my_songs_mert.npz

Each 30 s clip is cut into 5 s windows (MERT was pre-trained on 5 s crops); per layer we pool
hidden states over time (mean and std, so some rhythmic variation survives) and then average
the windows. Output: features/<name>.npz with X [N, layers, dim] float16 and ids [N].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .common import DATA, FEATS, device

MODELS = {
    "mert": "m-a-p/MERT-v1-95M",
    "mert330": "m-a-p/MERT-v1-330M",
    "clap": "laion/larger_clap_music_and_speech",  # larger_clap_music on the Hub is broken: logit scale ~0, all embeddings ~identical
}


class MertEmbedder:
    def __init__(self, name: str, dev):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.proc = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(dev).eval()
        self.sr = self.proc.sampling_rate  # 24 kHz
        self.dev = dev

    @torch.no_grad()
    def __call__(self, windows: list[np.ndarray]) -> np.ndarray:
        inp = self.proc(windows, sampling_rate=self.sr, return_tensors="pt")
        out = self.model(input_values=inp["input_values"].to(self.dev), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)                # [W, L, T, D]
        feat = torch.cat([hs.mean(2), hs.std(2)], dim=-1)         # [W, L, 2D]
        return feat.mean(0).float().cpu().numpy()                 # [L, 2D]


class ClapEmbedder:
    def __init__(self, name: str, dev):
        from transformers import ClapModel, ClapProcessor
        self.proc = ClapProcessor.from_pretrained(name)
        self.model = ClapModel.from_pretrained(name).to(dev).eval()
        self.sr = self.proc.feature_extractor.sampling_rate  # 48 kHz
        self.dev = dev

    @torch.no_grad()
    def __call__(self, windows: list[np.ndarray]) -> np.ndarray:
        inp = self.proc(audios=windows, sampling_rate=self.sr, return_tensors="pt")
        emb = self.model.get_audio_features(input_features=inp["input_features"].to(self.dev))
        emb = torch.nn.functional.normalize(emb, dim=-1).mean(0, keepdim=True)
        return emb.float().cpu().numpy()                          # [1, 512]


def windows_of(y: np.ndarray, sr: int, win_s: float) -> list[np.ndarray]:
    n = int(win_s * sr)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return [y[i:i + n] for i in range(0, len(y) - n + 1, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="mert")
    ap.add_argument("--manifest", type=Path, default=DATA / "manifest.csv")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--window", type=float, default=None, help="seconds; default 5 (MERT) / 10 (CLAP)")
    args = ap.parse_args()

    dev = device()
    is_clap = args.model == "clap"
    emb = (ClapEmbedder if is_clap else MertEmbedder)(MODELS[args.model], dev)
    win = args.window or (10.0 if is_clap else 5.0)
    out = args.out or FEATS / f"{args.model}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, dtype={"id": str})
    feats, ids, failed = [], [], []
    partial = out.with_suffix(".partial.npz")  # checkpoint so an interrupted run resumes
    if partial.exists():
        z = np.load(partial)
        feats, ids = list(z["X"]), [str(i) for i in z["ids"]]
        print(f"Resuming from {partial}: {len(ids)} clips done")
    done = set(ids)
    todo = df[~df["id"].isin(done)]
    for k, row in enumerate(tqdm(todo.itertuples(), total=len(todo), desc=f"{args.model} on {dev}")):
        try:
            y, _ = librosa.load(row.path, sr=emb.sr, mono=True, duration=30.0)
            feats.append(emb(windows_of(y, emb.sr, win)).astype(np.float16))
            ids.append(row.id)
        except Exception as e:  # corrupt or missing previews should not kill a long run
            failed.append((row.id, repr(e)))
        if (k + 1) % 250 == 0:
            np.savez(partial, X=np.stack(feats), ids=np.array(ids))
    tmp = out.with_suffix(".tmp.npz")  # write then rename, so readers never see a half-written file
    np.savez_compressed(tmp, X=np.stack(feats), ids=np.array(ids))
    tmp.replace(out)
    partial.unlink(missing_ok=True)
    print(f"Saved {len(ids)} x {feats[0].shape} -> {out}; failed: {len(failed)}")
    for f in failed[:20]:
        print("  ", *f)


if __name__ == "__main__":
    main()
