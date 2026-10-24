"""Make a leakage-safe train/val/test split (70/15/15) and save data/split.csv.

Extended Ballroom contains repeated songs (exact copies, karaoke and re-recorded versions; see
Marchand & Peeters 2016), so a random split would put the same song on both sides. Clips are
grouped when their normalised title+artist match, when the dataset XML marks them as
repeats, or when their audio embeddings are
near-identical, and whole groups go to one side. The split is stratified by style.

    python -m ballroom.make_split --features features/mert.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .common import DATA, SEED, norm_text


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DATA / "manifest.csv")
    ap.add_argument("--features", type=Path, default=Path("features/mert.npz"))
    ap.add_argument("--sim", type=float, default=0.995, help="cosine threshold for audio duplicates")
    ap.add_argument("--out", type=Path, default=DATA / "split.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"id": str, "dup_of": str})
    z = np.load(args.features)
    pos = {i: k for k, i in enumerate(z["ids"])}
    df = df[df["id"].isin(pos)].reset_index(drop=True)
    n = len(df)
    uf = UnionFind(n)
    idx = {i: k for k, i in enumerate(df["id"])}

    # 1) same normalised title + artist
    key = df["title"].fillna("").map(norm_text) + " | " + df["artist"].fillna("").map(norm_text)
    first = {}
    for i, k in enumerate(key):
        if k.strip(" |"):
            uf.union(i, first.setdefault(k, i))

    # 1b) duplicates annotated in the dataset XML (version / exact / time / karaoke)
    xml_pairs = 0
    if "dup_of" in df:
        for i, d in enumerate(df["dup_of"]):
            for j in (d.split(";") if isinstance(d, str) else []):
                if j in idx:
                    uf.union(i, idx[j])
                    xml_pairs += 1

    # 2) near-identical audio (last layer, mean-pooled half of the vector)
    X = z["X"][[pos[i] for i in df["id"]]][:, -1, :].astype(np.float32)
    X = X[:, : X.shape[1] // 2] if X.shape[1] > 1000 else X
    X -= X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    audio_pairs = 0
    for s in range(0, n, 512):
        S = X[s:s + 512] @ X.T
        for a, b in zip(*np.where(S > args.sim)):
            if s + a < b:
                uf.union(s + a, b)
                audio_pairs += 1

    df["group"] = [uf.find(i) for i in range(n)]
    sizes = df["group"].value_counts()
    print(f"{n} clips -> {len(sizes)} groups; {int((sizes > 1).sum())} groups with repeats "
          f"({int(sizes[sizes > 1].sum())} clips); {xml_pairs} XML-annotated duplicate links, "
          f"{audio_pairs} audio-duplicate pairs")
    cross = df.groupby("group")["style"].nunique()
    if (cross > 1).any():
        print(f"Note: {int((cross > 1).sum())} groups span several styles (same song in two dances)")

    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.empty(n, dtype=int)
    for f, (_, idx) in enumerate(sgkf.split(df, df["style"], df["group"])):
        fold[idx] = f
    df["split"] = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))
    df[["id", "style", "group", "split"]].to_csv(args.out, index=False)
    print(pd.crosstab(df["style"], df["split"]).to_string())


if __name__ == "__main__":
    main()
