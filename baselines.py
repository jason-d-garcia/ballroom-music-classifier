"""Non-foundation-model baselines, on the same split and metrics as the probe.

Ballroom styles are largely defined by tempo and meter, and Sturm (2016, "The 'Horse' Inside")
showed classic Ballroom systems mostly learn tempo. So the foundation model has to beat:

  tempo_oracle      the dataset's annotated BPM alone (boosted trees) -> how far tempo gets you
  tempo_estimated   librosa's estimated BPM alone (boosted trees)     -> same, but realistic
  rhythm_timbre     autocorrelation tempogram + MFCC stats (logistic) -> classic MIR features

    python baselines.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common import DATA, FEATS, RUNS, STYLE2ID
from metrics import evaluate

SR = 22050


def handcrafted(path: str) -> tuple[float, np.ndarray]:
    y, sr = librosa.load(path, sr=SR, mono=True, duration=30.0)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    bpm = float(librosa.feature.tempo(onset_envelope=onset, sr=sr)[0])
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=sr, win_length=384).mean(1)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return bpm, np.concatenate([tg, mf.mean(1), mf.std(1)])


def tempo_feats(bpm: np.ndarray) -> np.ndarray:
    # log scale, plus half/double so octave errors land near the right neighbours
    lb = np.log2(np.clip(bpm, 30, 400))
    return np.stack([lb, lb - 1, lb + 1], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DATA / "manifest.csv")
    ap.add_argument("--split", type=Path, default=DATA / "split.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"id": str}).merge(
        pd.read_csv(args.split, dtype={"id": str})[["id", "split"]], on="id")
    cache = FEATS / "handcrafted.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        got = dict(zip(z["ids"], zip(z["bpm"], z["F"])))
    else:
        got = {}
    todo = df[~df["id"].isin(got)]
    for row in tqdm(todo.itertuples(), total=len(todo), desc="tempo + tempogram"):
        try:
            got[row.id] = handcrafted(row.path)
        except Exception as e:
            print("skip", row.id, e)
    ids = list(got)
    FEATS.mkdir(exist_ok=True)
    np.savez_compressed(cache, ids=np.array(ids), bpm=np.array([got[i][0] for i in ids]),
                        F=np.stack([got[i][1] for i in ids]))

    df = df[df["id"].isin(got)]
    df["est_bpm"] = df["id"].map(lambda i: got[i][0])
    F = np.stack([got[i][1] for i in df["id"]])
    y = df["style"].map(STYLE2ID).to_numpy()
    tr, te = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()

    def run(name, X, clf, extra=None):
        clf.fit(X[tr], y[tr])
        return evaluate(name, clf.predict_proba(X[te]), y[te], RUNS, extra)

    # Not k-NN: BPMs are heavily tied (librosa returns ~23 distinct values, annotations are integers), and
    # k-NN then picks its k neighbours among hundreds of ties by row order, i.e. mostly from one style.
    tempo_clf = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                 random_state=42)
    if df["bpm"].notna().mean() > 0.9:
        ok = df["bpm"].notna().to_numpy()
        tr_, te_ = tr, te
        tr, te = tr & ok, te & ok
        run("tempo_oracle", tempo_feats(pd.to_numeric(df["bpm"], errors="coerce").fillna(120).to_numpy()), tempo_clf(),
            {"note": "annotated BPM from the dataset; upper bound for tempo-only"})
        tr, te = tr_, te_
    run("tempo_estimated", tempo_feats(df["est_bpm"].to_numpy()), tempo_clf())
    run("rhythm_timbre", F, make_pipeline(StandardScaler(), LogisticRegression(
        C=0.05, max_iter=3000, class_weight="balanced")))


if __name__ == "__main__":
    main()
