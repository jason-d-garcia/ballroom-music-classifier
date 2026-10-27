"""Tag any song with the ballroom styles it fits, for music discovery.

    python classify.py song1.mp3 some_folder/                # local audio
    python classify.py --songs songs.csv --out tagged.csv    # CSV with title,artist -> Deezer 30 s previews

Deezer's public search API returns a 30 s preview per track with no key or login, the same
length as the training clips. Previews are only cached locally (data/previews/), never committed.
Output: top-3 styles with probabilities and the estimated tempo, so a dancer can sanity-check
("Rumba 0.81 at 100 BPM").
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import requests
import torch

from common import DATA, RUNS, device
from embed import MODELS, MertEmbedder, windows_of
from train_probe import LayerMixProbe

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def deezer_preview(title: str, artist: str, cache: Path) -> tuple[Path | None, str | None]:
    slug = re.sub(r"[^a-z0-9]+", "_", f"{artist} {title}".lower()).strip("_")[:120]
    path = cache / f"{slug}.mp3"
    q = f'artist:"{artist}" track:"{title}"'
    r = requests.get("https://api.deezer.com/search", params={"q": q, "limit": 1}, timeout=15).json()
    hits = r.get("data") or requests.get("https://api.deezer.com/search",
                                         params={"q": f"{artist} {title}", "limit": 1}, timeout=15).json().get("data")
    if not hits or not hits[0].get("preview"):
        return None, None
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(hits[0]["preview"], timeout=30).content)
        time.sleep(0.1)  # stay well under Deezer's rate limit
    return path, hits[0].get("link")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("--songs", type=Path, help="CSV with title,artist columns")
    ap.add_argument("--probe", type=Path, default=RUNS / "mert_probe.pt")
    ap.add_argument("--out", type=Path, default=Path("tagged.csv"))
    args = ap.parse_args()

    ck = torch.load(args.probe, weights_only=False)
    L, D = ck["shape"]
    probe = LayerMixProbe(L, D, len(ck["styles"]), ck["hidden"], ck["dropout"])
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    emb = MertEmbedder(MODELS[ck["features"]], device())

    items = []  # (label dict, audio path)
    for p in args.inputs:
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        items += [({"file": str(f)}, f) for f in files]
    if args.songs:
        for row in pd.read_csv(args.songs).itertuples():
            try:
                path, link = deezer_preview(str(row.title), str(row.artist), DATA / "previews")
            except Exception as e:
                path, link = None, None
                print("deezer error:", row.title, e)
            items.append(({"title": row.title, "artist": row.artist, "deezer": link}, path))

    rows = []
    for meta, path in items:
        if path is None:
            rows.append({**meta, "status": "no preview"})
            continue
        y, _ = librosa.load(path, sr=emb.sr, mono=True, duration=30.0)
        x = (emb(windows_of(y, emb.sr, 5.0))[None] - ck["mu"]) / ck["sd"]
        with torch.no_grad():
            p = torch.softmax(probe(torch.tensor(x, dtype=torch.float32)), 1)[0].numpy()
        top = np.argsort(-p)[:3]
        bpm = float(librosa.feature.tempo(y=librosa.resample(y, orig_sr=emb.sr, target_sr=22050), sr=22050)[0])
        row = {**meta, "status": "ok", "est_bpm": round(bpm)}
        for k, i in enumerate(top, 1):
            row[f"style{k}"], row[f"p{k}"] = ck["styles"][i], round(float(p[i]), 3)
        rows.append(row)
        print(f"{meta.get('title') or meta.get('file')}: " +
              ", ".join(f"{row[f'style{k}']} {row[f'p{k}']:.2f}" for k in (1, 2, 3)) + f"  (~{round(bpm)} BPM)")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
