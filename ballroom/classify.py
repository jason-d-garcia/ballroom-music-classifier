"""Rank the ballroom dances a song fits, with its tempo and which styles' tempo rules it meets.

    ballroom-classify song.mp3 some_folder/                    # local audio
    ballroom-classify --songs songs.csv --out tagged.csv       # CSV with title,artist -> Deezer previews
    ballroom-classify song.mp3 --model tempo                   # no MERT: tempo + meter rules only
    ballroom-classify song.mp3 --json                          # machine-readable output

Two models:
  mert   (default when runs/mert_probe.pt exists) frozen MERT-v1-95M + the trained probe;
         95.6% top-1 on held-out Extended Ballroom clips. Needs torch and transformers.
  tempo  ranks dances by estimated tempo and meter against the competition tempo rules.
         Works anywhere librosa does; much less accurate.

Each ranked dance comes with its tempo in MPM (bars per minute, corrected for half/double-tempo
errors given that dance) and its fit against each style's rule, e.g. "Foxtrot 29 MPM:
International Standard fits, American Smooth close". Deezer's public search API returns a 30 s
preview per track with no key, the same length as the training clips. Previews are cached in
data/previews/ and never committed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np

from .common import DATA, RUNS
from .dances import DANCES, describe_fit, style_fit, to_mpm
from .rhythm import SR as RHYTHM_SR
from .rhythm import Rhythm, TempoRuleClassifier, analyze

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
CLIP_SECONDS = 30.0


class TempoModel:
    name = "tempo"
    sr = RHYTHM_SR

    def __init__(self):
        self.clf = TempoRuleClassifier()
        self.styles = self.clf.styles

    def predict(self, y: np.ndarray, rhythm: Rhythm) -> np.ndarray:
        return self.clf.scores(rhythm)


class ProbeModel:
    """Frozen foundation-model embedder + the LayerMixProbe saved by train_probe.py."""
    name = "mert"

    def __init__(self, probe_path: Path, embedder_factory=None):
        import torch
        from .embed import MODELS, MertEmbedder
        from .train_probe import LayerMixProbe
        from .common import device

        self._torch = torch
        ck = torch.load(probe_path, weights_only=False, map_location="cpu")
        n_layers, dim = ck["shape"]
        self.styles = list(ck["styles"])
        self.probe = LayerMixProbe(n_layers, dim, len(self.styles), ck["hidden"], ck["dropout"])
        self.probe.load_state_dict(ck["state_dict"])
        self.probe.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        if embedder_factory is None:
            embedder_factory = lambda: MertEmbedder(MODELS[ck["features"]], device())  # noqa: E731
        self.emb = embedder_factory()
        self.sr = self.emb.sr

    def predict(self, y: np.ndarray, rhythm: Rhythm) -> np.ndarray:
        from .embed import windows_of
        torch = self._torch
        x = (self.emb(windows_of(y, self.sr, 5.0))[None] - self.mu) / self.sd
        with torch.no_grad():
            return torch.softmax(self.probe(torch.tensor(x, dtype=torch.float32)), 1)[0].numpy()


def load_model(kind: str, probe_path: Path):
    if kind == "tempo" or (kind == "auto" and not probe_path.exists()):
        if kind == "auto":
            print(f"No trained probe at {probe_path}; using the tempo-only model "
                  "(pass --probe, or see the README to train one).", file=sys.stderr)
        return TempoModel()
    if not probe_path.exists():
        raise SystemExit(f"--model mert needs a trained probe; {probe_path} does not exist")
    return ProbeModel(probe_path)


def rank(proba: np.ndarray, styles: list[str], bpm: float, top_k: int) -> list[dict]:
    out = []
    for r, i in enumerate(np.argsort(-proba)[:top_k], 1):
        name = styles[i]
        d = DANCES[name]
        mpm = to_mpm(bpm, name)
        out.append({"rank": r, "dance": name, "probability": round(float(proba[i]), 3),
                    "categories": d.categories, "mpm": round(mpm, 1), "bpm": round(mpm * d.bpb, 1),
                    "style_fit": style_fit(mpm, name)})
    return out


def classify_audio(y: np.ndarray, sr: int, model, top_k: int = 3) -> dict:
    """y: mono audio at `sr`. Returns the estimated tempo/meter and the top_k ranked dances."""
    y_r = y if sr == RHYTHM_SR else librosa.resample(y, orig_sr=sr, target_sr=RHYTHM_SR)
    rhythm = analyze(y_r, RHYTHM_SR)
    y_m = y if sr == model.sr else librosa.resample(y, orig_sr=sr, target_sr=model.sr)
    proba = model.predict(y_m, rhythm)
    return {"model": model.name, "tempo_bpm": round(rhythm.bpm, 1), "meter": rhythm.meter,
            "p_triple": round(rhythm.p_triple, 2), "dances": rank(proba, model.styles, rhythm.bpm, top_k)}


def format_result(label: str, res: dict) -> str:
    if res.get("status", "ok") != "ok":
        return f"{label}: {res['status']}"
    lines = [f"{label}  (tracker: ~{res['tempo_bpm']:.0f} BPM, sounds like {res['meter']}; model: {res['model']})"]
    for d in res["dances"]:
        lines.append(f"  {d['rank']}. {d['dance']:<15} {d['probability']:.2f}   "
                     f"{d['mpm']:.1f} MPM ({d['bpm']:.0f} BPM)")
        lines.append(f"     {describe_fit(d['style_fit'])}")
    return "\n".join(lines)


def flat_row(meta: dict, res: dict) -> dict:
    row = {**meta, "status": res.get("status", "ok")}
    if row["status"] != "ok":
        return row
    row.update(tempo_bpm=res["tempo_bpm"], meter=res["meter"])
    for d in res["dances"]:
        k = d["rank"]
        row[f"dance{k}"], row[f"p{k}"], row[f"mpm{k}"] = d["dance"], d["probability"], d["mpm"]
        row[f"fits{k}"] = "; ".join(f"{f['category']} ({f['dance']})" for f in d["style_fit"]
                                    if f["fit"] == "fits")
    return row


def deezer_preview(title: str, artist: str, cache: Path) -> tuple[Path | None, str | None]:
    import requests
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


def collect_inputs(inputs: list[Path], songs: Path | None) -> list[tuple[dict, Path | None]]:
    items = []
    for p in inputs:
        if not p.exists():
            items.append(({"file": str(p)}, None))
            continue
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        items += [({"file": str(f)}, f) for f in files]
    if songs:
        import pandas as pd
        for row in pd.read_csv(songs).itertuples():
            try:
                path, link = deezer_preview(str(row.title), str(row.artist), DATA / "previews")
            except Exception as e:  # one bad lookup should not stop a playlist
                path, link = None, None
                print("deezer error:", row.title, e, file=sys.stderr)
            items.append(({"title": row.title, "artist": row.artist, "deezer": link}, path))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ballroom-classify", description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="*", type=Path, help="audio files or folders")
    ap.add_argument("--songs", type=Path, help="CSV with title,artist columns (fetches Deezer previews)")
    ap.add_argument("--model", choices=["auto", "mert", "tempo"], default="auto")
    ap.add_argument("--probe", type=Path, default=RUNS / "mert_probe.pt")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--offset", type=float, default=0.0, help="seconds into each file to start the 30 s clip")
    ap.add_argument("--out", type=Path, help="write results to .csv or .json")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    if not args.inputs and not args.songs:
        ap.error("give audio files/folders or --songs")

    model = load_model(args.model, args.probe)
    results = []
    for meta, path in collect_inputs(args.inputs, args.songs):
        label = meta.get("title") or meta.get("file")
        if path is None:
            res = {"status": "not found" if "file" in meta else "no preview"}
        else:
            try:
                y, sr = librosa.load(path, sr=model.sr, mono=True, offset=args.offset, duration=CLIP_SECONDS)
                res = classify_audio(y, sr, model, args.top_k)
            except Exception as e:  # unreadable file: report it and keep going
                res = {"status": f"error: {e}"}
        results.append({**meta, **res})
        if not args.json:
            print(format_result(label, res))

    if args.json:
        print(json.dumps(results, indent=2))
    if args.out:
        if args.out.suffix.lower() == ".json":
            args.out.write_text(json.dumps(results, indent=2))
        else:
            import pandas as pd
            meta_keys = ("file", "title", "artist", "deezer")
            pd.DataFrame([flat_row({k: r[k] for k in meta_keys if k in r}, r) for r in results]) \
                .to_csv(args.out, index=False)
        print(f"Wrote {len(results)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
