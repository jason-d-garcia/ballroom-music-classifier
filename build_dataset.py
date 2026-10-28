"""Build dataset/labels.csv: one row per clip or song, labelled with dance, ballroom style family
(International Standard / American Smooth / International Latin / American Rhythm), tempo in
BPM and MPM, and meter, checked against the competition tempo rules in dance_rules.py.

Sources (metadata only; audio is never committed):
  * Extended Ballroom XML (Marchand & Peeters 2016): dance + annotated BPM for 4,180 clips.
    Mirror: github.com/Pliploop/AllTempo, extended_ballroom/ballroom_extended_2016/.
  * Ballroom 2004 beat annotations (Krebs et al. 2013, github.com/CPJKU/BallroomAnnotations):
    beat times + beat-in-bar numbers. "Media-<id>" files are the same ballroomdancers.com
    previews as Extended Ballroom, so they add a second, beat-derived tempo and a measured meter.
  * ballroom-test/candidate_songs.csv: real-world pop songs, labels unconfirmed until Jason checks.

    python build_dataset.py --xml data/extendedballroom_v1.1.xml \
        --beats ../BallroomAnnotations --candidates candidate_songs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, norm_text, raw_to_style
from dance_rules import METER, families, nominal_mpm, tempo_fits
from make_split import UnionFind
from manifest import parse_xml

DROPPED_NAMES = {"salsa": "Salsa", "wcswing": "West Coast Swing"}


def beat_stats(path: Path) -> tuple[float, int]:
    """Tempo (BPM, from the median inter-beat interval) and beats per bar of a .beats file."""
    a = np.loadtxt(path, ndmin=2)
    bpm = 60.0 / float(np.median(np.diff(a[:, 0])))
    return round(bpm, 1), int(a[:, 1].max()) if a.shape[1] > 1 else 0


def extended(xml: Path, beats: Path | None) -> pd.DataFrame:
    df = parse_xml(xml)
    raw = df["raw_dance"].str.lower()
    df["dance"] = raw.map(lambda d: raw_to_style(d) or DROPPED_NAMES.get(d))
    df["in_10_class_set"] = raw.map(lambda d: raw_to_style(d) is not None)
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["source"] = "extended_ballroom"
    df["label_status"] = "dataset_annotation"

    # Same grouping as make_split.py (title+artist and XML repeat links), minus the audio step.
    uf, idx = UnionFind(len(df)), {i: k for k, i in enumerate(df["id"])}
    key = df["title"].fillna("").map(norm_text) + " | " + df["artist"].fillna("").map(norm_text)
    first = {}
    for i, k in enumerate(key):
        if k.strip(" |"):
            uf.union(i, first.setdefault(k, i))
    for i, d in enumerate(df["dup_of"]):
        for j in (d.split(";") if isinstance(d, str) else []):
            if j in idx:
                uf.union(i, idx[j])
    df["dup_group"] = [df["id"][uf.find(i)] for i in range(len(df))]

    if beats is not None:
        stats = {p.stem.split("-", 1)[1]: beat_stats(p)
                 for p in beats.glob("Media-*.beats")}
        df["in_ballroom2004"] = df["id"].isin(stats)
        df["beats_bpm"] = df["id"].map(lambda i: stats.get(i, (None, None))[0])
        df["beats_per_bar_measured"] = df["id"].map(lambda i: stats.get(i, (None, None))[1])
    return df


def candidates(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path)
    confirmed = c.get("jason_confirms", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    return pd.DataFrame({
        "id": ["cand%03d" % i for i in range(len(c))], "title": c["title"], "artist": c["artist"],
        "dance": c["dance"].replace({"Cha Cha": "Cha-cha", "Viennese Waltz": "Viennese waltz"}),
        "in_10_class_set": True, "bpm": np.nan, "source": "candidate_songs",
        "label_status": np.where(confirmed != "", "jason_confirmed", "unconfirmed_guess"),
    })


def add_tempo_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["beats_per_bar"] = df["dance"].map(lambda d: METER.get(d, (None, None))[0])
    df["meter"] = df["dance"].map(lambda d: METER.get(d, (None, None))[1])
    df["mpm"] = (df["bpm"] / df["beats_per_bar"]).round(1)
    df["style_families"] = df["dance"].map(lambda d: ";".join(families(d)) or "Social (not ballroom)")
    df["tempo_fits"] = [";".join(tempo_fits(d, m)) if pd.notna(m) else ""
                        for d, m in zip(df["dance"], df["mpm"])]

    def flag(d, m):
        r = nominal_mpm(d)
        if r is None or pd.isna(m):
            return ""
        lo, hi = r
        if m < lo * 0.75:
            return "half_time?"
        if m > hi * 1.33:
            return "double_time?"
        return "slow" if m < lo - 2 else "fast" if m > hi + 2 else "in_range"
    df["tempo_check"] = [flag(d, m) for d, m in zip(df["dance"], df["mpm"])]
    return df


COLUMNS = ["id", "source", "label_status", "dance", "in_10_class_set", "style_families",
           "meter", "beats_per_bar", "bpm", "mpm", "tempo_fits", "tempo_check", "title",
           "artist", "album", "dup_group", "dup_of", "in_ballroom2004", "beats_bpm",
           "beats_per_bar_measured", "hash"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--beats", type=Path, help="BallroomAnnotations checkout (optional)")
    ap.add_argument("--candidates", type=Path, help="ballroom-test/candidate_songs.csv (optional)")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "labels.csv")
    args = ap.parse_args()

    parts = [extended(args.xml, args.beats)]
    if args.candidates:
        parts.append(candidates(args.candidates))
    df = add_tempo_columns(pd.concat(parts, ignore_index=True))
    df = df.reindex(columns=COLUMNS)
    args.out.parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)

    ext = df[df["source"] == "extended_ballroom"]
    print(f"{len(df)} rows -> {args.out}")
    print(pd.crosstab(ext["dance"], ext["tempo_check"], margins=True).to_string())
    if args.beats:
        m = ext[ext["in_ballroom2004"] == True]  # noqa: E712
        agree = (m["beats_bpm"] / m["bpm"]).between(0.96, 1.04).mean()
        meter_ok = (m["beats_per_bar_measured"] == m["beats_per_bar"].where(
            m["beats_per_bar"] != 2, 4)).mean()
        print(f"\nBallroom 2004 overlap: {len(m)} clips; tempo agrees within 4%: {agree:.1%}; "
              f"measured beats/bar matches expected: {meter_ok:.1%}")


if __name__ == "__main__":
    main()
