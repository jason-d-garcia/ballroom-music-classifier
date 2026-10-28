"""Build data/manifest.csv from the Extended Ballroom XML and the downloaded audio.

Get the data first (audio is not redistributed; the official script fetches 30 s previews):
    1. Download extendedballroom_v1.1.zip from
       http://anasynth.ircam.fr/home/media/ExtendedBallroom and unzip it into data/.
    2. Run its getExtendedBallroomDataset.py so the mp3s land under data/ (one folder per dance).

    python manifest.py --xml data/extendedballroom_v1.1.xml --audio data/

The XML is parsed loosely: every element with an `id` attribute is a track, its dance is the
nearest ancestor tag (or a `genre`/`dance` attribute), and other attributes (title, artist, bpm,
hash) are kept when present; child <version|exact|time|karaoke id=".."/> links go to `dup_of`.
Audio is matched by file stem == id, falling back to the folder name as the dance when the XML
does not give one.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from common import DATA, is_raw_dance, raw_to_style

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
# Extended Ballroom marks repeats as child elements of a song pointing at the other track's id.
DUP_TAGS = {"version", "exact", "time", "karaoke"}


def parse_xml(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    rows = []

    def walk(el, ancestors):
        if "id" in el.attrib and not is_raw_dance(el.tag):
            row = {k.lower(): v for k, v in el.attrib.items()}
            dance = row.get("genre") or row.get("dance")
            if not dance:
                for a in reversed(ancestors):
                    if is_raw_dance(a.tag) or is_raw_dance(a.attrib.get("name", "")):
                        dance = a.tag if is_raw_dance(a.tag) else a.attrib["name"]
                        break
            row["raw_dance"] = dance
            dups = []
            for child in el:  # <title>..</title> style children
                if child.text and child.text.strip():
                    row.setdefault(child.tag.lower(), child.text.strip())
                elif child.tag.lower() in DUP_TAGS and "id" in child.attrib:
                    dups.append(child.attrib["id"])  # <version id=".."/>, <exact id=".."/>, ...
            row["dup_of"] = ";".join(dups) or None
            rows.append(row)
            return
        for child in el:
            walk(child, ancestors + [el])

    walk(root, [])
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--audio", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=DATA / "manifest.csv")
    args = ap.parse_args()

    meta = parse_xml(args.xml)
    files = {p.stem: p for p in args.audio.rglob("*") if p.suffix.lower() in AUDIO_EXT}
    meta["path"] = meta["id"].map(lambda i: str(files[i]) if i in files else None)
    meta["raw_dance"] = meta["raw_dance"].fillna(
        meta["path"].map(lambda p: Path(p).parent.name if p else None))
    meta["style"] = meta["raw_dance"].map(lambda d: raw_to_style(d) if isinstance(d, str) else None)

    print(f"XML tracks: {len(meta)}, with audio: {meta['path'].notna().sum()}")
    print("Raw dance counts:\n", meta["raw_dance"].value_counts(dropna=False).to_string())
    keep = meta[meta["path"].notna() & meta["style"].notna()].copy()
    for col in ("title", "artist", "bpm", "dup_of"):
        if col not in keep:
            keep[col] = None
    keep = keep[["id", "path", "style", "raw_dance", "title", "artist", "bpm", "dup_of"]]
    keep.to_csv(args.out, index=False)
    print(f"\nKept {len(keep)} clips in {keep['style'].nunique()} styles -> {args.out}")
    print(keep["style"].value_counts().to_string())


if __name__ == "__main__":
    main()
