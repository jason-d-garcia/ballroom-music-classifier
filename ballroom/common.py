"""Shared constants and helpers for the ballroom dance-style classifier."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# Working directory for data/features/runs: the repo checkout, or $BALLROOM_HOME if set.
ROOT = Path(os.environ.get("BALLROOM_HOME", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"          # audio + metadata (never committed)
FEATS = ROOT / "features"     # cached embeddings
RUNS = ROOT / "runs"          # metrics, checkpoints, confusion matrices

# Extended Ballroom folder/genre names -> our label. Slow waltz is the same dance as waltz
# (it is the competition name), so it is merged, as in Pavlin et al. (ICPR 2020).
# Salsa (47 clips) and West Coast Swing (23) are too small to evaluate honestly and are dropped.
RAW_TO_STYLE = {
    "chacha": "Cha-cha",
    "foxtrot": "Foxtrot",
    "jive": "Jive",
    "pasodoble": "Paso doble",
    "quickstep": "Quickstep",
    "rumba": "Rumba",
    "samba": "Samba",
    "tango": "Tango",
    "viennesewaltz": "Viennese waltz",
    "waltz": "Waltz",
    "slowwaltz": "Waltz",
}
DROPPED_RAW = {"salsa", "wcswing"}
STYLES = sorted(set(RAW_TO_STYLE.values()))
STYLE2ID = {s: i for i, s in enumerate(STYLES)}

SEED = 42


def raw_to_style(raw: str) -> str | None:
    key = re.sub(r"[^a-z]", "", raw.lower())
    return RAW_TO_STYLE.get(key)


def is_raw_dance(name: str) -> bool:
    key = re.sub(r"[^a-z]", "", name.lower())
    return key in RAW_TO_STYLE or key in DROPPED_RAW


def norm_text(s: str) -> str:
    """Normalise a title/artist for duplicate detection ("Moon River (Waltz) - 2005 mix")."""
    s = s.lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"\b(remix|mix|version|edit|karaoke|instrumental|live|remaster(ed)?)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def top_k_accuracy(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))
