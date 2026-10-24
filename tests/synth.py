"""Synthetic dance-like audio with a known tempo and meter, so tests need no dataset."""
from __future__ import annotations

import numpy as np

SR = 22050


def _tone(freq: float, dur: float, sr: int, decay: float, n: int | None = None) -> np.ndarray:
    t = np.arange(n or int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)


def dance_track(bpm: float, beats_per_bar: int, seconds: float = 20.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Bass + accented click on each downbeat, lighter chord stabs on the other beats, and a
    chord change every bar. Roughly what a waltz (3) or a 4/4 dance band sounds like."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * sr))
    beat = 60.0 / bpm
    roots = [110.0, 146.8, 164.8, 130.8]  # A, D, E, C: one chord per bar
    for i in range(int(seconds / beat)):
        start = int(i * beat * sr)
        bar, pos = divmod(i, beats_per_bar)
        root = roots[bar % len(roots)]
        n = int(0.25 * sr)
        if pos == 0:
            ev = 1.0 * _tone(root / 2, 0, sr, 12, n) + 0.6 * _tone(2000, 0, sr, 120, n)
        else:
            ev = 0.35 * sum(_tone(root * m, 0, sr, 25, n) for m in (2, 2.52, 3)) / 3 \
                + 0.2 * _tone(3000, 0, sr, 150, n)
        end = min(len(y), start + len(ev))
        y[start:end] += ev[: end - start]
    y += 0.002 * rng.standard_normal(len(y))
    return (0.8 * y / np.abs(y).max()).astype(np.float32)
