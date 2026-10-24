"""Tempo and meter estimates with librosa, plus a tempo-only fallback classifier.

Ballroom dances are defined largely by tempo and meter, so these are reported next to every
prediction (a dancer can sanity-check "Rumba at 25 MPM"), and `TempoRuleClassifier` ranks
dances from them alone when the MERT model isn't available. On Extended Ballroom a learned
tempo-only model reaches 57% (estimated BPM) to 87% (annotated BPM), so treat the fallback as a
rough guide, not a substitute for the MERT probe.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .common import STYLES
from .dances import DANCES

SR = 22050
HOP = 512


@dataclass
class Rhythm:
    bpm: float              # quarter-note beats per minute as the beat tracker reports it
    p_triple: float         # 0..1, confidence the music is in 3 (waltz family) rather than 2 or 4
    n_beats: int

    @property
    def meter(self) -> str:
        return "3/4" if self.p_triple >= 0.5 else "4/4"


def _ac(x: np.ndarray, lag: int) -> float:
    """Normalised autocorrelation of a beat-synchronous sequence at `lag` beats."""
    x = x - x.mean()
    denom = float((x * x).sum())
    if len(x) <= lag or denom <= 1e-12:
        return 0.0
    return float((x[:-lag] * x[lag:]).sum() / denom)


def analyze(y: np.ndarray, sr: int) -> Rhythm:
    """Estimate tempo and whether the meter is triple.

    Meter: downbeats are marked by louder onsets, more bass, and chord changes. We take one value
    per beat for each of those cues and ask whether the pattern repeats every 3 beats or every 4
    (normalised autocorrelation at lag 3 vs lag 4). Duple and quadruple music both repeat at 4.
    This is a heuristic: it is solid on clear accents and weaker on legato ballads.
    """
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    if len(beats) < 9:
        return Rhythm(bpm=bpm, p_triple=0.5, n_beats=len(beats))

    S = np.abs(librosa.stft(y, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR)
    bass = S[freqs < 200].sum(0)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=SR)
    chroma_flux = np.r_[0.0, np.abs(np.diff(chroma, axis=1)).sum(0)]

    cues = []
    for env in (onset, bass, chroma_flux):
        n = min(len(env), S.shape[1])
        sync = librosa.util.sync(env[None, :n], beats, aggregate=np.mean)[0][1:-1]
        if len(sync) > 8 and sync.std() > 0:
            cues.append(_ac((sync - sync.mean()) / sync.std(), 3) - _ac((sync - sync.mean()) / sync.std(), 4))
    margin = float(np.mean(cues)) if cues else 0.0
    return Rhythm(bpm=bpm, p_triple=float(1 / (1 + np.exp(-8 * margin))), n_beats=len(beats))


class TempoRuleClassifier:
    """Ranks dances by how well the estimated tempo and meter match the competition rules.

    For each dance, score = tempo fit (Gaussian in log-tempo around the dance's BPM range, with
    half and double tempo allowed at a penalty) x meter fit (triple for the waltzes, otherwise
    duple/quadruple). Scores are normalised into a distribution."""

    sigma = 0.05          # octaves; about 3.5% of tempo
    octave_penalty = 0.4  # weight for the half/double-tempo readings

    def __init__(self, styles: list[str] = STYLES):
        self.styles = list(styles)

    def scores(self, r: Rhythm) -> np.ndarray:
        out = []
        for name in self.styles:
            d = DANCES[name]
            lo, hi = np.log2(d.bpm_range)
            fit = 0.0
            for k, w in ((1.0, 1.0), (0.5, self.octave_penalty), (2.0, self.octave_penalty)):
                lb = np.log2(r.bpm * k)
                dist = 0.0 if lo <= lb <= hi else min(abs(lb - lo), abs(lb - hi))
                fit = max(fit, w * np.exp(-0.5 * (dist / self.sigma) ** 2))
            meter = r.p_triple if d.meter == 3 else 1 - r.p_triple
            out.append(fit * (0.2 + 0.8 * meter) + 1e-4)
        s = np.array(out)
        return s / s.sum()
