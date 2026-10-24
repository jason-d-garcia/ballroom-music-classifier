"""Which ballroom styles each dance belongs to, and the competition tempo rules for each.

The model predicts the dance (Waltz, Tango, ...), not the style: Smooth and Standard Waltz,
Tango and Foxtrot use the same recordings, so audio can't tell them apart. The style fit is
derived afterwards from the estimated tempo in MPM (bars per minute), which is how WDSF, NDCA
and USA Dance write their tempo rules.

Sources: WDSF tempos for International Standard and Latin; USA Dance ranges for American
Smooth and Rhythm (via Wikipedia's "List of DanceSport dances"). NDCA reference tempos come from
a secondary chart and are not verified against the current rulebook.

`bpb` (beats per bar) follows the Ballroom datasets' annotation convention: Samba and Paso doble
are 2/4, Tango is annotated near 128 BPM (4 pulses per bar), the waltzes are 3/4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

STANDARD = "International Standard"
SMOOTH = "American Smooth"
LATIN = "International Latin"
RHYTHM = "American Rhythm"
CATEGORIES = (STANDARD, SMOOTH, LATIN)  # the project's three target styles


@dataclass(frozen=True)
class TempoRule:
    category: str
    body: str
    mpm: tuple[float, float]
    name: str | None = None  # when the dance goes by another name in this style
    ndca_ref: float | None = None


@dataclass(frozen=True)
class Dance:
    name: str  # matches the model's class names in common.STYLES
    bpb: int
    rules: tuple[TempoRule, ...]

    @property
    def categories(self) -> list[str]:
        return [r.category for r in self.rules if r.category in CATEGORIES]

    @property
    def bpm_range(self) -> tuple[float, float]:
        """Widest competition range across the three target styles, in beats per minute.

        American Rhythm readings (faster American Rumba, Bolero) are reported by style_fit but
        left out here, or Rumba's range would swallow Cha-cha's."""
        rules = [r for r in self.rules if r.category in CATEGORIES]
        return (min(r.mpm[0] for r in rules) * self.bpb, max(r.mpm[1] for r in rules) * self.bpb)

    @property
    def meter(self) -> int:
        return 3 if self.bpb == 3 else 4  # 2/4 dances still read as duple, not triple


DANCES: dict[str, Dance] = {d.name: d for d in [
    Dance("Waltz", 3, (TempoRule(STANDARD, "WDSF", (28, 30)),
                       TempoRule(SMOOTH, "USA Dance", (28, 30), ndca_ref=30))),
    Dance("Tango", 4, (TempoRule(STANDARD, "WDSF", (31, 33)),
                       TempoRule(SMOOTH, "USA Dance", (30, 32), ndca_ref=30))),
    Dance("Viennese waltz", 3, (TempoRule(STANDARD, "WDSF", (58, 60)),
                                TempoRule(SMOOTH, "USA Dance", (53, 54), ndca_ref=53))),
    Dance("Foxtrot", 4, (TempoRule(STANDARD, "WDSF", (28, 30), name="Slow Foxtrot"),
                         TempoRule(SMOOTH, "USA Dance", (30, 32), ndca_ref=30))),
    Dance("Quickstep", 4, (TempoRule(STANDARD, "WDSF", (50, 52)),)),
    Dance("Cha-cha", 4, (TempoRule(LATIN, "WDSF", (30, 32)),
                         TempoRule(RHYTHM, "USA Dance", (30, 30), ndca_ref=30))),
    Dance("Samba", 2, (TempoRule(LATIN, "WDSF", (50, 52)),)),
    # The same slow ballad often suits International Rumba, Bolero, or (faster) American Rumba.
    Dance("Rumba", 4, (TempoRule(LATIN, "WDSF", (25, 27)),
                       TempoRule(RHYTHM, "USA Dance", (32, 36), ndca_ref=31),
                       TempoRule(RHYTHM, "USA Dance", (24, 26), name="Bolero", ndca_ref=23))),
    Dance("Paso doble", 2, (TempoRule(LATIN, "WDSF", (60, 62)),)),
    Dance("Jive", 4, (TempoRule(LATIN, "WDSF", (42, 44)),)),
]}


def dances_in(category: str) -> list[str]:
    return [d.name for d in DANCES.values() if any(r.category == category for r in d.rules)]


def to_mpm(bpm: float, dance: str) -> float:
    """Bars per minute for this dance, after fixing the beat tracker's half/double-tempo errors.

    Beat trackers often report half or double the tempo; we pick the octave (bpm/2, bpm, 2*bpm)
    that lands closest to the dance's competition range."""
    d = DANCES[dance]
    lo, hi = d.bpm_range
    best = min((bpm * k for k in (0.5, 1.0, 2.0)), key=lambda b: _log_dist(b, lo, hi))
    return best / d.bpb


def _log_dist(x: float, lo: float, hi: float) -> float:
    if lo <= x <= hi:
        return 0.0
    return abs(math.log2(x / (lo if x < lo else hi)))


def style_fit(mpm: float, dance: str, tolerance: float = 1.0) -> list[dict]:
    """How well a tempo fits each style's rule for this dance.

    fit is "fits" inside the range, "close" within `tolerance` MPM, else "slow" or "fast"."""
    out = []
    for r in DANCES[dance].rules:
        lo, hi = r.mpm
        if lo <= mpm <= hi:
            fit = "fits"
        elif lo - tolerance <= mpm <= hi + tolerance:
            fit = "close"
        else:
            fit = "slow" if mpm < lo else "fast"
        out.append({"category": r.category, "dance": r.name or dance, "body": r.body,
                    "mpm_range": [lo, hi], "fit": fit})
    return out


def describe_fit(fits: list[dict]) -> str:
    """One line for the terminal, e.g. 'International Standard (Slow Foxtrot 28-30): fits'."""
    def rng(a, b):
        return f"{a:g}" if a == b else f"{a:g}-{b:g}"
    return "; ".join(f"{f['category']} ({f['dance']} {rng(*f['mpm_range'])}): {f['fit']}" for f in fits)
