"""Competition tempo and meter rules per dance, from the project's tempo atlas
(reference/ballroom-tempo-atlas.html: WDSF for International, USA Dance / NDCA for American).

MPM = bars per minute, BPM = beats per minute = MPM * beats_per_bar.
"""
from __future__ import annotations

STD, SMOOTH, LATIN, RHYTHM = ("International Standard", "American Smooth",
                              "International Latin", "American Rhythm")
BODY = {STD: "WDSF", SMOOTH: "USA Dance", LATIN: "WDSF", RHYTHM: "USA Dance"}

# (family, dance as in common.STYLES, min MPM, max MPM, NDCA reference MPM or None)
RULES = [
    (STD, "Waltz", 28, 30, None), (STD, "Tango", 31, 33, None),
    (STD, "Viennese waltz", 58, 60, None), (STD, "Foxtrot", 28, 30, None),  # Slow Foxtrot
    (STD, "Quickstep", 50, 52, None),
    (SMOOTH, "Waltz", 28, 30, 30), (SMOOTH, "Tango", 30, 32, 30),
    (SMOOTH, "Foxtrot", 30, 32, 30), (SMOOTH, "Viennese waltz", 53, 54, 53),
    (LATIN, "Samba", 50, 52, None), (LATIN, "Cha-cha", 30, 32, None),
    (LATIN, "Rumba", 25, 27, None), (LATIN, "Paso doble", 60, 62, None),
    (LATIN, "Jive", 42, 44, None),
    (RHYTHM, "Cha-cha", 30, 30, 30), (RHYTHM, "Rumba", 32, 36, 31),
    # Bolero, East Coast Swing and Mambo are in the atlas but have no training clips.
]

# Beats per bar used for MPM (matches the atlas) and the usual written time signature.
METER = {
    "Waltz": (3, "3/4"), "Viennese waltz": (3, "3/4"), "Tango": (4, "2/4 or 4/4"),
    "Foxtrot": (4, "4/4"), "Quickstep": (4, "4/4"), "Cha-cha": (4, "4/4"),
    "Rumba": (4, "4/4"), "Jive": (4, "4/4"), "Samba": (2, "2/4"), "Paso doble": (2, "2/4"),
    "Salsa": (4, "4/4"), "West Coast Swing": (4, "4/4"),
}


def families(dance: str) -> list[str]:
    return [f for f, d, *_ in RULES if d == dance]


def tempo_fits(dance: str, mpm: float, tol: float = 2.0) -> list[str]:
    """Families whose competition tempo range for this dance contains mpm (± tol MPM)."""
    return [f for f, d, lo, hi, _ in RULES if d == dance and lo - tol <= mpm <= hi + tol]


def nominal_mpm(dance: str) -> tuple[float, float] | None:
    """Widest competition MPM range for the dance across families."""
    r = [(lo, hi) for _, d, lo, hi, _ in RULES if d == dance]
    return (min(a for a, _ in r), max(b for _, b in r)) if r else None
