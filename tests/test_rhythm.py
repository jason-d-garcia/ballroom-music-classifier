import numpy as np
import pytest

from ballroom.common import STYLES
from ballroom.rhythm import SR, Rhythm, TempoRuleClassifier, analyze
from synth import dance_track


def octave_close(est, true, tol=0.04):
    return any(abs(est * k - true) / true < tol for k in (0.5, 1, 2))


@pytest.mark.parametrize("bpm,bpb", [(87, 3), (116, 4), (128, 4), (172, 4), (100, 2)])
def test_tempo_and_meter_on_synthetic_tracks(bpm, bpb):
    r = analyze(dance_track(bpm, bpb), SR)
    assert octave_close(r.bpm, bpm), r
    assert r.meter == ("3/4" if bpb == 3 else "4/4"), r


def test_analyze_resamples_other_rates():
    r = analyze(dance_track(116, 4, sr=24000), 24000)
    assert octave_close(r.bpm, 116)


def test_short_clip_falls_back_to_unknown_meter():
    r = analyze(dance_track(120, 4, seconds=3), SR)
    assert r.p_triple == 0.5


def top(r):
    p = TempoRuleClassifier().scores(r)
    assert p.sum() == pytest.approx(1.0)
    return STYLES[int(np.argmax(p))]


@pytest.mark.parametrize("bpm,p_triple,dance", [
    (87, 0.95, "Waltz"),
    (177, 0.95, "Viennese waltz"),
    (176, 0.05, "Jive"),             # same BPM as Viennese, but not triple
    (116, 0.05, "Foxtrot"),
    (204, 0.05, "Quickstep"),
    (89, 0.95, "Waltz"),
])
def test_tempo_rules_rank_clear_cases(bpm, p_triple, dance):
    assert top(Rhythm(bpm=bpm, p_triple=p_triple, n_beats=40)) == dance


def test_meter_shifts_probability_toward_the_waltzes():
    clf = TempoRuleClassifier()
    duple = clf.scores(Rhythm(bpm=88, p_triple=0.05, n_beats=40))
    triple = clf.scores(Rhythm(bpm=88, p_triple=0.95, n_beats=40))
    w = STYLES.index("Waltz")
    assert triple[w] > duple[w]
    assert STYLES[int(np.argmax(triple))] == "Waltz"
