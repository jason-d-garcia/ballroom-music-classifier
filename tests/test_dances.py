import pytest

from ballroom.common import STYLES
from ballroom.dances import (DANCES, LATIN, RHYTHM, SMOOTH, STANDARD, dances_in, describe_fit,
                             style_fit, to_mpm)


def test_every_model_class_has_tempo_rules():
    assert set(DANCES) == set(STYLES)


def test_style_membership():
    assert sorted(dances_in(STANDARD)) == sorted(["Waltz", "Tango", "Viennese waltz", "Foxtrot", "Quickstep"])
    assert sorted(dances_in(LATIN)) == sorted(["Cha-cha", "Samba", "Rumba", "Paso doble", "Jive"])
    assert sorted(dances_in(SMOOTH)) == sorted(["Waltz", "Tango", "Viennese waltz", "Foxtrot"])
    assert DANCES["Waltz"].categories == [STANDARD, SMOOTH]
    assert DANCES["Rumba"].categories == [LATIN]  # American Rhythm is reported but not a target style


@pytest.mark.parametrize("bpm,dance,mpm", [
    (87, "Waltz", 29),
    (174, "Waltz", 29),            # tracker doubled the tempo
    (89, "Viennese waltz", 59.3),  # tracker halved it
    (176, "Jive", 44),
    (100, "Samba", 50),            # 2/4
    (122, "Paso doble", 61),
    (102, "Quickstep", 51),
])
def test_to_mpm_fixes_octave_errors(bpm, dance, mpm):
    assert to_mpm(bpm, dance) == pytest.approx(mpm, abs=0.1)


def fits(mpm, dance):
    return {(f["category"], f["dance"]): f["fit"] for f in style_fit(mpm, dance)}


def test_viennese_waltz_separates_smooth_and_standard():
    assert fits(59, "Viennese waltz") == {(STANDARD, "Viennese waltz"): "fits", (SMOOTH, "Viennese waltz"): "fast"}
    assert fits(53.5, "Viennese waltz") == {(STANDARD, "Viennese waltz"): "slow", (SMOOTH, "Viennese waltz"): "fits"}


def test_foxtrot_uses_slow_foxtrot_name_for_standard():
    f = fits(31, "Foxtrot")
    assert f[(STANDARD, "Slow Foxtrot")] == "close"
    assert f[(SMOOTH, "Foxtrot")] == "fits"


def test_rumba_reports_bolero_and_american_rumba():
    f = fits(25.5, "Rumba")
    assert f[(LATIN, "Rumba")] == "fits"
    assert f[(RHYTHM, "Bolero")] == "fits"
    assert f[(RHYTHM, "Rumba")] == "slow"


def test_describe_fit_is_readable():
    s = describe_fit(style_fit(29, "Waltz"))
    assert s == "International Standard (Waltz 28-30): fits; American Smooth (Waltz 28-30): fits"
