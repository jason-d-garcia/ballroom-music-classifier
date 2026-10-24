import csv
import json

import numpy as np
import pytest
import soundfile as sf

from ballroom import classify
from ballroom.common import STYLES
from synth import SR, dance_track


@pytest.fixture
def waltz_wav(tmp_path):
    path = tmp_path / "waltz.wav"
    sf.write(path, dance_track(87, 3, seconds=20), SR)
    return path


def test_cli_tempo_model_text_and_csv(waltz_wav, tmp_path, capsys):
    out = tmp_path / "tagged.csv"
    assert classify.main([str(waltz_wav), "--model", "tempo", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "1. Waltz" in text
    assert "International Standard (Waltz 28-30): fits" in text
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["dance1"] == "Waltz"
    assert rows[0]["meter"] == "3/4"
    assert "American Smooth (Waltz)" in rows[0]["fits1"]


def test_cli_json_and_missing_files(waltz_wav, tmp_path, capsys):
    classify.main([str(waltz_wav), str(tmp_path / "nope.mp3"), "--model", "tempo", "--json", "--top-k", "5"])
    res = json.loads(capsys.readouterr().out)
    assert len(res[0]["dances"]) == 5
    d = res[0]["dances"][0]
    assert d["dance"] == "Waltz" and 28 <= d["mpm"] <= 30
    assert d["categories"] == ["International Standard", "American Smooth"]
    assert res[1]["status"] == "not found"


def test_cli_folder_input(waltz_wav, capsys):
    classify.main([str(waltz_wav.parent), "--model", "tempo", "--json"])
    assert len(json.loads(capsys.readouterr().out)) == 1


def test_auto_falls_back_to_tempo_without_a_probe(tmp_path):
    assert isinstance(classify.load_model("auto", tmp_path / "missing.pt"), classify.TempoModel)
    with pytest.raises(SystemExit):
        classify.load_model("mert", tmp_path / "missing.pt")


class FakeEmbedder:
    """Stands in for MERT: [layers, dim] features from simple audio statistics."""
    sr = 24000

    def __call__(self, windows):
        w = np.stack(windows)
        f = np.array([w.mean(), w.std(), np.abs(w).max(), (w ** 2).mean()])
        return np.tile(f, (3, 2))  # 3 layers x 8 dims


def test_probe_model_with_saved_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from ballroom.train_probe import LayerMixProbe

    torch.manual_seed(0)
    probe = LayerMixProbe(3, 8, len(STYLES), hidden=16, p=0.0)
    ck = tmp_path / "probe.pt"
    torch.save({"state_dict": probe.state_dict(), "mu": np.zeros((1, 3, 8)), "sd": np.ones((1, 3, 8)),
                "styles": STYLES, "shape": (3, 8), "hidden": 16, "dropout": 0.0, "features": "mert"}, ck)
    model = classify.ProbeModel(ck, embedder_factory=FakeEmbedder)
    res = classify.classify_audio(dance_track(116, 4, seconds=12, sr=24000), 24000, model, top_k=10)
    assert res["model"] == "mert"
    probs = [d["probability"] for d in res["dances"]]
    assert sorted(d["dance"] for d in res["dances"]) == sorted(STYLES)
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=0.01)
