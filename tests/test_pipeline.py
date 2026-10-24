"""The training pipeline on tiny synthetic data (the real run needs the dataset and MERT)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from ballroom.common import STYLES, norm_text, raw_to_style, top_k_accuracy


def test_label_mapping_merges_slow_waltz_and_drops_small_classes():
    assert raw_to_style("Slow Waltz") == "Waltz"
    assert raw_to_style("VienneseWaltz") == "Viennese waltz"
    assert raw_to_style("ChaCha") == "Cha-cha"
    assert raw_to_style("Salsa") is None
    assert len(STYLES) == 10


def test_norm_text_ignores_versions():
    assert norm_text("Moon River (Waltz) - Karaoke Version") == norm_text("moon river")


def test_top_k_accuracy():
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    assert top_k_accuracy(p, np.array([1, 0]), 1) == 0.0
    assert top_k_accuracy(p, np.array([1, 0]), 2) == 0.5


XML = """<?xml version="1.0"?>
<extendedballroom>
  <Waltz>
    <song id="100" bpm="87"><title>Moon River</title><artist>A</artist></song>
    <song id="101" bpm="86"><title>Moon River (karaoke)</title><artist>A</artist><karaoke id="100"/></song>
  </Waltz>
  <Salsa><song id="200" bpm="180"><title>S</title><artist>B</artist></song></Salsa>
  <Jive><song id="300" bpm="176"><title>J</title><artist>C</artist></song></Jive>
</extendedballroom>"""


def test_manifest_parses_xml_and_duplicates(tmp_path, monkeypatch):
    from ballroom import manifest
    (tmp_path / "x.xml").write_text(XML)
    for i in ("100", "101", "200", "300"):
        (tmp_path / f"{i}.mp3").write_bytes(b"")
    out = tmp_path / "manifest.csv"
    monkeypatch.setattr(sys, "argv", ["manifest", "--xml", str(tmp_path / "x.xml"),
                                      "--audio", str(tmp_path), "--out", str(out)])
    manifest.main()
    df = pd.read_csv(out, dtype={"id": str, "dup_of": str}).set_index("id")
    assert sorted(df.index) == ["100", "101", "300"]   # salsa dropped
    assert df.loc["101", "style"] == "Waltz" and df.loc["101", "dup_of"] == "100"
    assert df.loc["300", "bpm"] == 176


def synthetic_set(n_per=30, layers=3, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    ids, styles, X = [], [], []
    for c, s in enumerate(STYLES):
        for k in range(n_per):
            v = rng.normal(0, 1, (layers, dim))
            v[:, c % dim] += 4  # separable class signal
            ids.append(f"{c}_{k}"); styles.append(s); X.append(v)
    return ids, styles, np.stack(X).astype(np.float16)


def test_split_keeps_duplicates_together_and_probe_learns(tmp_path, monkeypatch):
    from ballroom import make_split, train_probe
    from ballroom.metrics import evaluate

    ids, styles, X = synthetic_set()
    titles = [f"song {i}" for i in ids]
    titles[1] = titles[0]  # a repeated song in the same style
    pd.DataFrame({"id": ids, "style": styles, "title": titles, "artist": "x", "dup_of": None,
                  "path": ""}).to_csv(tmp_path / "manifest.csv", index=False)
    np.savez(tmp_path / "f.npz", X=X, ids=np.array(ids))
    monkeypatch.setattr(sys, "argv", ["make_split", "--manifest", str(tmp_path / "manifest.csv"),
                                      "--features", str(tmp_path / "f.npz"), "--out", str(tmp_path / "split.csv")])
    make_split.main()
    sp = pd.read_csv(tmp_path / "split.csv", dtype={"id": str}).set_index("id")
    assert sp.loc["0_0", "split"] == sp.loc["0_1", "split"]
    assert set(sp["split"]) == {"train", "val", "test"}
    assert sp.groupby("split")["style"].nunique().min() >= 8  # stratified

    torch = pytest.importorskip("torch")
    parts = train_probe.load(tmp_path / "f.npz", tmp_path / "split.csv")
    model, mu, sd, val_f1, _ = train_probe.train(parts, epochs=60, lr=3e-3, wd=1e-2, hidden=32,
                                                 dropout=0.1, batch=32, dev=torch.device("cpu"))
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor((parts["test"][0] - mu) / sd, dtype=torch.float32)), 1).numpy()
    res = evaluate("synthetic", proba, parts["test"][1], tmp_path)
    assert res["accuracy"] > 0.9
    assert json.loads((tmp_path / "synthetic_metrics.json").read_text())["model"] == "synthetic"
    assert (tmp_path / "synthetic_confusion.csv").exists()
