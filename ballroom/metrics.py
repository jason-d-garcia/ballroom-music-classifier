"""Evaluation shared by every model so the numbers are comparable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .common import STYLES, top_k_accuracy


def evaluate(name: str, proba: np.ndarray, y: np.ndarray, out_dir: Path, extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = proba.argmax(1)
    labels = list(range(len(STYLES)))
    res = {
        "model": name,
        "n_test": int(len(y)),
        "accuracy": round(accuracy_score(y, pred), 4),
        "top2_accuracy": round(top_k_accuracy(proba, y, 2), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro", labels=labels, zero_division=0), 4),
        "per_class_f1": dict(zip(STYLES, np.round(
            f1_score(y, pred, average=None, labels=labels, zero_division=0), 3).tolist())),
        **(extra or {}),
    }
    cm = confusion_matrix(y, pred, labels=labels)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(res, indent=2))
    np.savetxt(out_dir / f"{name}_confusion.csv", cm, fmt="%d", delimiter=",",
               header=",".join(STYLES), comments="")
    _plot(cm, name, out_dir / f"{name}_confusion.png")
    print(f"[{name}] acc {res['accuracy']:.3f}  top-2 {res['top2_accuracy']:.3f}  "
          f"bal-acc {res['balanced_accuracy']:.3f}  macro-F1 {res['macro_f1']:.3f}  (n={len(y)})")
    return res


def _plot(cm: np.ndarray, title: str, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(STYLES)), STYLES, rotation=45, ha="right")
    ax.set_yticks(range(len(STYLES)), STYLES)
    for i in range(len(STYLES)):
        for j in range(len(STYLES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
