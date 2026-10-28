#!/usr/bin/env bash
# Full pipeline after the Extended Ballroom audio is in data/ (see manifest.py docstring).
set -euo pipefail
cd "$(dirname "$0")"
XML=${XML:-$(ls data/*.xml | head -1)}
python manifest.py --xml "$XML" --audio data/
python embed.py --model mert                  # ~4.1k clips; the slow step
python make_split.py --features features/mert.npz
python baselines.py
python train_probe.py --features features/mert.npz --layer_sweep
python embed.py --model clap
python zeroshot_clap.py
python train_probe.py --features features/clap.npz
python - <<'EOF'
import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob("runs/*_metrics.json"))]
print(f"{'model':28s} acc   top2  macroF1")
for r in sorted(rows, key=lambda r: r["accuracy"]):
    print(f"{r['model']:28s} {r['accuracy']:.3f} {r['top2_accuracy']:.3f} {r['macro_f1']:.3f}")
EOF
