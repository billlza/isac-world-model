#!/bin/bash
# Small, paired, from-scratch development run. This does not use the study test set.
set -euo pipefail
: "${STUDY_PYTHON:?Set STUDY_PYTHON to the existing project interpreter}"
out=${1:?Supply a new output directory}
mkdir -p "$(dirname "$out")"
mkdir "$out"
export PYTHONPATH="$PWD/src"
for hidden in 96 192; do
  for architecture in structured structured-attention; do
    dest="$out/$architecture-$hidden"
    "$STUDY_PYTHON" -m isac_wm.cli train --device cpu --threads 2 \
      --architecture "$architecture" --hidden "$hidden" --features 48 --heads 4 --memory-size 16 \
      --episodes 24 --steps 64 --seed 19 --batch-size 8 --epochs 2 --patience 0 --out "$dest"
  done
done
"$STUDY_PYTHON" - "$out" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
scores = []
splits = None
for path in sorted(root.glob("*/manifest.json")):
    m = json.loads(path.read_text())
    observed = [m["train_episode_seeds"], m["validation_episode_seeds"]]
    assert m["completed_epochs"] == 2 and m["runtime"]["device"] == "cpu"
    assert len(observed[0]) == 24 and len(observed[1]) == 8
    assert not set(observed[0]).intersection(observed[1])
    if splits is None:
        splits = observed
    assert splits == observed
    history = json.loads((path.parent / "history.json").read_text())
    scores.append({"architecture": m["model"]["architecture"], "hidden": m["model"]["hidden_dim"],
                   "parameters": m["parameters"], "validation_rmse_m_per_epoch":
                   [row["validation_position_rmse_m"] for row in history]})
assert len(scores) == 4
result = {"scope": "CPU development, seed 19; validation current-position RMSE; not the H200 recovery test",
          "train_episodes": 24, "steps": 64, "epochs": 2, "scores": scores}
with (root / "development.json").open("x") as f:
    json.dump(result, f, indent=2, allow_nan=False)
print("STUDY_DEV_SCORES", json.dumps(result, allow_nan=False))
PY
