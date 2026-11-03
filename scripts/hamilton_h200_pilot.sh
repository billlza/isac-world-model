#!/bin/bash
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=00:30:00
#SBATCH --job-name=isac-wm-pilot
#SBATCH --output=slurm-%j.out

# Hamilton assigns CPU/RAM/local disk from the GPU request. Do not add -n/-c/--mem.
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script through sbatch; do not run on a login node}"
: "${ISAC_PYTHON:?Set ISAC_PYTHON to an existing absolute CUDA Python interpreter path}"
if [[ "$ISAC_PYTHON" != /* || ! -x "$ISAC_PYTHON" ]]; then
    printf '%s\n' 'ISAC_PYTHON must be an existing executable absolute path.' >&2
    exit 1
fi
if [[ ! "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]; then
    printf '%s\n' 'Invalid SLURM_JOB_ID.' >&2
    exit 1
fi

# Slurm executes a spooled copy, so BASH_SOURCE does not locate the checkout.
ISAC_PROJECT_ROOT="${ISAC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the project checkout}}"
cd -- "$ISAC_PROJECT_ROOT"
ISAC_PROJECT_ROOT="$PWD"
if [[ ! -f scripts/hamilton_preflight.py || ! -f src/isac_wm/cli.py ]]; then
    printf '%s\n' 'ISAC_PROJECT_ROOT/SLURM_SUBMIT_DIR is not the project checkout.' >&2
    exit 1
fi
export PYTHONPATH="$ISAC_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ISAC_RUNS_DIR="${ISAC_RUNS_DIR:-$ISAC_PROJECT_ROOT/runs}"
ISAC_PILOT_DIR="$ISAC_RUNS_DIR/hamilton-h200-$SLURM_JOB_ID"
mkdir -p -- "$ISAC_RUNS_DIR"
mkdir -- "$ISAC_PILOT_DIR"

# Do not run uv sync/pip here: the project's Linux default lock selects CPU torch.
"$ISAC_PYTHON" scripts/hamilton_preflight.py --out "$ISAC_PILOT_DIR/preflight.json"

"$ISAC_PYTHON" - "$ISAC_PILOT_DIR" <<'PY'
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch

out = Path(sys.argv[1])
preflight = json.loads((out / "preflight.json").read_text())
record = {
    "schema_version": 1,
    "kind": "hamilton-single-h200-pilot",
    "status": "running",
    "git_commit": preflight["git_commit"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "scope": "ToyScene trainer CUDA execution and resume; not the active-observation study",
    "stages": [],
}


def write():
    temporary = out / "pilot.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "pilot.json")


common = [
    "--episodes", "8", "--batch-size", "4", "--seed", "7",
    "--architecture", "structured-attention", "--steps", "16",
    "--device", "cuda", "--threads", "2", "--patience", "0",
]
write()
try:
    if preflight["status"] != "passed" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Single-CUDA-device preflight has not been established")
    for name, target_epochs in (("train", 1), ("resume", 2)):
        stage = {"name": name, "status": "running", "target_epochs": target_epochs}
        record["stages"].append(stage)
        args = ["train", "--out", str(out / name), "--epochs", str(target_epochs), *common]
        if name == "resume":
            args += ["--resume", str(out / "train" / "resume.pt")]
        stage["cli_args"] = args
        write()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        sys.argv = ["isac_wm.cli", *args]
        runpy.run_module("isac_wm.cli", run_name="__main__")
        torch.cuda.synchronize()
        manifest = json.loads((out / name / "manifest.json").read_text())
        if manifest["runtime"]["device"] != "cuda" or manifest["completed_epochs"] != target_epochs:
            raise RuntimeError("Training output did not confirm CUDA and the required total epoch count")
        if name == "resume" and manifest["resume_source"] != str(out / "train" / "resume.pt"):
            raise RuntimeError("Continuation did not confirm the expected resume checkpoint")
        stage.update(
            status="completed",
            wall_seconds=time.perf_counter() - started,
            training_seconds=manifest["training_seconds"],
            completed_epochs=manifest["completed_epochs"],
            resume_source=manifest["resume_source"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        write()
    record["checkpoint_sha256"] = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("train", "resume")
        for path in (out / name / "model.pt", out / name / "resume.pt")
    }
    record["status"] = "completed"
    write()
except BaseException as error:
    record.update(status="failed", failure_type=type(error).__name__, failure=str(error))
    if record["stages"] and record["stages"][-1]["status"] == "running":
        record["stages"][-1]["status"] = "failed"
    write()
    raise
print(json.dumps({"status": "completed", "evidence": str(out / "pilot.json")}))
PY
