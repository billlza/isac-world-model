#!/bin/bash
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=04:00:00
#SBATCH --job-name=isac-optim
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Submit through the width-optimization launcher}"
: "${ISAC_PROJECT_ROOT:?Missing study checkout}" "${ISAC_PYTHON:?Missing verified school Python}"
: "${ISAC_RUNS_DIR:?Missing study runs directory}" "${ISAC_OPTIM_PIN:?Missing study revision}"
historical_run=${ISAC_OPTIM_HISTORICAL_RUN:-"$HOME/isac-scale-249407dbbe40/runs/scale-19404810"}
if [[ ! "$SLURM_JOB_ID" =~ ^[0-9]+$ || -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    echo 'A single Slurm job is required.' >&2
    exit 2
fi
if [[ ! "$ISAC_OPTIM_PIN" =~ ^[0-9a-f]{40}$ || "$ISAC_PROJECT_ROOT" != /* || "$ISAC_RUNS_DIR" != /* || "$historical_run" != /* ]]; then
    echo 'Invalid study revision or absolute directory.' >&2
    exit 2
fi
if [[ "$ISAC_PYTHON" != /* || ! -x "$ISAC_PYTHON" ]]; then
    echo 'Verified school Python is unavailable.' >&2
    exit 2
fi
if [[ ! -d "$ISAC_PROJECT_ROOT" || -L "$ISAC_PROJECT_ROOT" || -L "$ISAC_RUNS_DIR" ]]; then
    echo 'Expected regular study directories; existing paths preserved.' >&2
    exit 2
fi
out="$ISAC_RUNS_DIR/optim-$SLURM_JOB_ID"
mkdir -p -- "$ISAC_RUNS_DIR"
if ! mkdir -- "$out"; then
    echo "Output directory already exists or cannot be reserved: $out" >&2
    exit 1
fi
stage=preflight
write_status() {
    printf '{"schema_version":1,"status":"%s","stage":"%s","slurm_job_id":"%s","git_commit":"%s","updated_utc":"%s","exit_code":%s}\n' \
        "$1" "$stage" "$SLURM_JOB_ID" "$ISAC_OPTIM_PIN" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" > "$out/.status.tmp"
    mv -- "$out/.status.tmp" "$out/status.json"
}
finish() {
    rc=$?
    trap - EXIT
    if [[ "$rc" != 0 ]]; then
        write_status failed "$rc"
        printf 'Stage %s failed (exit %s). Evidence retained: %s\n' "$stage" "$rc" "$out" >&2
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_status running null

cd -- "$ISAC_PROJECT_ROOT"
check_source() {
    local untracked
    if [[ "$(git rev-parse HEAD)" != "$ISAC_OPTIM_PIN" ]]; then
        echo 'Checkout differs from the submitted source revision.' >&2
        return 1
    fi
    if ! git diff --quiet HEAD --; then
        echo 'Tracked source changed after submission.' >&2
        return 1
    fi
    if ! untracked=$(git ls-files --others --exclude-standard -- src scripts '*.py'); then
        echo 'Could not check untracked source.' >&2
        return 1
    fi
    if [[ -n "$untracked" ]]; then
        echo 'Untracked source could shadow the pinned implementation.' >&2
        return 1
    fi
}
check_source
if [[ ! -d "$historical_run" || -L "$historical_run" ]]; then
    echo 'The registered Phase C source run is missing or is a symlink; existing evidence is required.' >&2
    exit 1
fi
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LC_ALL=C
"$ISAC_PYTHON" scripts/hamilton_preflight.py --out "$out/preflight.json"
"$ISAC_PYTHON" - "$out/preflight.json" <<'PY'
import json
import os
import sys
import numpy
import torch

observed = (sys.version_info[:3], numpy.__version__, str(torch.__version__), torch.version.cuda)
expected = ((3, 13, 9), "2.3.4", "2.9.0+cu128", "12.8")
if observed != expected:
    raise SystemExit(f"School runtime changed: {observed!r}; expected {expected!r}")
with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
if data.get("status") != "passed" or data.get("slurm", {}).get("job_id") != os.environ["SLURM_JOB_ID"]:
    raise SystemExit("Preflight did not pass for this job")
if data.get("git_commit") != os.environ["ISAC_OPTIM_PIN"]:
    raise SystemExit("Preflight source revision differs from submitted revision")
PY
stage=study
write_status running null
"$ISAC_PYTHON" scripts/width_optimization_study.py --out "$out/study" --device cuda --profile formal \
    --historical-run "$historical_run"
if [[ ! -s "$out/study/study.json" ]]; then
    echo 'Training did not produce a study record.' >&2
    exit 1
fi
stage=verify-source
write_status running null
check_source
stage=package
write_status running null
"$ISAC_PYTHON" scripts/package_width_optimization_study.py --run-dir "$out" --out "$out/isac-optim-$SLURM_JOB_ID.tar.gz"
stage=verify-source-final
write_status running null
check_source
stage=done
write_status completed 0
printf 'Phase D width-optimization and checkpoint-replay study completed. Evidence: %s\n' "$out"
