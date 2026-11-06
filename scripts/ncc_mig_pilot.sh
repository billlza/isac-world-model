#!/bin/bash
#SBATCH --account=ncc_users
#SBATCH --partition=res-gpu-small
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --time=00:15:00
#SBATCH --job-name=isac-ncc-mig
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Submit with the NCC pilot launcher}"
: "${ISAC_PROJECT_ROOT:?Missing source checkout}" "${ISAC_NCC_PIN:?Missing source pin}"
: "${ISAC_RUNS_DIR:?Missing output root}" "${ISAC_NCC_RUNTIME_ROOT:?Missing NCC runtime root}"
: "${ISAC_NCC_SETUP_JOB_ID:?Missing runtime setup job}"
if [[ ! "$SLURM_JOB_ID" =~ ^[0-9]+$ || -n "${SLURM_ARRAY_JOB_ID:-}" ||
      ! "$ISAC_NCC_SETUP_JOB_ID" =~ ^[0-9]+$ || ! "$ISAC_NCC_PIN" =~ ^[0-9a-f]{40}$ ]]; then
    echo 'Invalid Slurm identity or source pin.' >&2
    exit 2
fi
for directory in "$ISAC_PROJECT_ROOT" "$ISAC_RUNS_DIR" "$ISAC_NCC_RUNTIME_ROOT"; do
    if [[ "$directory" != /* ]]; then
        echo 'NCC source, outputs and runtime must use absolute paths.' >&2
        exit 2
    fi
done
out="$ISAC_RUNS_DIR/ncc-mig-$SLURM_JOB_ID"
mkdir -p -- "$ISAC_RUNS_DIR"
mkdir -- "$out"
stage=runtime
write_status() {
    printf '{"schema_version":1,"status":"%s","stage":"%s","slurm_job_id":"%s","setup_job_id":"%s","git_commit":"%s","updated_utc":"%s","exit_code":%s}\n' \
        "$1" "$stage" "$SLURM_JOB_ID" "$ISAC_NCC_SETUP_JOB_ID" "$ISAC_NCC_PIN" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" > "$out/.status.tmp"
    mv -- "$out/.status.tmp" "$out/status.json"
}
finish() {
    rc=$?
    trap - EXIT
    if [[ "$rc" != 0 ]]; then
        write_status failed "$rc"
        printf 'NCC stage %s failed (exit %s). Evidence: %s\n' "$stage" "$rc" "$out" >&2
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_status running null
cd -- "$ISAC_PROJECT_ROOT"
ncc_python="$ISAC_NCC_RUNTIME_ROOT/venv/bin/python"
if [[ ! -x "$ncc_python" || ! -s "$ISAC_NCC_RUNTIME_ROOT/runtime.json" ]]; then
    echo 'The dependent CPU job did not create a usable NCC runtime.' >&2
    exit 1
fi
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LC_ALL=C
unset PYTHONHOME
"$ncc_python" -I - "$ISAC_NCC_RUNTIME_ROOT/runtime.json" <<'PY'
import json
import os
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
if data.get("status") != "ready" or data.get("git_commit") != os.environ["ISAC_NCC_PIN"]:
    raise SystemExit("Runtime setup did not pass for the submitted source")
if data.get("slurm_setup_job_id") != os.environ["ISAC_NCC_SETUP_JOB_ID"]:
    raise SystemExit("Runtime evidence belongs to another setup job")
if Path(sys.prefix).resolve() != (Path(os.environ["ISAC_NCC_RUNTIME_ROOT"]) / "venv").resolve():
    raise SystemExit("Interpreter is outside the submitted NCC runtime")
PY
stage=preflight
write_status running null
"$ncc_python" scripts/ncc_mig_preflight.py --out "$out/preflight.json" \
    --expected-commit "$ISAC_NCC_PIN" --runtime-manifest "$ISAC_NCC_RUNTIME_ROOT/runtime.json"
stage=model_check
write_status running null
"$ncc_python" scripts/ncc_mig_pilot.py --out "$out/model-check" \
    --preflight "$out/preflight.json" --expected-commit "$ISAC_NCC_PIN"
stage=done
write_status completed 0
printf 'NCC A512 MIG development check completed. Evidence: %s\n' "$out"
