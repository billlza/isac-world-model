#!/bin/bash
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=04:00:00
#SBATCH --job-name=isac-capacity
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Submit through the capacity-study launcher}"
: "${ISAC_PROJECT_ROOT:?Missing study checkout}" "${ISAC_PYTHON:?Missing verified school Python}"
: "${ISAC_RUNS_DIR:?Missing study runs directory}" "${ISAC_CAPACITY_PIN:?Missing study revision}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_JOB_ID:-}" ]]
[[ "$ISAC_CAPACITY_PIN" =~ ^[0-9a-f]{40}$ && "$ISAC_PROJECT_ROOT" = /* && "$ISAC_RUNS_DIR" = /* ]]
[[ "$ISAC_PYTHON" = /* && -x "$ISAC_PYTHON" ]]

cd -- "$ISAC_PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$ISAC_CAPACITY_PIN" ]]
git diff --quiet HEAD --
[[ -z "$(git ls-files --others --exclude-standard -- src scripts '*.py')" ]]
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LC_ALL=C
out="$ISAC_RUNS_DIR/capacity-$SLURM_JOB_ID"
mkdir -p -- "$ISAC_RUNS_DIR"
mkdir -- "$out"  # Preserve completed and interrupted runs; never restart in place.
stage=preflight

write_status() {
  # Values interpolated into JSON are fixed labels, validated identifiers, or UTC text.
  printf '{"schema_version":1,"status":"%s","stage":"%s","slurm_job_id":"%s","git_commit":"%s","updated_utc":"%s","exit_code":%s}\n' \
    "$1" "$stage" "$SLURM_JOB_ID" "$ISAC_CAPACITY_PIN" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" > "$out/.status.tmp"
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
printf 'Capacity/data study outputs: %s\n' "$out"
"$ISAC_PYTHON" scripts/hamilton_preflight.py --out "$out/preflight.json"
"$ISAC_PYTHON" - "$out/preflight.json" <<'PY'
import json, os, platform, sys
import numpy, torch
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
versions = (platform.python_version(), numpy.__version__, str(torch.__version__), torch.version.cuda)
if versions != ("3.13.9", "2.3.4", "2.9.0+cu128", "12.8"):
    raise SystemExit(f"Unexpected compute-node runtime: {versions!r}")
if (data["status"] != "passed" or data["slurm"]["job_id"] != os.environ["SLURM_JOB_ID"]
        or data["git_commit"] != os.environ["ISAC_CAPACITY_PIN"]):
    raise SystemExit("Current job/revision preflight mismatch")
PY

stage=study
write_status running null
"$ISAC_PYTHON" scripts/capacity_data_study.py --out "$out/study" --device cuda --profile formal
[[ -s "$out/study/study.json" ]]

stage=package
write_status running null
"$ISAC_PYTHON" scripts/package_capacity_study.py --run-dir "$out" --out "$out/isac-capacity-$SLURM_JOB_ID.tar.gz"
stage=done
write_status completed 0
printf 'Capacity/data study completed. Evidence: %s\n' "$out"
