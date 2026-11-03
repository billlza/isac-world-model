#!/bin/bash
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=01:00:00
#SBATCH --job-name=isac-memory
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Submit through the follow-up launcher}"
: "${ISAC_PROJECT_ROOT:?Missing study checkout}" "${ISAC_PYTHON:?Missing verified school Python}"
: "${ISAC_RUNS_DIR:?Missing study runs directory}" "${ISAC_FOLLOWUP_PIN:?Missing study revision}"
: "${ISAC_ROUND2_DIR:?Missing original Hamilton array directory}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_JOB_ID:-}" ]]
[[ "$ISAC_FOLLOWUP_PIN" =~ ^[0-9a-f]{40}$ && "$ISAC_PROJECT_ROOT" = /* && "$ISAC_RUNS_DIR" = /* ]]
[[ "$ISAC_PYTHON" = /* && -x "$ISAC_PYTHON" && "$ISAC_ROUND2_DIR" = /* && -d "$ISAC_ROUND2_DIR" ]]

cd -- "$ISAC_PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$ISAC_FOLLOWUP_PIN" ]]
git diff --quiet HEAD --
[[ -z "$(git ls-files --others --exclude-standard -- src scripts)" ]]
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
out="$ISAC_RUNS_DIR/followup-$SLURM_JOB_ID"
mkdir -p -- "$ISAC_RUNS_DIR"
mkdir -- "$out"  # Never overwrite earlier or interrupted evidence.
stage=preflight

write_status() {
  # All interpolated JSON values are fixed labels or validated decimal/hex identifiers.
  printf '{"schema_version":1,"status":"%s","stage":"%s","slurm_job_id":"%s","git_commit":"%s","source_array":"19341792","exit_code":%s}\n' \
    "$1" "$stage" "$SLURM_JOB_ID" "$ISAC_FOLLOWUP_PIN" "$2" > "$out/.status.tmp"
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
printf 'Source array: %s\nFollow-up outputs: %s\n' "$ISAC_ROUND2_DIR" "$out"
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
        or data["git_commit"] != os.environ["ISAC_FOLLOWUP_PIN"]):
    raise SystemExit("Current job/revision preflight mismatch")
PY

stage=verify-prior
write_status running null
"$ISAC_PYTHON" scripts/verify_round2_inputs.py --array-dir "$ISAC_ROUND2_DIR" --out "$out/prior-inputs.json"

stage=frozen
write_status running null
"$ISAC_PYTHON" scripts/frozen_recovery.py --array-dir "$ISAC_ROUND2_DIR" --out "$out/frozen.json" --device cuda

stage=crossed
write_status running null
"$ISAC_PYTHON" scripts/crossed_seed_study.py --out "$out/crossed" --device cuda
[[ -s "$out/prior-inputs.json" && -s "$out/frozen.json" && -s "$out/crossed/crossed.json" ]]
stage=done
write_status completed 0
printf 'Follow-up completed. Evidence: %s\n' "$out"
