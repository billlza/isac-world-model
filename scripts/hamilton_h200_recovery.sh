#!/bin/bash
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --job-name=isac-recovery
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail
: "${SLURM_ARRAY_JOB_ID:?submit the study array through sbatch}" "${SLURM_ARRAY_TASK_ID:?missing array task}"
: "${ISAC_PROJECT_ROOT:?missing source checkout}" "${ISAC_RUNS_DIR:?missing output root}" "${ISAC_PYTHON:?missing school Python}"
[[ "$SLURM_ARRAY_JOB_ID" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" =~ ^[0-2]$ && "$ISAC_RUNS_DIR" = /* ]]
[[ "$ISAC_PYTHON" = /* && -x "$ISAC_PYTHON" ]]
cd -- "$ISAC_PROJECT_ROOT"
git diff --exit-code HEAD -- >/dev/null
export PYTHONPATH="$PWD/src"
seeds=(7 11 13)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
dir="$ISAC_RUNS_DIR/array-$SLURM_ARRAY_JOB_ID/seed-$seed"
mkdir -p -- "$(dirname "$dir")"
mkdir -- "$dir"
echo "Recovery study: seed=$seed, evidence=$dir"
"$ISAC_PYTHON" scripts/hamilton_preflight.py --out "$dir/preflight.json"
"$ISAC_PYTHON" scripts/run_recovery_seed.py --dir "$dir" --seed "$seed"
