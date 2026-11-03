#!/bin/bash
# From the Mac project checkout: bash /path/to/this_script EXACT_FULL_SHA DURHAM_SHORT_USERNAME
set -euo pipefail
umask 077
base=b2547848b09b8c8cb3e7a03ce0154595465006d6
ref=refs/remotes/origin/research/memory-evidence-seeds
pin=${1:-}
account=${2:-}
if [[ "$(uname -s)" != Darwin || ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Run on your Mac in the existing project checkout, with the published SHA and Durham short username.' >&2
  exit 2
fi
cd -- "$(git rev-parse --show-toplevel)"
if [[ "$(git rev-parse "$ref")" != "$pin" ]]; then
  echo 'Fetched research/memory-evidence-seeds does not match the published revision.' >&2; exit 1
fi
git merge-base --is-ancestor "$base" "$pin"
transfer_dir=$(mktemp -d)
bundle="$transfer_dir/followup.bundle"
trap 'rm -f -- "$bundle"; rmdir -- "$transfer_dir"' EXIT
git bundle create "$bundle" "$ref" "^${base}"
remote="$account@hamilton8.dur.ac.uk"
leaf="isac-memory-$pin-$$-$(date +%s).bundle"
scp "$bundle" "$remote:$leaf"
ssh "$remote" "bash -l -s -- $pin $leaf" <<'HAMILTON'
set -eo pipefail
umask 077
pin=$1
leaf=$2
base=b2547848b09b8c8cb3e7a03ce0154595465006d6
ref=refs/remotes/origin/research/memory-evidence-seeds
prior_root="$HOME/isac-recovery-b2547848b09b"
root="$HOME/isac-memory-${pin:0:12}"
claim="$root/.memory-submission"
bundle="$HOME/$leaf"
trap 'rm -f -- "$bundle"' EXIT
case "$(hostname -f)" in
  login1.ham8.dur.ac.uk|login2.ham8.dur.ac.uk) ;;
  *) echo 'Unexpected remote login host; stopping.' >&2; exit 1 ;;
esac
if [[ ! -f "$bundle" || ! -d "$prior_root" || "$(git -C "$prior_root" rev-parse HEAD)" != "$base" ]]; then
  echo 'Missing bundle, missing original Hamilton round-2 checkout, or unexpected prior revision.' >&2; exit 1
fi
if [[ ! -d "$root" ]]; then
  git -C "$prior_root" bundle verify "$bundle"
  git -C "$prior_root" fetch "$bundle" "$ref"
  [[ "$(git -C "$prior_root" rev-parse FETCH_HEAD)" == "$pin" ]]
  git -C "$prior_root" worktree add --detach "$root" "$pin"
fi
cd -- "$root"
[[ "$(git rev-parse HEAD)" == "$pin" ]]
git diff --quiet HEAD --
[[ -z "$(git ls-files --others --exclude-standard -- src scripts)" ]]
report_receipt() {
  local line job
  line=$(cat "$claim/receipt")
  job=${line%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid saved receipt. Inspect the claim and squeue --me.' >&2; return 1; }
  printf 'Slurm job: %s\nQuery on Hamilton: sacct -X -j %s --format=JobID,State,ExitCode,Elapsed\n' "$job" "$job"
  printf 'Log: %s/slurm-%s.out\nResults: %s/runs/followup-%s\n' "$root" "$job" "$root" "$job"
}
if [[ -d "$claim" ]]; then
  if [[ -s "$claim/receipt" ]]; then
    echo 'Already submitted; no duplicate job.'; report_receipt; exit 0
  fi
  echo 'Submission claim exists without a confirmed receipt. Inspect the claim and squeue --me before retrying.' >&2
  exit 1
fi
module load python/3.13.9cuda
export ISAC_PYTHON=/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python
export ISAC_PROJECT_ROOT="$root" ISAC_RUNS_DIR="$root/runs" ISAC_FOLLOWUP_PIN="$pin"
export ISAC_ROUND2_DIR="$prior_root/runs/array-19341792"
[[ -s "$prior_root/runs/comparison-19341792.json" ]]
for s in 7 11 13; do
  dir="$ISAC_ROUND2_DIR/seed-$s"
  [[ -s "$dir/study.json" ]]
  for h in 96 192; do
    [[ -s "$dir/recovery-$h.json" ]]
    for a in structured structured-attention; do
      [[ -s "$dir/$a-$h/model.pt" ]] || { echo "Missing prior checkpoint: $dir/$a-$h/model.pt" >&2; exit 1; }
    done
  done
done
"$ISAC_PYTHON" - <<'PY'
import sys
import numpy, torch
v = (sys.version_info[:3], numpy.__version__, str(torch.__version__), torch.version.cuda)
if v != ((3, 13, 9), "2.3.4", "2.9.0+cu128", "12.8"):
    raise SystemExit(f"Unexpected Hamilton school runtime: {v!r}")
print("Verified Hamilton Python:", v)
PY
# Atomic claim survives disconnect/error. Never retry sbatch automatically.
mkdir -- "$claim" || { echo 'Concurrent submission; inspect .memory-submission/receipt.' >&2; exit 1; }
if ! sbatch --parsable --export=ALL scripts/hamilton_memory_followup.sh >"$claim/receipt.pending" 2>"$claim/submission.stderr"; then
  cat "$claim/submission.stderr" >&2
  echo 'No confirmed receipt. Claim retained; inspect squeue --me and submission.stderr.' >&2
  exit 1
fi
receipt=$(cat "$claim/receipt.pending")
if [[ ! "${receipt%%;*}" =~ ^[0-9]+$ ]]; then
  echo 'Unexpected scheduler receipt. Inspect .memory-submission/receipt.pending and squeue --me.' >&2; exit 1
fi
mv -- "$claim/receipt.pending" "$claim/receipt"
echo 'Submitted memory and seed study.'
report_receipt
HAMILTON
