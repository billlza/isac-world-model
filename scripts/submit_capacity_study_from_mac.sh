#!/bin/bash
# From the existing Mac project checkout: bash SCRIPT EXACT_FULL_SHA DURHAM_SHORT_USERNAME
set -euo pipefail
umask 077
base=245a9dd56a49a72f942328d9c9cf9f27490b4d06
ref=refs/remotes/origin/research/capacity-data-h200
pin=${1:-}
account=${2:-}
if [[ "$(uname -s)" != Darwin || ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Run on your Mac in the existing project checkout, with the published SHA and Durham short username.' >&2
  exit 2
fi
cd -- "$(git rev-parse --show-toplevel)"
if [[ "$(git rev-parse "$ref")" != "$pin" ]]; then
  echo 'Fetched research/capacity-data-h200 does not match the published revision.' >&2
  exit 1
fi
git merge-base --is-ancestor "$base" "$pin"
transfer_dir=$(mktemp -d)
bundle="$transfer_dir/capacity.bundle"
trap 'rm -f -- "$bundle"; rmdir -- "$transfer_dir"' EXIT
git bundle create "$bundle" "$ref" "^${base}"
remote="$account@hamilton8.dur.ac.uk"
leaf="isac-capacity-$pin-$$-$(date +%s).bundle"
scp "$bundle" "$remote:$leaf"
ssh "$remote" "bash -l -s -- $pin $leaf" <<'HAMILTON'
set -eo pipefail
umask 077
pin=$1
leaf=$2
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$leaf" =~ ^isac-capacity-[0-9a-f]{40}-[0-9]+-[0-9]+\.bundle$ ]]
base=245a9dd56a49a72f942328d9c9cf9f27490b4d06
ref=refs/remotes/origin/research/capacity-data-h200
prior_root="$HOME/isac-memory-245a9dd56a49"
root="$HOME/isac-capacity-${pin:0:12}"
claim="$root/.capacity-submission"
bundle="$HOME/$leaf"
trap 'rm -f -- "$bundle"' EXIT
case "$(hostname -f)" in
  login1.ham8.dur.ac.uk|login2.ham8.dur.ac.uk) ;;
  *) echo 'Unexpected remote login host; stopping.' >&2; exit 1 ;;
esac
[[ -f "$bundle" ]] || { echo 'Transferred Git bundle is missing.' >&2; exit 1; }
if [[ ! -d "$root" ]]; then
  if [[ ! -d "$prior_root" || "$(git -C "$prior_root" rev-parse HEAD)" != "$base" ]]; then
    echo 'Missing memory-study checkout or unexpected base revision. Prior result files are not required.' >&2
    exit 1
  fi
  git -C "$prior_root" bundle verify "$bundle"
  git -C "$prior_root" fetch "$bundle" "$ref"
  [[ "$(git -C "$prior_root" rev-parse FETCH_HEAD)" == "$pin" ]]
  git -C "$prior_root" worktree add --detach "$root" "$pin"
fi
cd -- "$root"
[[ "$(git rev-parse HEAD)" == "$pin" ]]
git diff --quiet HEAD --
[[ -z "$(git ls-files --others --exclude-standard -- src scripts '*.py')" ]]

report_receipt() {
  local line job
  line=$(cat "$claim/receipt")
  [[ "$line" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]] || {
    echo 'Invalid saved receipt. Inspect the claim and squeue --me.' >&2; return 1;
  }
  job=${line%%;*}
  printf 'Slurm job: %s\nQuery on Hamilton: sacct -X -j %s --format=JobID,State,ExitCode,Elapsed\n' "$job" "$job"
  printf 'Queue on Hamilton: squeue -j %s -o "%%.18i %%.12T %%.10M %%R"\n' "$job"
  printf 'Log: %s/slurm-%s.out\nResults: %s/runs/capacity-%s\n' "$root" "$job" "$root" "$job"
  printf 'After COMPLETED (0:0), inspect: cat "%s/runs/capacity-%s/status.json"\n' "$root" "$job"
  printf 'If status.json is completed, run these three lines on the Mac:\n'
  printf 'capacity_backup="$HOME/Downloads/isac-capacity-%s"\n' "$job"
  printf 'mkdir -- "$capacity_backup" &&\n'
  printf 'scp %s@hamilton8.dur.ac.uk:%s/runs/capacity-%s/isac-capacity-%s.tar.gz %s@hamilton8.dur.ac.uk:%s/runs/capacity-%s/status.json "$capacity_backup/"\n' \
    "$(id -un)" "$root" "$job" "$job" "$(id -un)" "$root" "$job"
}
if [[ -d "$claim" ]]; then
  if [[ -s "$claim/receipt" ]]; then
    echo 'Already submitted; no duplicate job.'
    report_receipt
    exit 0
  fi
  echo 'Submission claim exists without a confirmed receipt. Inspect the claim and squeue --me before retrying.' >&2
  exit 1
fi

# C is available on every login node and avoids the Mac's unsupported LC_CTYPE=UTF-8 setting.
export LC_ALL=C
module load python/3.13.9cuda
export ISAC_PYTHON=/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python
export ISAC_PROJECT_ROOT="$root" ISAC_RUNS_DIR="$root/runs" ISAC_CAPACITY_PIN="$pin"
"$ISAC_PYTHON" - <<'PY'
import sys
import numpy, torch
v = (sys.version_info[:3], numpy.__version__, str(torch.__version__), torch.version.cuda)
if v != ((3, 13, 9), "2.3.4", "2.9.0+cu128", "12.8"):
    raise SystemExit(f"Unexpected Hamilton school runtime: {v!r}")
print("Verified Hamilton Python:", v)
PY

# A surviving claim prevents duplicate work after a disconnect or ambiguous scheduler response.
mkdir -- "$claim" || { echo 'Concurrent submission; inspect .capacity-submission/receipt.' >&2; exit 1; }
printf '{"schema_version":1,"git_commit":"%s","study":"capacity-data","profile":"formal"}\n' "$pin" > "$claim/request.json"
if ! sbatch --parsable --export=ALL scripts/hamilton_capacity_study.sh >"$claim/receipt.pending" 2>"$claim/submission.stderr"; then
  cat "$claim/submission.stderr" >&2
  echo 'No confirmed receipt. Claim retained; inspect squeue --me and submission.stderr.' >&2
  exit 1
fi
receipt=$(cat "$claim/receipt.pending")
if [[ ! "$receipt" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]]; then
  echo 'Unexpected scheduler receipt. Inspect .capacity-submission/receipt.pending and squeue --me.' >&2
  exit 1
fi
mv -- "$claim/receipt.pending" "$claim/receipt"
echo 'Submitted capacity and data study.'
report_receipt
HAMILTON
