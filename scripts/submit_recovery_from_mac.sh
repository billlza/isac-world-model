#!/bin/bash
# Upload the exact tested change from an existing Mac Git checkout, then submit once.
# Usage (from the Mac checkout root): bash scripts/submit_recovery_from_mac.sh SHA USER
set -euo pipefail
base=46ef34373a56525f2f2b38dfd7400bc67a124491
ref=refs/remotes/origin/research/paired-recovery-h200
pin=${1:-}
account=${2:-}
if [[ "$(uname -s)" != Darwin || ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Run on your Mac, in the project checkout, with the published full SHA and Durham username.' >&2
  exit 2
fi
cd "$(git rev-parse --show-toplevel)"
if [[ "$(git rev-parse "$ref")" != "$pin" ]]; then
  echo 'The fetched study branch is not the published revision. Stop and check the SHA.' >&2
  exit 1
fi
git merge-base --is-ancestor "$base" "$pin"
transfer_dir=$(mktemp -d)
bundle="$transfer_dir/recovery.bundle"
trap 'rm -f -- "$bundle"; rmdir -- "$transfer_dir"' EXIT
git bundle create "$bundle" "$ref" "^$base"
remote="$account@hamilton8.dur.ac.uk"
leaf="isac-recovery-$pin-$$-$(date +%s).bundle"
scp "$bundle" "$remote:$leaf"
# The positional arguments contain only checked hex / alphanumeric characters and separators.
ssh "$remote" "bash -l -s -- $pin $leaf" <<'HAMILTON'
set -eo pipefail
pin=$1
leaf=$2
base=46ef34373a56525f2f2b38dfd7400bc67a124491
ref=refs/remotes/origin/research/paired-recovery-h200
case "$(hostname -f)" in
  login1.ham8.dur.ac.uk|login2.ham8.dur.ac.uk) ;;
  *) echo 'Unexpected remote login host; stopping.' >&2; exit 1 ;;
esac
base_root="$HOME/isac-wm-46ef343"
root="$HOME/isac-recovery-${pin:0:12}"
bundle="$HOME/$leaf"
if [[ ! -f "$bundle" || ! -d "$base_root" ]]; then
  echo 'The transferred bundle or the previous pilot checkout is missing.' >&2; exit 1
fi
trap 'rm -f -- "$bundle"' EXIT
git -C "$base_root" cat-file -e "$base^{commit}"
if [[ ! -d "$root" ]]; then
  git -C "$base_root" bundle verify "$bundle"
  git -C "$base_root" fetch "$bundle" "$ref"
  if [[ "$(git -C "$base_root" rev-parse FETCH_HEAD)" != "$pin" ]]; then
    echo 'Transferred revision does not match; stopping.' >&2; exit 1
  fi
  git -C "$base_root" worktree add --detach "$root" "$pin"
fi
cd "$root"
if [[ "$(git rev-parse HEAD)" != "$pin" ]]; then
  echo 'Existing study checkout has a different revision.' >&2; exit 1
fi
git diff --exit-code HEAD -- >/dev/null
module load python/3.13.9cuda
export ISAC_PYTHON=/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python
export ISAC_PROJECT_ROOT="$root"
export ISAC_RUNS_DIR="$root/runs"
# The allocation wrapper owns its hardware checks; only check the build on the login node.
"$ISAC_PYTHON" - <<'PY'
import sys
import numpy
import torch
versions = (sys.version_info[:3], numpy.__version__, str(torch.__version__), torch.version.cuda)
if versions != ((3, 13, 9), "2.3.4", "2.9.0+cu128", "12.8"):
    raise SystemExit(f"Unexpected Hamilton module versions: {versions!r}")
print("Verified Hamilton CUDA Python build:", versions)
PY
claim="$root/.recovery-submission"
if ! mkdir "$claim"; then
  if [[ -s "$claim/receipt" ]]; then
    printf 'Already submitted (Slurm receipt): '; cat "$claim/receipt"
  else
    echo 'Submission claim exists without a receipt; inspect squeue --me and the claim before retrying.' >&2
  fi
  exit 1
fi
# Keep the claim if the connection or submission fails: never charge for an automatic duplicate.
receipt=$(sbatch --parsable --export=ALL scripts/hamilton_h200_recovery.sh 2>"$claim/submission.stderr") || {
  cat "$claim/submission.stderr" >&2; exit 1
}
printf '%s\n' "$receipt" > "$claim/receipt"
job=${receipt%%;*}
[[ "$job" =~ ^[0-9]+$ ]]
printf 'Submitted recovery study: %s\n' "$job"
printf 'On Hamilton: sacct -X -j %s --format=JobID,State,ExitCode,Elapsed\n' "$job"
printf 'Checkout / logs: %s\nOutputs: %s\n' "$root" "$ISAC_RUNS_DIR"
HAMILTON
