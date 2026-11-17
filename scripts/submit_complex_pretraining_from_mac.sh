#!/bin/bash
# After fetching research/complex-scene-pretraining in the Mac checkout:
# bash scripts/submit_complex_pretraining_from_mac.sh EXACT_FULL_SHA DURHAM_SHORT_USERNAME
set -euo pipefail
umask 077
ref=refs/remotes/origin/research/complex-scene-pretraining
pin=${1:-}
account=${2:-}
if [[ "$(uname -s)" != Darwin || ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Run on your Mac in the project checkout, with the published SHA and Durham short username.' >&2
  exit 2
fi
cd -- "$(git rev-parse --show-toplevel)"
if [[ "$(git rev-parse "$ref")" != "$pin" ]]; then
  echo 'Fetched research/complex-scene-pretraining does not match the published revision.' >&2
  exit 1
fi
transfer_dir=$(mktemp -d)
bundle="$transfer_dir/pretrain.bundle"
trap 'rm -f -- "$bundle"; rmdir -- "$transfer_dir"' EXIT
# A full bundle does not depend on any previous experiment checkout on Hamilton.
git bundle create "$bundle" "$ref"
remote="$account@hamilton8.dur.ac.uk"
leaf="isac-pretrain-$pin-$$-$(date +%s).bundle"
scp "$bundle" "$remote:$leaf"
ssh "$remote" "bash -l -s -- $pin $leaf $account" <<'HAMILTON'
set -eo pipefail
umask 077
pin=$1
leaf=$2
account=$3
# Keep guards explicit: Bash 3.2 does not apply errexit to standalone [[ ... ]].
if [[ ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$leaf" =~ ^isac-pretrain-[0-9a-f]{40}-[0-9]+-[0-9]+\.bundle$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Invalid remote revision, bundle filename, or account; stopping.' >&2
  exit 2
fi
ref=refs/remotes/origin/research/complex-scene-pretraining
root="$HOME/isac-pretrain-${pin:0:12}"
claim="$root/.pretrain-submission"
bundle="$HOME/$leaf"
trap 'rm -f -- "$bundle"' EXIT
case "$(hostname -f)" in
  login1.ham8.dur.ac.uk|login2.ham8.dur.ac.uk) ;;
  *) echo 'Unexpected remote login host; stopping.' >&2; exit 1 ;;
esac
if [[ "$(id -un)" != "$account" ]]; then
  echo 'Remote username differs from the requested account; stopping.' >&2
  exit 1
fi
if [[ ! -f "$bundle" || -L "$bundle" ]]; then
  echo 'Transferred Git bundle is missing or is not a regular owned file.' >&2
  exit 1
fi
if [[ ! -e "$root" && ! -L "$root" ]]; then
  mkdir -- "$root" || { echo 'Could not reserve the new checkout.' >&2; exit 1; }
  git -C "$root" init --quiet
  git -C "$root" bundle verify "$bundle"
  git -C "$root" fetch "$bundle" "$ref"
  if [[ "$(git -C "$root" rev-parse FETCH_HEAD)" != "$pin" ]]; then
    echo 'Fetched remote revision does not match the requested revision.' >&2
    exit 1
  fi
  git -C "$root" checkout --detach "$pin"
fi
if [[ ! -d "$root" || -L "$root" ]]; then
  echo 'Expected a regular experiment checkout directory; existing path preserved.' >&2
  exit 1
fi
cd -- "$root"
if [[ "$(git rev-parse HEAD)" != "$pin" ]]; then
  echo 'Remote checkout does not match the requested revision.' >&2
  exit 1
fi
if ! git diff --quiet HEAD --; then
  echo 'Remote checkout has tracked changes or could not be inspected; stopping.' >&2
  exit 1
fi
if ! untracked=$(git ls-files --others --exclude-standard -- src scripts '*.py'); then
  echo 'Could not inspect untracked remote source files; stopping.' >&2
  exit 1
fi
if [[ -n "$untracked" ]]; then
  echo 'Remote checkout contains untracked source files; stopping.' >&2
  exit 1
fi

report_receipt() {
  local line job archive checksum status package
  if [[ ! -f "$claim/receipt" || -L "$claim/receipt" ]]; then
    echo 'Missing or invalid saved receipt; inspect .pretrain-submission and squeue --me.' >&2
    return 1
  fi
  line=$(cat "$claim/receipt") || return 1
  if [[ ! "$line" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]]; then
    echo 'Invalid saved receipt; inspect .pretrain-submission and squeue --me.' >&2
    return 1
  fi
  job=${line%%;*}
  archive="$account@hamilton8.dur.ac.uk:$root/runs/pretrain-$job/isac-pretrain-$job.tar.gz"
  checksum="$archive.sha256"
  package="$account@hamilton8.dur.ac.uk:$root/runs/pretrain-$job/package.json"
  status="$account@hamilton8.dur.ac.uk:$root/runs/pretrain-$job/status.json"
  printf 'Slurm job: %s\nQuery on Hamilton: sacct -X -j %s --format=JobID,State,ExitCode,Elapsed\n' "$job" "$job"
  printf 'Queue on Hamilton: squeue -j %s -o "%%.18i %%.12T %%.10M %%R"\n' "$job"
  printf 'Log: %s/slurm-%s.out\nResults: %s/runs/pretrain-%s\n' "$root" "$job" "$root" "$job"
  printf 'After COMPLETED (0:0), inspect: cat %q\n' "$root/runs/pretrain-$job/status.json"
  printf 'If status.json is completed, download the weights and evidence on the Mac:\n'
  printf 'pretrain_backup="$HOME/Downloads/isac-pretrain-%s"\n' "$job"
  printf 'mkdir -- "$pretrain_backup" &&\n'
  printf 'scp %q %q %q %q "$pretrain_backup/"\n' "$archive" "$checksum" "$status" "$package"
  printf '(cd -- "$pretrain_backup" && shasum -a 256 -c isac-pretrain-%s.tar.gz.sha256)\n' "$job"
}
if [[ -e "$claim" || -L "$claim" ]]; then
  if [[ ! -d "$claim" || -L "$claim" || ! -f "$claim/git-commit" || -L "$claim/git-commit" ]]; then
    echo 'Invalid existing submission claim; existing files preserved.' >&2
    exit 1
  fi
  if [[ "$(cat "$claim/git-commit")" != "$pin" ]]; then
    echo 'Submission claim belongs to a different revision; stopping.' >&2
    exit 1
  fi
  if [[ -s "$claim/receipt" ]]; then
    echo 'Already submitted; no duplicate job.'
    report_receipt
    exit $?
  fi
  echo 'Submission claim exists without a confirmed receipt. Inspect .pretrain-submission and squeue --me; do not blindly retry.' >&2
  exit 1
fi

export LC_ALL=C
module load python/3.13.9cuda
export ISAC_PYTHON=/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python
export ISAC_PROJECT_ROOT="$root" ISAC_RUNS_DIR="$root/runs" ISAC_PRETRAIN_PIN="$pin"
export PYTHONNOUSERSITE=1
"$ISAC_PYTHON" - <<'PY'
import sys
import numpy, torch
observed = (sys.version_info[:3], numpy.__version__, str(torch.__version__), torch.version.cuda)
expected = ((3, 13, 9), "2.3.4", "2.9.0+cu128", "12.8")
if observed != expected:
    raise SystemExit(f"Unexpected Hamilton school runtime: {observed!r}; expected {expected!r}")
print("Verified Hamilton Python:", observed)
PY

# Inherited SBATCH_* variables must not override the registered GPU or job budget.
while IFS= read -r option; do
  [[ -z "$option" ]] || unset "$option"
done < <(compgen -v SBATCH_ || true)
# Preserve the claim even after a disconnect, failed sbatch, or malformed receipt.
mkdir -- "$claim" || { echo 'Concurrent submission; inspect .pretrain-submission/receipt.' >&2; exit 1; }
printf '%s\n' "$pin" > "$claim/git-commit"
printf '{"schema_version":1,"git_commit":"%s","study":"complex-scene-pretraining","profile":"formal"}\n' "$pin" > "$claim/request.json"
printf '%s\n' submitting > "$claim/stage"
if ! sbatch --parsable --export=ALL scripts/hamilton_complex_pretraining.sh >"$claim/receipt.pending" 2>"$claim/submission.stderr"; then
  cat "$claim/submission.stderr" >&2
  echo 'No confirmed receipt. Claim retained; inspect squeue --me and submission.stderr.' >&2
  exit 1
fi
receipt=$(cat "$claim/receipt.pending")
if [[ ! "$receipt" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]]; then
  echo 'Unexpected scheduler receipt. Inspect .pretrain-submission/receipt.pending and squeue --me.' >&2
  exit 1
fi
mv -- "$claim/receipt.pending" "$claim/receipt"
printf '%s\n' "${receipt%%;*}" > "$claim/job-id"
printf '%s\n' submitted > "$claim/stage"
echo 'Submitted random-initialized complex-scene pretraining. Training is not complete; inspect the recorded job status.'
report_receipt
HAMILTON
