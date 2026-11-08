#!/bin/bash
# From the Mac project checkout: bash SCRIPT EXACT_FULL_SHA DURHAM_SHORT_USERNAME
set -euo pipefail
umask 077
ref=refs/remotes/origin/research/width-optimization-h200
pin=${1:-}
account=${2:-}
if [[ "$(uname -s)" != Darwin || ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Run on your Mac in the project checkout, with the published SHA and Durham short username.' >&2
  exit 2
fi
cd -- "$(git rev-parse --show-toplevel)"
if [[ "$(git rev-parse "$ref")" != "$pin" ]]; then
  echo 'Fetched research/width-optimization-h200 does not match the published revision.' >&2
  exit 1
fi
transfer_dir=$(mktemp -d)
bundle="$transfer_dir/optim.bundle"
trap 'rm -f -- "$bundle"; rmdir -- "$transfer_dir"' EXIT
# A full bundle does not depend on any previous experiment checkout on Hamilton.
git bundle create "$bundle" "$ref"
bundle_digest=$(shasum -a 256 "$bundle")
bundle_digest=${bundle_digest%% *}
if [[ ! "$bundle_digest" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'Could not obtain a valid Git bundle checksum; stopping.' >&2
  exit 1
fi
remote="$account@hamilton8.dur.ac.uk"
leaf="isac-optim-$pin-$$-$(date +%s).bundle"
scp "$bundle" "$remote:$leaf"
ssh "$remote" "bash -l -s -- $pin $leaf $account $bundle_digest" <<'HAMILTON'
set -eo pipefail
umask 077
pin=$1
leaf=$2
account=$3
bundle_digest=$4
# Keep guards explicit: Bash 3.2 does not apply errexit to standalone [[ ... ]].
if [[ ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$leaf" =~ ^isac-optim-[0-9a-f]{40}-[0-9]+-[0-9]+\.bundle$ || ! "$account" =~ ^[a-zA-Z0-9]+$ || ! "$bundle_digest" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'Invalid remote revision, bundle filename, or account; stopping.' >&2
  exit 2
fi
ref=refs/remotes/origin/research/width-optimization-h200
root="$HOME/isac-optim-${pin:0:12}"
claim="$root/.optim-submission"
bundle="$HOME/$leaf"
historical_run="$HOME/isac-scale-249407dbbe40/runs/scale-19404810"
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
actual_digest=$(sha256sum "$bundle")
actual_digest=${actual_digest%% *}
if [[ "$actual_digest" != "$bundle_digest" ]]; then
  echo 'Transferred Git bundle checksum differs; stopping.' >&2
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
  local line job archive status expected_digest receipt_digest
  if [[ ! -f "$claim/receipt" || -L "$claim/receipt" ]]; then
    echo 'Missing or invalid saved receipt; inspect .optim-submission and squeue --me.' >&2
    return 1
  fi
  if [[ ! -f "$claim/receipt.sha256" || -L "$claim/receipt.sha256" ]]; then
    echo 'Receipt checksum is missing or invalid; inspect .optim-submission and squeue --me.' >&2
    return 1
  fi
  expected_digest=$(cat "$claim/receipt.sha256") || return 1
  receipt_digest=$(sha256sum "$claim/receipt") || return 1
  receipt_digest=${receipt_digest%% *}
  if [[ ! "$expected_digest" =~ ^[0-9a-f]{64}$ || "$receipt_digest" != "$expected_digest" ]]; then
    echo 'Saved receipt checksum differs; inspect .optim-submission and squeue --me.' >&2
    return 1
  fi
  line=$(cat "$claim/receipt") || return 1
  if [[ ! "$line" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]]; then
    echo 'Invalid saved receipt; inspect .optim-submission and squeue --me.' >&2
    return 1
  fi
  job=${line%%;*}
  archive="$account@hamilton8.dur.ac.uk:$root/runs/optim-$job/isac-optim-$job.tar.gz"
  status="$account@hamilton8.dur.ac.uk:$root/runs/optim-$job/status.json"
  printf 'Slurm job: %s\nQuery on Hamilton: sacct -X -j %s --format=JobID,State,ExitCode,Elapsed\n' "$job" "$job"
  printf 'Queue on Hamilton: squeue -j %s -o "%%.18i %%.12T %%.10M %%R"\n' "$job"
  printf 'Log: %s/slurm-%s.out\nResults: %s/runs/optim-%s\n' "$root" "$job" "$root" "$job"
  printf 'After COMPLETED (0:0), inspect: cat %q\n' "$root/runs/optim-$job/status.json"
  printf 'If status.json is completed, run these three lines on the Mac:\n'
  printf 'optim_backup="$HOME/Downloads/isac-optim-%s"\n' "$job"
  printf 'mkdir -- "$optim_backup" &&\n'
  printf 'scp %q %q "$optim_backup/"\n' "$archive" "$status"
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
  echo 'Submission claim exists without a confirmed receipt. Inspect .optim-submission and squeue --me; do not blindly retry.' >&2
  exit 1
fi

export LC_ALL=C
module load python/3.13.9cuda
export ISAC_PYTHON=/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python
export ISAC_PROJECT_ROOT="$root" ISAC_RUNS_DIR="$root/runs" ISAC_OPTIM_PIN="$pin"
export ISAC_OPTIM_HISTORICAL_RUN="$historical_run"
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

# Check the existing completed source before reserving a new GPU job. This only
# reads bounded evidence headers and checkpoint metadata; replay runs on CUDA.
"$ISAC_PYTHON" - "$historical_run" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.is_absolute() or not root.is_dir() or root.is_symlink():
    raise SystemExit("Registered Phase C source run is missing or is a symlink")
root = root.resolve(strict=True)

def read(relative, budget):
    path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SystemExit(f"Missing or non-regular Phase C source evidence: {relative}")
    if not 0 < path.stat().st_size <= budget:
        raise SystemExit(f"Phase C evidence size differs from its budget: {relative}")
    with path.open("rb") as stream:
        raw = stream.read(budget + 1)
    if not 0 < len(raw) <= budget:
        raise SystemExit(f"Phase C evidence grew beyond its budget: {relative}")
    return raw

revision = "249407dbbe4039f68d7cd54c6cc336a3f9c96f06"
expected = "b9ecba054a7e2ddb1f26bca8312f751160c38236e7988d871ac305c7679ba2ee"
status = json.loads(read("status.json", 65536))
if any(status.get(k) != v for k, v in {
    "git_commit": revision, "slurm_job_id": "19404810",
    "status": "completed", "stage": "done", "exit_code": 0,
}.items()):
    raise SystemExit("Registered Phase C source has not completed with the expected identity")
study_bytes = read("study/study.json", 64 * 1024**2)
if hashlib.sha256(study_bytes).hexdigest() != expected:
    raise SystemExit("Registered Phase C study differs from the reviewed evidence")
prefix = "study/cells/a512-layernorm-init-11"
for name, budget in (("model.pt", 256 * 1024**2), ("resume.pt", 2 * 1024**3)):
    path = root / prefix / name
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SystemExit(f"Required Phase C checkpoint is missing or non-regular: {name}")
    if not 0 < path.stat().st_size <= budget:
        raise SystemExit(f"Phase C checkpoint exceeds its registered byte budget: {name}")
    checksum = read(f"{prefix}/{name}.sha256", 128).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise SystemExit(f"Invalid Phase C checkpoint checksum: {name}")
    if name == "model.pt" and checksum != "35a24d691085748867be24763ad3e95f62696e97714bf3cd27dfea0ca9fa766f":
        raise SystemExit("Phase C selected-model identity differs from the reviewed evidence")
print("Verified existing Phase C source for read-only best/final checkpoint replay.")
PY

# Inherited SBATCH_* variables must not override the registered GPU or job budget.
while IFS= read -r option; do
  [[ -z "$option" ]] || unset "$option"
done < <(compgen -v SBATCH_ || true)
# Preserve the claim even after a disconnect, failed sbatch, or malformed receipt.
mkdir -- "$claim" || { echo 'Concurrent submission; inspect .optim-submission/receipt.' >&2; exit 1; }
printf '%s\n' "$pin" > "$claim/git-commit"
printf '{"schema_version":1,"git_commit":"%s","study":"width-optimization-phase-d","profile":"formal","historical_job_id":"19404810","historical_commit":"249407dbbe4039f68d7cd54c6cc336a3f9c96f06"}\n' "$pin" > "$claim/request.json"
printf '%s\n' "$bundle_digest" > "$claim/bundle.sha256"
printf '%s\n' submitting > "$claim/stage"
if ! sbatch --parsable --export=ALL scripts/hamilton_width_optimization.sh >"$claim/receipt.pending" 2>"$claim/submission.stderr"; then
  cat "$claim/submission.stderr" >&2
  echo 'No confirmed receipt. Claim retained; inspect squeue --me and submission.stderr.' >&2
  exit 1
fi
receipt=$(cat "$claim/receipt.pending")
if [[ ! "$receipt" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]]; then
  echo 'Unexpected scheduler receipt. Inspect .optim-submission/receipt.pending and squeue --me.' >&2
  exit 1
fi
receipt_digest=$(sha256sum "$claim/receipt.pending")
receipt_digest=${receipt_digest%% *}
if [[ ! "$receipt_digest" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'Could not checksum scheduler receipt; claim retained for inspection.' >&2
  exit 1
fi
printf '%s\n' "$receipt_digest" > "$claim/receipt.sha256.pending"
mv -- "$claim/receipt.pending" "$claim/receipt"
mv -- "$claim/receipt.sha256.pending" "$claim/receipt.sha256"
printf '%s\n' "${receipt%%;*}" > "$claim/job-id"
printf '%s\n' submitted > "$claim/stage"
echo 'Submitted Phase D width-optimization study (historical checkpoint replay, six fresh controlled cells; no fresh test).'
report_receipt
HAMILTON
