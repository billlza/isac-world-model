#!/bin/bash
# From the existing Mac checkout: bash SCRIPT EXACT_FULL_SHA DURHAM_SHORT_USERNAME
# Submit a CPU runtime build followed by one explicit 1g.10gb MIG development pilot.
set -euo pipefail
umask 077

ref=refs/remotes/origin/research/ncc-mig-pilot
pin=${1:-}
account=${2:-}
if [[ "$(uname -s)" != Darwin || ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Run on your Mac in the existing project checkout, with the published SHA and Durham short username.' >&2
  exit 2
fi
cd -- "$(git rev-parse --show-toplevel)"
if [[ "$(git rev-parse "$ref")" != "$pin" ]]; then
  echo 'Fetched research/ncc-mig-pilot does not match the published revision.' >&2
  exit 1
fi

transfer_dir=$(mktemp -d)
bundle="$transfer_dir/ncc-mig.bundle"
trap 'rm -f -- "$bundle"; rmdir -- "$transfer_dir"' EXIT
# NCC has no earlier project checkout: this bundle must have no prerequisite commits.
git bundle create "$bundle" "$ref"
remote="$account@ncc1.clients.dur.ac.uk"
leaf="isac-ncc-mig-$pin-$$-$(date +%s).bundle"
scp "$bundle" "$remote:$leaf"
ssh "$remote" "bash -l -s -- $pin $leaf $account" <<'NCC'
set -euo pipefail
umask 077
export LC_ALL=C
pin=$1
leaf=$2
account=$3
if [[ ! "$pin" =~ ^[0-9a-f]{40}$ || ! "$leaf" =~ ^isac-ncc-mig-[0-9a-f]{40}-[0-9]+-[0-9]+\.bundle$ || ! "$account" =~ ^[a-zA-Z0-9]+$ ]]; then
  echo 'Invalid remote revision, username or bundle filename; stopping.' >&2
  exit 2
fi
ref=refs/remotes/origin/research/ncc-mig-pilot
root="$HOME/isac-ncc-mig-${pin:0:12}"
runtime_root="$HOME/isac-ncc-runtime-${pin:0:12}"
claim="$root/.ncc-mig-submission"
bundle="$HOME/$leaf"
trap 'rm -f -- "$bundle"' EXIT
case "$(hostname -f)" in
  ncc1|ncc1.clients.dur.ac.uk) ;;
  *) echo 'Unexpected remote login host; stopping.' >&2; exit 1 ;;
esac
if [[ "$(id -un)" != "$account" ]]; then
  echo 'Remote login identity does not match the requested Durham username.' >&2
  exit 1
fi
[[ -f "$bundle" ]] || { echo 'Transferred Git bundle is missing.' >&2; exit 1; }

if [[ ! -e "$root" && ! -L "$root" ]]; then
  # mkdir is the exclusive initialization claim. A partial checkout is preserved.
  mkdir -- "$root" || { echo 'Concurrent checkout initialization; inspect the remote directory.' >&2; exit 1; }
  git -C "$root" init --quiet
  git -C "$root" bundle verify "$bundle"
  git -C "$root" fetch "$bundle" "$ref"
  if [[ "$(git -C "$root" rev-parse FETCH_HEAD)" != "$pin" ]]; then
    echo 'Fetched remote revision does not match the requested revision; checkout retained.' >&2
    exit 1
  fi
  git -C "$root" checkout --detach "$pin"
fi
if [[ ! -d "$root" || -L "$root" ]]; then
  echo 'The expected remote checkout path is not an independent directory; stopping.' >&2
  exit 1
fi
cd -- "$root"
if [[ "$(git rev-parse HEAD)" != "$pin" ]]; then
  echo 'Remote checkout is incomplete or has a different revision. Existing files are preserved.' >&2
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

read_job_id() {
  local receipt
  receipt=$(cat "$1") || return 1
  if [[ ! "$receipt" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]]; then
    printf 'Invalid saved Slurm receipt: %s\n' "$1" >&2
    return 1
  fi
  printf '%s\n' "${receipt%%;*}"
}

report_receipts() {
  local setup_job pilot_job
  setup_job=$(read_job_id "$claim/setup.receipt") || return 1
  printf 'CPU runtime setup job: %s\nSetup log: %s/slurm-%s.out\n' "$setup_job" "$root" "$setup_job"
  printf 'Runtime directory: %s\nRuntime status: %s/runtime.json\n' "$runtime_root" "$runtime_root"
  if [[ ! -s "$claim/pilot.receipt" ]]; then
    printf 'MIG pilot has no confirmed receipt. Inspect %s and the NCC queue before any retry.\n' "$claim" >&2
    return 1
  fi
  pilot_job=$(read_job_id "$claim/pilot.receipt") || return 1
  printf 'MIG A512 pilot job: %s (afterok:%s)\n' "$pilot_job" "$setup_job"
  printf 'Query on NCC: sacct -X -j %s,%s --format=JobID,JobName,State,ExitCode,Elapsed\n' "$setup_job" "$pilot_job"
  printf 'Queue on NCC: squeue -j %s,%s -o "%%.18i %%.12T %%.10M %%R"\n' "$setup_job" "$pilot_job"
  printf 'Pilot log: %s/slurm-%s.out\nPilot results: %s/runs/ncc-mig-%s\n' "$root" "$pilot_job" "$root" "$pilot_job"
  printf 'Pilot status: %s/runs/ncc-mig-%s/status.json\n' "$root" "$pilot_job"
  printf 'Saved job IDs: %s/setup.job-id and %s/pilot.job-id\n' "$claim" "$claim"
}

if [[ -e "$claim" || -L "$claim" ]]; then
  if [[ ! -d "$claim" || -L "$claim" || ! -f "$claim/git-commit" || "$(cat "$claim/git-commit")" != "$pin" ]]; then
    echo 'Existing submission claim is incomplete or does not match this revision; inspect it before retrying.' >&2
    exit 1
  fi
  if [[ -s "$claim/setup.receipt" ]]; then
    echo 'Existing submission claim found; no jobs will be resubmitted.'
    report_receipts
    exit 0
  fi
  printf 'Submission claim exists without a confirmed setup receipt. Inspect %s and squeue --me before retrying.\n' "$claim" >&2
  exit 1
fi
if [[ -e "$runtime_root" || -L "$runtime_root" ]]; then
  echo 'Runtime directory already exists without this submission claim; inspect it before submitting.' >&2
  exit 1
fi

export ISAC_PROJECT_ROOT="$root" ISAC_NCC_PIN="$pin" ISAC_RUNS_DIR="$root/runs"
export ISAC_NCC_RUNTIME_ROOT="$runtime_root"
# Submission must use the scripts' declared resources, not inherited SBATCH_* overrides.
while IFS= read -r option; do
  [[ -z "$option" ]] || unset "$option"
done < <(compgen -v SBATCH_ || true)
unset ISAC_NCC_SETUP_JOB_ID

# Preserve the claim before contacting Slurm, including ambiguous/failed submissions.
mkdir -- "$claim" || { echo 'Concurrent submission; inspect .ncc-mig-submission.' >&2; exit 1; }
printf '%s\n' "$pin" > "$claim/git-commit"
printf '{"schema_version":1,"git_commit":"%s","study":"ncc-mig-a512-pilot","profile":"development","gres":"gpu:1g.10gb:1","username":"%s"}\n' "$pin" "$account" > "$claim/request.json"
printf '%s\n' 'submitting_setup' > "$claim/stage"
if ! sbatch --parsable --export=ALL scripts/ncc_runtime_setup.sh >"$claim/setup.receipt.pending" 2>"$claim/setup.submit.error"; then
  cat "$claim/setup.submit.error" >&2
  echo 'Runtime setup submission has no confirmed receipt. Claim retained; inspect the queue and setup.submit.error.' >&2
  exit 1
fi
setup_job=$(read_job_id "$claim/setup.receipt.pending") || exit 1
mv -- "$claim/setup.receipt.pending" "$claim/setup.receipt"
printf '%s\n' "$setup_job" > "$claim/setup.job-id"

export ISAC_NCC_SETUP_JOB_ID="$setup_job"
printf '%s\n' 'submitting_pilot' > "$claim/stage"
if ! sbatch --parsable --export=ALL --dependency="afterok:$setup_job" --kill-on-invalid-dep=yes scripts/ncc_mig_pilot.sh >"$claim/pilot.receipt.pending" 2>"$claim/pilot.submit.error"; then
  cat "$claim/pilot.submit.error" >&2
  printf 'Runtime setup job %s is submitted; MIG pilot submission has no confirmed receipt. Inspect pilot.submit.error and the queue.\n' "$setup_job" >&2
  exit 1
fi
pilot_job=$(read_job_id "$claim/pilot.receipt.pending") || exit 1
mv -- "$claim/pilot.receipt.pending" "$claim/pilot.receipt"
printf '%s\n' "$pilot_job" > "$claim/pilot.job-id"
printf '%s\n' 'submitted' > "$claim/stage"
echo 'Submitted CPU runtime setup and the explicit 1g.10gb MIG A512 development pilot.'
report_receipts
NCC
