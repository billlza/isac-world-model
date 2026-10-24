# Reproducible checks

## Setup

Use Python 3.11-3.13. `bash scripts/bootstrap.sh` creates a project-local venv,
installs pinned direct dependencies, installs Rust 1.85.1 components using an
existing rustup, then runs all gates. On Linux it deliberately installs CPU torch.
For a CUDA environment install the appropriate official torch build separately;
do not overwrite an existing research environment without checking compatibility.
If rustup is absent, the script stops. Explicit opt-in `INSTALL_RUSTUP=1` downloads
and executes the official installer. OS linker/build tools must be installed.

The dependency constraints are direct pins, **not** a transitive hash lock or
supply-chain audit. Cargo.lock has no third-party Rust crates; Cargo checks run
`--locked --offline` after toolchain provisioning.

## Local execution

```bash
source .venv/bin/activate
python scripts/ci.py --profile all --out validation/run-001
```

Output must be a new directory so prior evidence is not silently overwritten.
`python`, `native`, and `quality` profiles allow the same jobs on separate workers.
Exit 0 means all selected checks passed, 1 means a failed check, and 2 means a
missing prerequisite. Unavailable Cargo/Ruff is never silently skipped. Each step
has a deadline, log and JSON status. Pytest auto-discovery of unrelated globally
installed plugins is disabled; the coverage plugin is loaded explicitly. Native
and BLAS thread settings are bounded for repeatability.

Python gates cover source/workflow contracts, pytest + combined statement/branch coverage (floor 88%),
actual train/resume/calibrate/delayed-evaluate/stress, wheel+sdist, and importing the
installed wheel outside the source tree. Wheel validation reuses host dependencies;
it is not a clean-room reproduction of the whole dependency graph.

Native gates cover rustfmt, Clippy with warnings denied, unit tests, release build,
and 25,005 seeded differential protocol commands plus malformed stream rejection.
The same oracle is checked against Python EventQueue. Integer test timestamps are
in the exactly representable float64 domain; this does not establish equivalence
for arbitrary u64 nanosecond values in Python's float-seconds representation.

Quality requires Ruff fatal-correctness checks (E9/F63/F7/F82). Broader style rules
and a vulnerability database audit are not claimed by this gate.

## GitHub Actions

`.github/workflows/tests.yml` defines Ubuntu/Python 3.11 and 3.13, macOS/Python 3.12
(CPU, **not MPS**), native Linux, quality, and an aggregate `ci-gate` that fails on
failed/cancelled/skipped prerequisites. Actions are pinned to verified commit SHAs;
permissions are read-only, checkout does not persist credentials, and no secrets
or privileged `pull_request_target` trigger is used. Failure logs are uploaded.
Branch protection must still be configured in the repository to require `ci-gate`.

After deliberately pushing the exact commit to the private repository:

```bash
bash scripts/verify_remote_ci.sh billlza/isac-world-model main
```

It verifies owner, private visibility, and matching local/remote SHA, dispatches
an identifiable workflow run, waits and exits nonzero if the run fails. Writing
this workflow or executing local checks is **not** evidence of a GitHub Actions run.
See the release validation record for which environments were actually executed.
