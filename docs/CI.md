# CI and toolchain contract — v0.4.0

## Policy

The source of truth for direct versions is `ci/versions.json`; runtime/build/dev pins and
`ci/constraints.txt` must agree. Full selected dependency resolution and artifact hashes live in
`uv.lock`. Do not replace exact versions with moving `latest` references in required CI.
`ci/artifact-scope.json` describes the reviewed wheel targets. Linux intentionally uses the explicit
PyTorch CPU index; accelerator environments require a separate validated profile.

The default/reference interpreter is 3.14.9. Supported minors are 3.12, 3.13, 3.14, not 3.15 until the
PyTorch binary compatibility decision is reviewed. mpmath remains 1.3.0 because SymPy requires <1.4.
These are recorded compatibility exceptions, not skipped upgrades.

## Bootstrap

```bash
INSTALL_RUSTUP=1 bash scripts/bootstrap.sh
```

Use a fresh directory. Existing .venv causes an explicit stop unless RECREATE_VENV=1 is set.
The bootstrap needs a pre-existing Python >=3.10 to install uv and the chosen managed interpreter.
It creates project-local .bootstrap, .venv, CARGO_HOME and RUSTUP_HOME, and does not change global
Rust/Python defaults. It checks, but does not sudo-install, the system linker. Rust installation is
opt-in; the rustup installer is fetched over HTTPS, while the Rust toolchain itself is pinned.

The bootstrap script's policy and shell syntax were checked; dependencies were installed and all
commands executed in the isolated CI environment. The entire bootstrap has not been re-executed on
macOS or every Linux distribution. Do not equate platform configuration with platform execution.

## Profiles

```bash
python scripts/ci.py --profile all --out validation/new-run
python scripts/ci.py --profile python --out validation/new-python-run
python scripts/ci.py --profile native --out validation/new-native-run
python scripts/ci.py --profile quality --out validation/new-quality-run
python scripts/ci.py --profile security --out validation/new-security-run
```

The output directory must be empty. Commands have deadlines; child process groups are terminated
on timeout on POSIX. The selected checks are mandatory. Status codes are 0 passed, 1 failed,
2 missing prerequisite. Network advisories can fail a release gate if the service is unavailable;
there is no fallback to an invented clean audit.

The all-profile has 17 required steps: repository contracts, direct dependency policy, installed
versions, pip check, inventory, pytest with the unchanged 88% combined coverage gate, end-to-end
smoke, sdist/wheel build, wheel installation outside the source checkout, Ruff lint, Ruff format,
known-advisory audit, rustfmt, Clippy, Rust tests, release build, native protocol differential test.

The known-advisory audit explicitly maps torch backend suffixes to upstream release metadata and
records the mapping. Skipped/unrecognised packages fail the audit. This is not binary provenance
verification and does not imply absence of unknown vulnerabilities.

## Remote jobs

Linux CPU Python 3.12.15 / 3.13.17 / 3.14.9; macOS CPU Python 3.14.9; Rust on Linux and macOS;
quality/advisory audit; required ci-gate. The selected official Actions were checked and SHA-pinned:
checkout v7.0.1, setup-python v7.1.0, upload-artifact v7.0.2, setup-uv v10.3.2.

GitHub itself has not run this revision. No access token was obtained and the target repository is
not visible to the connector. Branch protection is an account/repository setting, not something
created by adding workflow YAML. Require `ci-gate` after the real repository is available.

## Evidence

`validation/v040-release/ci-report.json` contains the actually executed checks. Its 50 source hashes
match the exported implementation. User documentation and release packaging were completed after
those checks; no tested source was changed during packaging. CI report logs are retained.
