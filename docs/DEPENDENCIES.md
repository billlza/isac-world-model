# Dependency policy and compatibility exceptions

## Toolchain audit, 2026-10-25

Official release metadata confirms all 16 declared Python packages (including optional
Playwright), Rust 1.99.1, rustup 1.29.2 and all four SHA-pinned GitHub Actions are current
stable releases. Python 3.14.9 remains the latest compatible reference interpreter:
PyTorch 2.14.1 publishes no CPython 3.15 wheels, while Python's latest stable release is 3.15.1.

Across all optional and platform dependencies, 51 of 53 unique locked third-party packages
match current stable releases. The two compatibility exceptions are mpmath 1.3.0
(SymPy 1.14.0 requires `mpmath<1.4,>=1.1.0`) and pyee 14.0.0
(Playwright 1.64.1 requires `pyee<15,>=13`; upstream pyee is 16.0.0).
Do not override these bounds merely to select the newest version.

The local bootstrap installs Python without user-level executable links. Use
`bash scripts/env.sh <command>` to select the project's interpreter and native tools.
Optional browser tooling can be installed with
`bash scripts/env.sh uv sync --locked --extra dev --extra browser`;
install Chromium with `bash scripts/env.sh python -m playwright install chromium`.
The environment wrapper keeps browser binaries in `.toolchains/playwright`; browser testing
remains separate from the required CPU CI profile.

## Original v0.4 release validation

Checked on 2026-10-24 against official registries. All direct packages use exact stable pins.
The complete installed transitive-package check is `validation/v040-release/transitive-latest.json`.
48 of 49 installed third-party distributions are current upstream stable releases. mpmath1.3.0
is the sole package exception: installed/latest SymPy1.14.0 explicitly requires mpmath<1.4,>=1.1.0;
mpmath1.4.1 is therefore not force-installed. Python3.15.1 is the latest language release, but
stable PyTorch2.14.1's published wheels stop at CPython3.14, so the reference uses3.14.9.

The full graph contains55 packages across extras/platform conditions. The tested developer
installation contains50 distributions including this project. Optional Playwright1.64.1 and its
browser binaries were not installed/tested in this release. No old browser result is relabelled.

The supplied `uv.lock` keeps the resolver's graph, sources and package versions, but its wheel list
is narrowed to the reviewed CPython3.12/3.13/3.14 Linuxx86_64 and macOSarm64 targets. Selected artifacts
and their hashes are recorded; source archives remain. Both uv lock --check and uv sync --locked were
executed successfully with the final lock. Other platforms/free-threaded interpreters are outside
this tested scope and may need a separately reviewed lock.

Safe maintenance sequence: read upstream release metadata; review required-Python and transitive
constraints; update ci/versions.json plus exact declarations; resolve in a fresh environment;
run all CI profiles; inspect skipped advisory entries; preserve the old environment/results;
only then commit the new lock. Do not run pip --upgrade unbounded inside a published experiment.
The `latest` script is read-only; it does not mutate the environment or lock file.

This release keeps existing model definitions; most edits to model code are formatter/import
changes. Latest dependencies do not establish a new accuracy or speed result. Existing checkpoints
are versioned artifacts; report fresh evaluation/calibration under each research environment.
