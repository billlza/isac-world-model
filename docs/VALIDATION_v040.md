# ISAC World Model v0.4.0 — dependency and toolchain validation

Date: 2026-10-24. Baseline commit: `fa4e5651a9c220bb3dac2635a814f8abee30878f` (v0.3.0).

## Result

**Full implemented-project CI passed: 17 required checks, none blocked or skipped.**
This closes v0.3's missing Ruff execution and validates the updated Python/Rust environment.
The remote validation environment was Linux x86_64, 2 vCPUs / 4 GB RAM, CPU-only.
It was an isolated Vercel sandbox, not GitHub Actions or the user's Mac.
The packaging container uses an older runtime and is NOT the source of these test results.

| Check | Observed result |
|---|---|
| Python tests | 154 passed; 0 failures, 0 errors, 0 skipped |
| Coverage statements | 1343/1458 = 92.11% |
| Coverage branches | 385/484 = 79.55% |
| Combined coverage | 88.98%; existing 88% gate retained |
| Ruff lint and formatting | Passed with Ruff 0.17.2 |
| Dependency policy and environment | Passed; exact direct pins match installed releases |
| pip check | No broken requirements |
| Known-advisory scan | 49 installed third-party distributions audited; no reported known advisories or skipped packages |
| Rust | 15 tests passed; rustfmt, Clippy `-D warnings`, and release build passed |
| Native protocol parity | 25,005 commands across five seeds plus two malformed-stream tests passed |
| Integration | Random-init train → resume → independent calibration → fixed-150ms evaluation → six fault scenarios passed |
| Distribution | Wheel and sdist built with the updated backend; wheel import and CLI verified outside checkout |
| Export integrity | All 50 files covered by CI source hashes match the delivered source |

No coverage gate was lowered. Coverage is of `isac_wm`, not every CI script. The installed-wheel test uses
an empty installation target outside the checkout but reuses the tested host dependencies; it is not a
second full clean-machine dependency installation. The whole source/test/dependency environment was
created afresh for this round.

## Upgraded versions

| Component | Version |
|---|---|
| Python reference | 3.14.9 |
| PyTorch | 2.14.1+cpu (upstream release 2.14.1) |
| NumPy | 2.5.4 |
| Rust / Cargo | 1.99.1 |
| rustup | 1.29.2 |
| pytest / pytest-cov | 9.1.1 / 7.1.0 |
| coverage / Ruff | 7.17.0 / 0.17.2 |
| pip / uv | 26.2.2 / 0.13.5 |
| build / setuptools / wheel | 1.6.1 / 84.2.1 / 0.48.0 |
| packaging / pyproject-hooks | 26.3 / 1.3.3 |
| PyYAML / pip-audit | 6.0.3 / 2.10.1 |
| Optional Playwright | 1.64.1, declared and locked; not installed or browser-tested in this round |

Direct versions were checked against official PyPI metadata. A second live check included all 49
installed third-party packages: **48 match current upstream stable releases, with one required compatibility exception.**

### Explicit compatibility exceptions

1. **Python 3.15.1** is the latest upstream CPython release, but PyTorch 2.14.1's published wheel set does
   not contain CPython 3.15 wheels. The tested/reference runtime is the latest 3.14 patch, **3.14.9**.
   This is a binary-distribution compatibility choice, not a proof that PyTorch can never build from source on 3.15.
2. **mpmath 1.3.0** is retained because the current **SymPy 1.14.0** metadata requires
   `mpmath<1.4,>=1.1.0`. Upstream mpmath is 1.4.1. No `--no-deps` installation overrides that requirement.
3. NumPy 2.5.4 requires Python >=3.12. Project support is now **>=3.12,<3.15**. Older Python was not
   silently claimed to remain supported.

`uv.lock` freezes 55 packages across the selected extras and conditional platforms. The tested dev
installation contains the project plus 49 third-party distributions. The lock includes hashes and a
reviewed artifact subset for CPython 3.12–3.14 on Linux x86_64 and macOS Apple Silicon; source archives
are retained. `uv lock --check` and `uv sync --locked --extra dev --python 3.14.9` were both executed.
See `ci/artifact-scope.json`. Resolving a macOS artifact is not equivalent to executing it on a Mac.

## New engineering checks

- `scripts/dependencies.py`: offline pin/policy consistency, installed environment verification,
  complete installed-package inventory, read-only official-registry latest checks. Missing/yanked/
  prerelease/unavailable metadata never silently passes.
- `scripts/audit_dependencies.py`: explicitly maps approved torch `+cpu` / `+cuNNN` build labels to
  their upstream release for advisory lookup, records that mapping, and fails on skipped entries.
  The first raw pip-audit invocation skipped torch+cpu; this defect was found and corrected before
  the final all-profile run. This checks known release advisories, not backend-binary authenticity.
- `scripts/bootstrap.sh`: installs Python/uv/Rust into project-local directories, does not use sudo or
  change global default toolchains, requires explicit authorization for rustup installation, refuses
  silent replacement of an existing .venv, and runs the complete CI profile.
- CI records dependency metadata, command exit status, logs, source hashes, and build/install results.
  Missing prerequisites return 2; failed checks return 1; only complete selected success returns 0.
- All selected GitHub Actions are upgraded to current checked releases and pinned to full commit SHAs.
  Default permissions are read-only; no privileged PR trigger or persistently stored checkout credentials.

## What this release does not claim

This is a toolchain and reproducibility release, not a model-accuracy improvement. The integration
training is a small smoke experiment, not a replacement for the previous research training or benchmarks.
Existing v0.2 weights and v0.3 result files are retained with their original provenance. Their past metrics
are not re-labelled as newly measured under PyTorch 2.14.1. Recalibrate and re-evaluate before reporting
research results under a new backend/version.

GitHub Actions has not run: the connected lookup of `billlza/isac-world-model` returned 404. The workflow
is implemented, but there is no invented run URL or green remote badge. macOS, CUDA and MPS were not
executed here. Rust is tested as an independent event component, not yet the default neural-loop backend.
The optional browser check and hardware drivers are not part of this round's validated runtime.
No dependency-advisory result, unit-test count, or checksum is a physical safety certification.

## Evidence map

- `validation/v040-release/ci-report.json`: 17 required outcomes and 50 source hashes.
- `pytest.xml`, `python-tests.log`, `coverage.xml`: test/coverage evidence.
- `rust-tests.log`, `clippy.log`, `rustfmt.log`, `native-parity.json`: native evidence.
- `audit.json`, `dependency-audit.log`: all 49 dependency advisory checks and explicit backend mapping.
- `environment.json`, `dependency-environment.log`: installed releases and dependency metadata.
- `latest-versions.json`: direct/optional-package live release check.
- `transitive-latest.json`: all installed third-party package currentness and mpmath constraint.
- `source-export-verification.json`: validation source versus exported source verification.
- `integration.log`, `distribution.log`, `wheel-install.log`: actual integration/package commands.

## Official upstream references

- PyPI project and release JSON URLs are recorded in `ci/versions.json`.
- Python releases: https://www.python.org/downloads/
- Rust release announcements: https://blog.rust-lang.org/
- torch: https://pypi.org/project/torch/2.14.1/
- NumPy: https://pypi.org/project/numpy/2.5.4/
- SymPy dependency metadata: https://pypi.org/pypi/sympy/1.14.0/json

## External execution

The existing `isac-world-model-ci` Vercel project was reused. No public service, route, or port was opened.
Only the project's own source, test logs and public package/tool downloads were used. No research papers,
private institutional datasets, account credentials, or model weights were uploaded. Both temporary sessions
have stopped. One snapshot of the tool/source environment remains with the provider's seven-day expiration;
see `docs/EXTERNAL_ACTIONS_v040.md`. No new project was created in this round.
