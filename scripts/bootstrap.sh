#!/usr/bin/env bash
# Installs into project-local Python venv; Rust into the user's normal rustup home.
# Does not install OS packages, modify a repo, or silently ignore missing tools.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Use Python 3.11-3.13"'
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip>=25,<26'
if [[ "$(uname -s)" == Linux ]]; then
  .venv/bin/python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/python -m pip install 'torch==2.10.0'
fi
.venv/bin/python -m pip install -c ci/constraints.txt -e '.[dev]'
if ! command -v rustup >/dev/null; then
  if [[ "${INSTALL_RUSTUP:-0}" != 1 ]]; then
    echo 'rustup missing. Install from rustup.rs, or explicitly rerun INSTALL_RUSTUP=1 bash scripts/bootstrap.sh.' >&2
    exit 2
  fi
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --location https://sh.rustup.rs -o "$installer"
  sh "$installer" -y --profile minimal --default-toolchain 1.85.1
  source "$HOME/.cargo/env"
fi
rustup toolchain install 1.85.1 --profile minimal --component rustfmt,clippy
cargo --version
.venv/bin/python scripts/ci.py --profile all --out "validation/local-ci/$(date -u +%Y%m%dT%H%M%SZ)"
