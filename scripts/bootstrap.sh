#!/usr/bin/env bash
# Project-local toolchains. No sudo, no global default Rust or Python changes.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Bootstrap needs Python >=3.10"'
UV_VERSION="$("$PYTHON" -c 'import json; print(json.load(open("ci/versions.json"))["packages"]["uv"])')"
PIP_VERSION="$("$PYTHON" -c 'import json; print(json.load(open("ci/versions.json"))["packages"]["pip"])')"
RUST_VERSION="$("$PYTHON" -c 'import json; print(json.load(open("ci/versions.json"))["rust"]["version"])')"
TARGET_PYTHON="${TARGET_PYTHON:-$(cat .python-version)}"
"$PYTHON" -m venv .bootstrap
.bootstrap/bin/python -m pip install --upgrade "pip==$PIP_VERSION" "uv==$UV_VERSION"
export PATH="$ROOT/.bootstrap/bin:$PATH"
uv python install "$TARGET_PYTHON"
if [[ -e .venv && "${RECREATE_VENV:-0}" != 1 ]]; then
  echo '.venv exists; refusing a silent cross-version replacement. Use RECREATE_VENV=1 after saving it.' >&2
  exit 2
fi
uv venv --python "$TARGET_PYTHON" --clear .venv
uv sync --locked --extra dev --python "$TARGET_PYTHON"
export CARGO_HOME="$ROOT/.toolchains/cargo"
export RUSTUP_HOME="$ROOT/.toolchains/rustup"
export PATH="$ROOT/.venv/bin:$CARGO_HOME/bin:$PATH"
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  if [[ "${INSTALL_RUSTUP:-0}" != 1 ]]; then
    echo 'Project-local rustup is absent. Re-run with INSTALL_RUSTUP=1 to authorize installation.' >&2
    exit 2
  fi
  command -v curl >/dev/null || { echo 'curl is required' >&2; exit 2; }
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --location --max-time 90 https://sh.rustup.rs -o "$installer"
  sh "$installer" -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION"
fi
rustup toolchain install "$RUST_VERSION" --profile minimal --component rustfmt,clippy
command -v cc >/dev/null || { echo 'A system linker is required: Xcode CLT on macOS; build-essential on Linux.' >&2; exit 2; }
python scripts/dependencies.py check
python scripts/dependencies.py environment
python -m pip check
python scripts/ci.py --profile all --out "validation/local-ci/$(date -u +%Y%m%dT%H%M%SZ)"
