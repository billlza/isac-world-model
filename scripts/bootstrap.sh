#!/usr/bin/env bash
# Project-local toolchains. No sudo, no global default Rust or Python changes.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
# Stop before any installation if an existing environment was not explicitly approved.
if [[ -e .venv && "${RECREATE_VENV:-0}" != 1 ]]; then
  echo '.venv exists; use scripts/env.sh, or explicitly set RECREATE_VENV=1.' >&2
  exit 2
fi
command -v git >/dev/null || { echo 'git is required for archive/history verification.' >&2; exit 2; }
command -v cc >/dev/null || { echo 'A system linker is required (Xcode CLT or build-essential).' >&2; exit 2; }
export UV_PYTHON_INSTALL_DIR="$ROOT/.toolchains/python"
export UV_CACHE_DIR="$ROOT/.toolchains/uv-cache"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Bootstrap needs Python >=3.10"'
UV_VERSION="$("$PYTHON" -c 'import json; print(json.load(open("ci/versions.json"))["packages"]["uv"])')"
PIP_VERSION="$("$PYTHON" -c 'import json; print(json.load(open("ci/versions.json"))["packages"]["pip"])')"
RUST_VERSION="$("$PYTHON" -c 'import json; print(json.load(open("ci/versions.json"))["rust"]["version"])')"
TARGET_PYTHON="${TARGET_PYTHON:-$(cat .python-version)}"
"$PYTHON" -m venv .bootstrap
.bootstrap/bin/python -m pip install --upgrade "pip==$PIP_VERSION" "uv==$UV_VERSION"
export PATH="$ROOT/.bootstrap/bin:$PATH"
# The venv resolves this managed interpreter directly; do not create user-level launchers.
uv python install --no-bin "$TARGET_PYTHON"
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
OUT="validation/local-ci/$(date -u +%Y%m%dT%H%M%SZ)-$$"
python scripts/ci.py --profile all --out "$OUT"
python scripts/evidence.py --report "$OUT/ci-report.json"
printf '\nReusable environment: bash scripts/env.sh <command>\nFull evidence: %s/ci-report.json\n' "$OUT"
