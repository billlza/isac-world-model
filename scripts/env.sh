#!/usr/bin/env bash
# Run commands with this checkout's Python and native toolchain; no global edits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run scripts/bootstrap.sh first.' >&2; exit 2; }
export CARGO_HOME="$ROOT/.toolchains/cargo"
export RUSTUP_HOME="$ROOT/.toolchains/rustup"
export UV_PYTHON_INSTALL_DIR="$ROOT/.toolchains/python"
export UV_CACHE_DIR="$ROOT/.toolchains/uv-cache"
export PATH="$ROOT/.venv/bin:$ROOT/.bootstrap/bin:$CARGO_HOME/bin:$PATH"
[[ $# -gt 0 ]] || { echo 'Usage: bash scripts/env.sh <command> [args...]' >&2; exit 2; }
exec "$@"
