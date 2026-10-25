#!/usr/bin/env bash
# Defaults to offline preflight. --execute is required for remote writes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" "$ROOT/scripts/publish_private.py" "$@"
