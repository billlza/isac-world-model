#!/usr/bin/env bash
# No push or creation. Verify the exact local commit and every workflow job.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$ROOT/scripts/publish_private.py" "$@" --execute --verify-only
