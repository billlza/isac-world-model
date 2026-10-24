#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Missing native tools cause BLOCKED, never a silent successful skip.
python scripts/ci.py --profile all --out "validation/local-ci/$(date -u +%Y%m%dT%H%M%SZ)"
