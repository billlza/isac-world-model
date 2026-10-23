#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest
isac-wm train --out runs/smoke-train --epochs 2 --episodes 16 --device cpu
isac-wm evaluate --checkpoint runs/smoke-train/model.pt --out runs/smoke-eval --episodes 2 --device cpu
if command -v cargo >/dev/null; then cargo test --workspace; fi
