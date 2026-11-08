#!/usr/bin/env python3
"""Replay an existing checkpoint without training or consuming fresh-test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isac_wm.checkpoint_replay import evaluate_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=("best", "final"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--profile", choices=("formal", "development"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        kind=args.kind,
        device_name=args.device,
        profile=args.profile,
        source_commit=args.source_commit,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        output=args.out,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint_kind": report["checkpoint_kind"],
                "weights_update": report["weights_update"],
                "evidence": str(args.out),
                "reference_position_rmse_m": report["paths"]["batched"]["reference_position_rmse_m"],
                "streaming_position_rmse_m": report["paths"]["streaming"]["reference_position_rmse_m"],
                "fresh_test_consumed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
