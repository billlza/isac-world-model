"""Import an existing Mooring CSV locally; requires Python's standard library only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permit direct execution from a checkout without installing Torch or the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isac_wm.m7_telemetry import read_mooring_samples_csv, write_telemetry_import  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Mooring --csv samples export")
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    parser.add_argument(
        "--data-origin",
        choices=("real_device", "synthetic", "unknown"),
        default="unknown",
        help="Caller declaration; the CSV does not authenticate origin",
    )
    parser.add_argument("--collector-commit", help="Actual collector Git commit, if recorded (40 hex)")
    parser.add_argument("--firmware", help="Actual device firmware, if recorded; otherwise unknown")
    args = parser.parse_args(argv)
    try:
        result = read_mooring_samples_csv(
            args.input,
            data_origin=args.data_origin,
            collector_commit=args.collector_commit,
            firmware=args.firmware,
        )
        output = write_telemetry_import(result, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Import failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": result["manifest"]["status"],
                "output": str(output),
                "records": result["quality"]["records"],
                "records_with_any_rf_metric": result["quality"]["records_with_any_rf_metric"],
                "source_sha256": result["manifest"]["source"]["sha256"],
                "online_causal_replay": False,
                "calibrated_3d_rf_input": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
