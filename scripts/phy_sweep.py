"""Reproducible PHY-only MIMO-OFDM communication/sensing budget experiment.

No pretrained weights, MATLAB installation, or proprietary toolbox required.
It does not replace the toy event-driven world-model benchmark: this is the
separate physical measurement layer to be validated before integration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_wm.phy_ofdm import OFDMConfig, evaluate_frame

ROOT = Path(__file__).resolve().parents[1]


def run_sweep(
    out: Path,
    *,
    seed: int = 27_000,
    trials: int = 12,
    fractions: tuple[float, ...] = (0.0, 0.2, 0.4),
    radar_snr_db: float = -35.0,
    tx_antennas: int = 2,
    user_rx_antennas: int = 4,
    radar_rx_antennas: int = 8,
) -> dict:
    if not 1 <= trials <= 128 or not 0 <= seed < 2**32 - trials:
        raise ValueError("trials and seed must fit bounded independent random trials")
    if not fractions or len(fractions) > 16 or len(set(fractions)) != len(fractions):
        raise ValueError("fractions must contain 1-16 distinct settings")
    if any(not np.isfinite(f) or not 0 <= f <= 0.8 for f in fractions):
        raise ValueError("probe fractions must be finite and in [0,0.8]")
    cfg = replace(
        OFDMConfig(),
        radar_snr_db=radar_snr_db,
        tx_antennas=tx_antennas,
        user_rx_antennas=user_rx_antennas,
        radar_rx_antennas=radar_rx_antennas,
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {out}")
    # Configuration validation and space-safe bounded arguments precede all writes.
    out.mkdir(parents=True)
    details, summary = [], []
    for fraction in fractions:
        episodes = []
        for i in range(trials):
            result = evaluate_frame(cfg, probe_fraction=fraction, seed=seed + i)
            measured = result["sensing"]
            comm = result["communication"]
            row = {
                "seed": seed + i,
                "requested_probe_fraction": fraction,
                "realized_probe_fraction": result["probe_fraction_realized"],
                "uncoded_bit_errors": comm["uncoded_bit_errors"],
                "payload_bits": comm["payload_bits"],
                "ber": comm["ber"],
                "raw_payload_rate_bps": comm["nominal_raw_payload_rate_bps"],
                "correct_bit_rate_proxy_bps": comm["correct_bit_rate_proxy_bps"],
                "range_absolute_error_m": abs(measured["range_m"] - cfg.target_range_m),
                "bearing_absolute_error_deg": abs(measured["bearing_deg"] - cfg.target_bearing_deg),
                "speed_absolute_error_mps": abs(measured["radial_speed_mps"] - cfg.target_radial_speed_mps),
                "matched_peak_energy": measured["matched_peak_energy"],
            }
            # Scored on physical predeclared quantization tolerances, not tuned on tests.
            row["correct_target_cell"] = bool(
                row["range_absolute_error_m"] < cfg.range_bin_m
                and row["bearing_absolute_error_deg"] <= 4.0
                and row["speed_absolute_error_mps"] <= 2 * cfg.doppler_bin_mps
            )
            details.append(row)
            episodes.append(row)
        summary.append(
            {
                "requested_probe_fraction": fraction,
                "realized_probe_fraction": episodes[0]["realized_probe_fraction"],
                "trials": trials,
                "ber_pooled": sum(r["uncoded_bit_errors"] for r in episodes)
                / sum(r["payload_bits"] for r in episodes),
                "raw_payload_rate_bps": statistics.mean(r["raw_payload_rate_bps"] for r in episodes),
                "correct_bit_rate_proxy_bps": statistics.mean(
                    r["correct_bit_rate_proxy_bps"] for r in episodes
                ),
                "range_mae_m": statistics.mean(r["range_absolute_error_m"] for r in episodes),
                "bearing_mae_deg": statistics.mean(r["bearing_absolute_error_deg"] for r in episodes),
                "speed_mae_mps": statistics.mean(r["speed_absolute_error_mps"] for r in episodes),
                "target_cell_detection_rate": statistics.mean(r["correct_target_cell"] for r in episodes),
            }
        )
    manifest = {
        "schema_version": 1,
        "config": cfg.to_dict(),
        "probe_fractions": list(fractions),
        "trial_count_per_setting": trials,
        "trial_seed_start": seed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": hashlib.sha256((ROOT / "src/isac_wm/phy_ofdm.py").read_bytes()).hexdigest(),
        "summary": summary,
        "limitations": [
            "One simple point target and simplified deterministic multipath, not physical hardware evidence",
            "Target-cell rate is evaluated with truth; the online estimator never receives true target state",
            "Paired seeds and normalized TX energy; no channel coding or reliable-transport goodput claim",
            "The fixed sensing-beam angle is a configured sector, not a learned or oracle action",
            "This baseline does not yet train the visual/RF world model or infer materials/shapes",
            "No MATLAB parity run claimed: reference MATLAB implementation requires independent execution",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-contained joint MIMO-OFDM ISAC physical-layer sweep")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=27_000)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--fractions", type=str, default="0,0.2,0.4")
    parser.add_argument("--radar-snr-db", type=float, default=-35.0)
    parser.add_argument("--tx", type=int, default=2)
    parser.add_argument("--user-rx", type=int, default=4)
    parser.add_argument("--radar-rx", type=int, default=8)
    args = parser.parse_args()
    try:
        fractions = tuple(float(x) for x in args.fractions.split(","))
        report = run_sweep(
            args.out,
            seed=args.seed,
            trials=args.trials,
            fractions=fractions,
            radar_snr_db=args.radar_snr_db,
            tx_antennas=args.tx,
            user_rx_antennas=args.user_rx,
            radar_rx_antennas=args.radar_rx,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(f"Saved independent traces and parameter provenance to {args.out}")


if __name__ == "__main__":
    main()
