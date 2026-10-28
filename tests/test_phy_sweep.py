"""Provenance, paired-scene accounting and fail-closed CLI output behavior."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phy_sweep", ROOT / "scripts/phy_sweep.py")
assert spec and spec.loader
phy_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phy_sweep)


def test_small_paired_phy_sweep_retains_source_and_rows(tmp_path):
    target = tmp_path / "phy"
    report = phy_sweep.run_sweep(target, trials=2, fractions=(0.0, 0.25))
    assert report["trial_count_per_setting"] == 2
    assert len(report["summary"]) == 2
    assert report["summary"][0]["raw_payload_rate_bps"] > report["summary"][1]["raw_payload_rate_bps"]
    with (target / "episodes.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert [r["seed"] for r in rows] == ["27000", "27001", "27000", "27001"]
    assert all(r["correct_target_cell"] in ("True", "False") for r in rows)
    disk = json.loads((target / "manifest.json").read_text())
    assert disk["source_sha256"] == report["source_sha256"]
    assert len(disk["source_sha256"]) == 64
    assert disk["summary"] == report["summary"]
    with pytest.raises(FileExistsError):
        phy_sweep.run_sweep(target, trials=2, fractions=(0.0,))


def test_invalid_sweep_config_never_creates_output(tmp_path):
    for opts in (
        {"trials": 0},
        {"trials": 999},
        {"fractions": ()},
        {"fractions": (0.5, 0.5)},
        {"fractions": (float("nan"),)},
        {"radar_snr_db": float("inf")},
        {"tx_antennas": 5, "user_rx_antennas": 4},
    ):
        out = tmp_path / f"invalid-{len(list(tmp_path.iterdir()))}"
        with pytest.raises(ValueError):
            phy_sweep.run_sweep(out, **opts)
        assert not out.exists()
