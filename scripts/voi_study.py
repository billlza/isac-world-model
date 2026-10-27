"""Paired real-timing simulation for the Gaussian arrival-time VoI comparator.

This is not hyperparameter selection, independent hardware verification, or a
claim of improved error. All methods receive paired synthetic truth/noise keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from isac_wm.evaluation import rollout
from isac_wm.runtime import RuntimeConfig
from isac_wm.scenarios import scenario_config
from isac_wm.simulator import SceneConfig
from isac_wm.voi import VoIConfig

SOURCE_FILES = (
    "src/isac_wm/voi.py",
    "src/isac_wm/evaluation.py",
    "src/isac_wm/streaming.py",
    "src/isac_wm/simulator.py",
    "scripts/voi_study.py",
)


def run_study(
    out: Path, *, episodes: int = 12, steps: int = 48, seed: int = 80_000_000, voi_tradeoff: float = 0.4
) -> dict:
    if not 1 <= episodes <= 500 or not 8 <= steps <= 10_000 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Invalid study dimensions")
    voi_config = VoIConfig(tradeoff_m2_per_mbit=voi_tradeoff)
    if out.exists():
        raise FileExistsError(f"Study output exists: {out}")
    out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    configurations = ("nominal", "maneuver", "radio_outliers", "radio_bias", "camera_loss")
    policies = ("fixed",¶»§q«^