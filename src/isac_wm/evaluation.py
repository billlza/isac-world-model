from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import write_json
from .baselines import AirtimeBudget, ReplayKalman, guarded_risk_action, predictive_action, risk_budget_action
from .events import EventQueue, SnapshotBuffer
from .imm import IMMConfig, ReplayIMM
from .model import EventBeliefModel
from .runtime import RuntimeConfig
from .safety import MonitorConfig, assess
from .simulator import SceneConfig, ToyScene, communication_bits
from .training import INPUT_KEYS
from .validation import integer
from .voi import VoIConfig, choose_sensing_by_voi


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _reference_rollout(
    seed: int,
    config: SceneConfig,
    estimator: str,
    policy: str,
    model: EventBeliefModel | None = None,
    device: torch.device | None = None,
    *,
    monitor: MonitorConfig | None = None,
    quantile: float | None = None,
    sensor_budget: float = 0.1,
    gate_threshold: float | None = None,
    baseline_process_noise: float = 0.8,
    imm_config: IMMConfig | None = None,
    voi_config: VoIConfig | None = None,
) -> dict:
    if estimator not in {"kalman", "kalman-gated", "imm", "imm-gated", "model"} or policy not in {
        "fixed",
        "skip",
        "predictive",
        "risk",
        "guarded",
        "voi",
  ¶»§q«^