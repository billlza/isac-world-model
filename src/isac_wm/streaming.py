"""Causal virtual-time rollout with a single non-preemptive inference worker.

In-flight computation is evaluated immediately offline but its result, sensing
command, and monitor decision are embargoed until its declared finish time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .baselines import AirtimeBudget, ReplayKalman, guarded_risk_action, predictive_action, risk_budget_action
from .events import EventQueue, SnapshotBuffer
from .imm import IMMConfig, ReplayIMM
from .safety import MonitorConfig, assess
from .simulator import ToyScene, communication_bits
from .training import INPUT_KEYS
from .voi import VoIConfig, choose_sensing_by_voi


@dataclass
class _Result:
    start_s: float
    finish_s: float
    measured_s: float
    state: np.ndarray
    covariance: np.ndarray
    valid: np.ndarray
    age: np.ndarray
    requested_action: int
    observation_ids: list[int]
    voi_decision: dict | None = None


def propagate(state: np.ndarray, covariance: np.ndarray, dt: float, noise: float):
    """Constant velocity; preserve structured neural cross-covariance when supplied."""
    if dt < -1e-10:
        raise ValueError("Cannot publish an estimate before its state time")
    dt = max(0.0, dt)
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    q = noise * np.array(
        [
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt¶»§q«^