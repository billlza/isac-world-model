"""Independent-episode split conformal diagnostic for position uncertainty.

One score per independent episode: max normalized position error after startup.
Coverage is marginal over exchangeable *episodes* with an unchanged estimator,
policy and simulator distribution. It is NOT a guarantee of conditional accepted
risk, OOD robustness, or physical safety. Frames are not assumed independent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .artifacts import model_fingerprint, write_json
from .model import EventBeliefModel
from .runtime import RuntimeConfig
from .simulator import SceneConfig
from .validation import integer
from .voi import VoIConfig


def fit_episode_quantile(runs: list[dict], alpha: float = 0.1, scored_from: int = 5) -> dict:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    integer(scored_from, "scored_from", 0)
    ids = [r["seed"] for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Independent calibration episode IDs must be unique")
    n = len(runs)
    rank = math.ceil((n + 1) * (1 - alpha))
    if not n or rank > n:
        raise ValueError("Too few independent episodes for a finite conformal quantile")
    scores = []
    for run in runs:
        trace = run["trace"][scored_from:]
        if not trace:
            raise ValueError("No scored calibration frames")
        values = []
        fo¶»§q«^