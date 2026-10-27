"""Delay-aware *model-based* value-of-information baseline for the toy ISAC task.

The counterfactual is a *Gaussian covariance projection*, not an observed outcome
of an unexecuted beam or a learned/verified safety guarantee. The same finite
prediction horizon is used for all actions. No future truth or future packets enter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .baselines import transition
from .simulator import ACTIONS, SceneConfig, communication_bits
from .validation import finite


@dataclass(frozen=True)
class VoIConfig:
    """Explicit loss/cost exchange rate: mÂ² of expected error per lost Mbit."""

    horizon_s: float = 0.8
    tradeoff_m2_per_mbit: float = 0.4
    acceleration_spectral_density: float = 0.8
    conservatively_delay_jitter: bool = True

    def __post_init__(self) -> None:
        finite(self.horizon_s, "horizon_s", positive=True)
        finite(self.tradeoff_m2_per_mbit, "tradeoff_m2_per_mbit")
        finite(self.acceleration_spectral_density, "acceleration_spectral_density")
        if self.tradeoff_m2_per_mbit < 0 or self.acceleration_spectral_density < 0:
            raise ValueError("VoI tradeoff and acceleration variance must not be negative")
        if not isinstance(self.conservatively_delay_jitter, bool):
            raise ValueError("conservatively_delay_jitter must be bool")


@dataclass(frozen=True)
class VoIDecision:
    action: int
    scores_m2: tuple[float | ¶»§q«^