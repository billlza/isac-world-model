"""Result validity monitor. This is not a physical-system safety certificate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .validation import finite


@dataclass(frozen=True)
class MonitorConfig:
    max_information_age_s: float = 0.6
    error_tolerance_m: float = 0.75
    compute_deadline_ms: float = 100.0
    allow_uncalibrated_acceptance: bool = False

    def __post_init__(self):
        for name in ("max_information_age_s", "error_tolerance_m", "compute_deadline_ms"):
            finite(getattr(self, name), name, positive=True)
        if not isinstance(self.allow_uncalibrated_acceptance, bool):
            raise ValueError("allow_uncalibrated_acceptance must be bool")

    def to_dict(self) -> dict:
        return asdict(self)


def assess(
    state: np.ndarray,
    variance: np.ndarray,
    valid: np.ndarray,
    age: np.ndarray,
    compute_ms: float,
    room_m: float,
    config: MonitorConfig,
    quantile: float | None = None,
) -> dict:
    """Acceptance never uses target truth. Calibration validity is checked upstream."""
    state, variance, valid, age = map(np.asarray, (state, variance, valid, age))
    if (
        state.shape != (4,)
        or variance.shape != (4,)
        or valid.shape != (2,)
        or age.shape != (2,)
        or not np.isfinite(state).all()
        or not np.isfinite(variance).all()
        or not np.isfinite(age).all()
        or not np.isfinite(valid).all()
        or (variance < 0).any()
        or (age < 0).any()
        or not np.isin(valid, [0, 1]).all()
        or not math.isfinite(compute_ms)
        or compute_ms < 0
        or not math.isfinite(room_m)
        or room_m <= 0
    ):
        return {"accepted": False, "status": "invalid_numeric", "radius_m": None}
    if compute_ms > config.compute_deadline_ms:
        return {"accepted": False, "status": "compute_deadline_missed", "radius_m": None}
    recent = (valid > 0) & (age <= config.max_information_age_s)
    if not recent.any():
        return {"accepted": False, "status": "no_recent_evidence", "radius_m": None}
    if (state[:2] < 0).any() or (state[:2] > room_m).any():
        return {"accepted": False, "status": "outside_model_domain", "radius_m": None}
    if quantile is None:
        radius = 2.448 * math.sqrt(float(max(variance[:2])))
        accepted = config.allow_uncalibrated_acceptance and radius <= config.error_tolerance_m
        return {"accepted": accepted, "status": "uncalibrated", "radius_m": radius}
    finite(quantile, "calibration quantile", positive=True)
    radius = quantile * math.sqrt(float(max(variance[:2])))
    accepted = radius <= config.error_tolerance_m
    return {"accepted": accepted, "status": "accepted" if accepted else "uncertain", "radius_m": radius}
