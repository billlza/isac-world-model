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
    """Explicit loss/cost exchange rate: m² of expected error per lost Mbit."""

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
    scores_m2: tuple[float | None, float | None, float | None]
    gains_m2: tuple[float, float, float]
    lost_mbit: tuple[float, float, float]
    delivery_ticks: tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "selected": self.action,
            "score_m2": list(self.scores_m2),
            "gain_m2": list(self.gains_m2),
            "lost_mbit": list(self.lost_mbit),
            "delivery_ticks": list(self.delivery_ticks),
            "interpretation": "Gaussian covariance proxy; not measured counterfactual benefit",
        }


def _validated_state(state: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, p = np.asarray(state, dtype=np.float64), np.asarray(cov, dtype=np.float64)
    if x.shape != (4,) or p.shape != (4, 4) or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("VoI requires a finite 4-state and 4x4 covariance")
    if not np.allclose(p, p.T, atol=1e-8, rtol=0):
        raise ValueError("VoI requires a symmetric covariance")
    if np.linalg.eigvalsh(p).min() < -1e-8:
        raise ValueError("VoI covariance must be positive semidefinite")
    return x, (p + p.T) / 2


def _conditional_capture_covariance(state: np.ndarray, cov: np.ndarray, action: int) -> np.ndarray:
    """Joseph-form covariance update using range/bearing geometry at mean state."""
    spec = ACTIONS[action]
    x, y = state[:2]
    r = max(float(math.hypot(x, y)), 1e-3)
    cos, sin = float(x / r), float(y / r)
    jac = np.array([[cos, -r * sin], [sin, r * cos]])
    rmat = jac @ np.diag([spec.range_sigma_m**2, spec.angle_sigma_rad**2]) @ jac.T
    rmat += np.eye(2) * 1e-6
    h = np.c_[np.eye(2), np.zeros((2, 2))]
    innovation = h @ cov @ h.T + rmat
    gain = np.linalg.solve(innovation, h @ cov).T
    kh = np.eye(4) - gain @ h
    p_post = kh @ cov @ kh.T + gain @ rmat @ gain.T
    return (p_post + p_post.T) / 2


def choose_sensing_by_voi(
    state: np.ndarray,
    covariance: np.ndarray,
    scene: SceneConfig,
    tokens: float,
    *,
    planned_actuation_wait_s: float = 0.0,
    configuration: VoIConfig | None = None,
) -> VoIDecision:
    """Select skip/quick/precise by *arrival-time* expected position-risk reduction.

    State/cov describe the time of the current decision. A measurement is
    captured only after computation/actuation finishes. It becomes usable after
    action-specific sensor delay (conservatively including jitter by default).
    Candidate gains average the covariance reduction across the SAME future
    physical evaluation times; gains are zero before the observation is usable.
    This baseline ignores other future camera/radio observations, measurement
    bias, correlated evidence and estimator miscalibration. It is deliberately
    not an oracle or a proof that sensing improves actual target error.
    """
    cfg = configuration or VoIConfig()
    x, p = _validated_state(state, covariance)
    finite(tokens, "tokens")
    finite(planned_actuation_wait_s, "planned_actuation_wait_s")
    if tokens < 0 or planned_actuation_wait_s < 0:
        raise ValueError("Budget and actuation wait must be non-negative")
    # The synthetic scene has a finite tick period; all action deadlines are
    # rounded UP to the next tick, exactly as EventQueue.release / streaming do.
    tick = scene.dt
    count = math.floor((cfg.horizon_s + 1e-12) / tick)
    if count < 1 or count > 10_000:
        raise ValueError("VoI horizon must span 1 to 10,000 scene ticks")
    wait_ticks = max(0, math.ceil(planned_actuation_wait_s / tick - 1e-12))
    actual_wait = wait_ticks * tick
    fc, qc = transition(actual_wait, cfg.acceleration_spectral_density)
    capture_state, capture_cov = fc @ x, fc @ p @ fc.T + qc
    gains, scores, lost, delays = [0.0] * 3, [0.0] * 3, [0.0] * 3, [0] * 3
    best = 0
    for action in (1, 2):
        spec = ACTIONS[action]
        extra_delay = scene.radio_delay_extra_s + (
            scene.delay_jitter_s if cfg.conservatively_delay_jitter else scene.delay_jitter_s / 2
        )
        # Negative delay is rejected by the sensor event validation, not treated as immediate evidence.
        if spec.delay_s + extra_delay < 0:
            raise ValueError("Radio availability delay must not be negative")
        delivery_tick = wait_ticks + max(0, math.ceil((spec.delay_s + extra_delay) / tick - 1e-12))
        delays[action] = delivery_tick
        lost[action] = (communication_bits(scene, 0) - communication_bits(scene, action)) / 1e6
        if spec.airtime_fraction > tokens + 1e-10:
            scores[action] = None
            continue
        if delivery_tick > count:
            scores[action] = -cfg.tradeoff_m2_per_mbit * lost[action]
            continue
        p_post = _conditional_capture_covariance(capture_state, capture_cov, action)
        delta = capture_cov - p_post
        cumulative = 0.0
        for future_tick in range(max(delivery_tick, 1), count + 1):
            dt = future_tick * tick - actual_wait
            if dt < -1e-10:
                continue
            f, _ = transition(dt, cfg.acceleration_spectral_density)
            reduced = f @ delta @ f.T
            cumulative += max(float(np.trace(reduced[:2, :2])), 0.0)
        gains[action] = cumulative / count
        scores[action] = gains[action] - cfg.tradeoff_m2_per_mbit * lost[action]
        if scores[action] is not None and scores[action] > scores[best] + 1e-12:
            best = action
    return VoIDecision(best, tuple(scores), tuple(gains), tuple(lost), tuple(delays))
