"""Classical replay baseline and a transparent budgeted sensing heuristic."""

from __future__ import annotations

import math
from bisect import bisect_left

import numpy as np

from .events import Observation
from .simulator import ACTIONS
from .validation import finite, integer


def transition(dt: float, q: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    finite(q, "process noise")
    if not math.isfinite(dt):
        raise ValueError("Non-finite propagation interval")
    if dt < -1e-9:
        raise ValueError("Negative propagation interval")
    dt = max(0.0, dt)
    f = np.eye(4)
    f[:2, 2:] = np.eye(2) * dt
    noise = q * np.block(
        [[np.eye(2) * dt**3 / 3, np.eye(2) * dt**2 / 2], [np.eye(2) * dt**2 / 2, np.eye(2) * dt]]
    )
    return f, noise


def measurement_xy(obs: Observation, room_m: float) -> tuple[np.ndarray, np.ndarray] | None:
    if obs.modality == "vision":
        im = obs.payload
        score = np.maximum(im[0] - np.maximum(im[1], im[2]) - 0.2, 0)
        if score.sum() < 0.1:
            return None
        yy, xx = np.indices(score.shape)
        p = np.array([(xx * score).sum(), (yy * score).sum()]) / score.sum()
        p = p / (im.shape[-1] - 1) * room_m
        # Conservative pixel-grid proxy, not inferred from hidden target truth.
        sigma = room_m / (im.shape[-1] - 1) * 0.4
        return p, np.eye(2) * sigma**2
    r, cs, sn, _ = map(float, obs.payload)
    r *= room_m * math.sqrt(2)
    spec = ACTIONS[obs.action]
    jacobian = np.array([[cs, -r * sn], [sn, r * cs]])
    cov = jacobian @ np.diag([spec.range_sigma_m**2, spec.angle_sigma_rad**2]) @ jacobian.T
    return np.array([r * cs, r * sn]), cov + np.eye(2) * 1e-6


class ReferenceReplayKalman:
    """Full-history causal OOSM reprocessing; intentionally not optimized.

    Replays ONLY observations already delivered at query time, sorted by their
    acquisition time, then predicts to now. This is a stronger timing baseline
    than simply fusing arrivals as current measurements. It uses a calibrated
    toy-camera centroid and polar-to-Cartesian radio conversion, not true state.
    Radio radial velocity is currently not used; this is not a final tuned IMM.
    """

    def __init__(self, room_m: float):
        self.room_m = room_m
        self.observations: dict[int, Observation] = {}

    def ingest(self, ready: list[Observation], now: float) -> None:
        for obs in ready:
            if obs.available_s > now + 1e-9:
                raise ValueError("Undelivered measurement")
            if obs.uid in self.observations:
                raise ValueError("Duplicate evidence")
            self.observations[obs.uid] = obs

    def estimate(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([self.room_m / 2, self.room_m / 2, 0.0, 0.0])
        p = np.diag([self.room_m**2 / 12, self.room_m**2 / 12, 1.0, 1.0])
        h = np.c_[np.eye(2), np.zeros((2, 2))]
        time = 0.0
        for obs in sorted(self.observations.values(), key=lambda o: (o.capture_s, o.uid)):
            if obs.available_s > now + 1e-9:
                raise ValueError("Query precedes evidence availability")
            measurement = measurement_xy(obs, self.room_m)
            if measurement is None:
                continue
            f, q = transition(obs.capture_s - time)
            x, p = f @ x, f @ p @ f.T + q
            z, r = measurement
            innovation = h @ p @ h.T + r
            gain = np.linalg.solve(innovation, h @ p).T
            x = x + gain @ (z - h @ x)
            ikh = np.eye(4) - gain @ h
            p = ikh @ p @ ikh.T + gain @ r @ gain.T  # Joseph form
            p = (p + p.T) / 2
            time = obs.capture_s
        f, q = transition(now - time)
        return f @ x, f @ p @ f.T + q


class ReplayKalman:
    """Exact cached-prefix causal replay, with optional innovation gating.

    Delivered observations are sorted by capture time. Inserting an OOS packet
    invalidates only its suffix. Query-time extrapolation never alters the cached
    posterior, so repeated queries do not double-count process noise or evidence.
    A hard evidence limit fails explicitly instead of silently deleting history.
    """

    def __init__(
        self,
        room_m: float,
        *,
        gate_threshold: float | None = None,
        max_observations: int = 100_000,
        process_noise: float = 0.8,
    ):
        self.room_m = finite(room_m, "room_m", positive=True)
        self.process_noise = finite(process_noise, "process_noise", positive=True)
        self.max_observations = integer(max_observations, "max_observations", 1)
        self.gate_threshold = (
            None if gate_threshold is None else finite(gate_threshold, "gate_threshold", positive=True)
        )
        self.observations: dict[int, Observation] = {}
        self._keys: list[tuple[float, int]] = []
        self._ordered: list[Observation] = []
        self._cache: list[tuple[np.ndarray, np.ndarray, float, int]] = []
        self._now = 0.0
        self.replayed_updates = 0
        self.rejected_measurements = 0
        self._h = np.c_[np.eye(2), np.zeros((2, 2))]

    def ingest(self, ready: list[Observation], now: float) -> None:
        finite(now, "now")
        if now < self._now:
            raise ValueError("Ingest time must be monotonic")
        ids = [o.uid for o in ready]
        if len(set(ids)) != len(ids) or set(ids).intersection(self.observations):
            raise ValueError("Duplicate evidence")
        if len(self.observations) + len(ready) > self.max_observations:
            raise OverflowError("Replay history budget exhausted")
        if any(o.available_s > now + 1e-9 for o in ready):
            raise ValueError("Undelivered measurement")
        self._now = now
        for o in ready:
            key = (o.capture_s, o.uid)
            index = bisect_left(self._keys, key)
            self._keys.insert(index, key)
            self._ordered.insert(index, o)
            self.observations[o.uid] = o
            del self._cache[index:]

    def estimate(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        finite(now, "now")
        if now < self._now:
            raise ValueError("Query precedes evidence availability or previous query")
        self._now = now
        if self._cache:
            x, p, time, rejected = self._cache[-1]
            x, p = x.copy(), p.copy()
        else:
            x = np.array([self.room_m / 2, self.room_m / 2, 0.0, 0.0])
            p = np.diag([self.room_m**2 / 12, self.room_m**2 / 12, 1.0, 1.0])
            time, rejected = 0.0, 0
        for obs in self._ordered[len(self._cache) :]:
            f, q = transition(obs.capture_s - time, self.process_noise)
            x, p = f @ x, f @ p @ f.T + q
            time = obs.capture_s
            measurement = measurement_xy(obs, self.room_m)
            if measurement is not None:
                z, r = measurement
                residual = z - self._h @ x
                innovation = self._h @ p @ self._h.T + r
                distance = float(residual @ np.linalg.solve(innovation, residual))
                if self.gate_threshold is not None and distance > self.gate_threshold:
                    rejected += 1
                else:
                    gain = np.linalg.solve(innovation, self._h @ p).T
                    x = x + gain @ residual
                    ikh = np.eye(4) - gain @ self._h
                    p = ikh @ p @ ikh.T + gain @ r @ gain.T
                    p = (p + p.T) / 2
            self.replayed_updates += 1
            self._cache.append((x.copy(), p.copy(), time, rejected))
        self.rejected_measurements = rejected
        f, q = transition(now - time, self.process_noise)
        return f @ x, f @ p @ f.T + q


class AirtimeBudget:
    """Token bucket: cumulative use never exceeds per_frame*(step+1)."""

    def __init__(self, per_frame: float = 0.1, burst: float = 0.6):
        finite(per_frame, "per_frame", positive=True)
        finite(burst, "burst", positive=True)
        if not 0 < per_frame <= 1 or burst < per_frame:
            raise ValueError("Invalid budget")
        self.rate, self.burst = per_frame, burst
        self.tokens, self.spent = 0.0, 0.0

    def refill(self) -> None:
        self.tokens = min(self.burst, self.tokens + self.rate)

    def authorize(self, action: int) -> int:
        integer(action, "action", 0, 2)
        if action not in (0, 1, 2):
            raise ValueError("Invalid action")
        cost = ACTIONS[action].airtime_fraction
        if cost <= self.tokens + 1e-10:
            self.tokens = max(0.0, self.tokens - cost)
            self.spent += cost
            return action
        return 0


def predictive_action(
    position_variance: np.ndarray, expected_age_s: float, room_m: float, tokens: float
) -> int:
    """Explicit scalar covariance-reduction PROXY, not learned counterfactual VoI.

    Uses forecast risk at an availability horizon. Delay discounts the benefit;
    sensing duration consumes communication airtime. The first release does not
    claim this heuristic is optimal or that the model's variance is calibrated.
    """
    finite(expected_age_s, "expected_age_s")
    finite(room_m, "room_m", positive=True)
    finite(tokens, "tokens")
    variance = np.asarray(position_variance, dtype=float)
    if variance.shape != (2,) or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("Position variance must be two finite non-negative values")
    variance = np.maximum(variance, 1e-8)
    best_action, best_score = 0, 0.0
    for a in (1, 2):
        spec = ACTIONS[a]
        if spec.airtime_fraction > tokens + 1e-10:
            continue
        # Mid-range angular-to-Cartesian approximation; deliberately documented.
        r = spec.range_sigma_m**2 + (room_m * 0.5 * spec.angle_sigma_rad) ** 2
        posterior = variance * r / (variance + r)
        gain = float((variance - posterior).sum())
        score = gain * math.exp(-(expected_age_s + spec.delay_s)) - 0.12 * spec.airtime_fraction
        if score > best_score:
            best_action, best_score = a, score
    return best_action


def risk_budget_action(
    position_variance: np.ndarray,
    velocity_variance: np.ndarray,
    tokens: float,
    room_m: float,
    *,
    extra_delay_s: float = 0.0,
    target_radius_m: float = 0.6,
    cost_weight: float = 0.25,
) -> int:
    """Arrival-risk heuristic with skip/quick/precise actions; NOT learned VoI.

    Unlike v0.1's always-positive proxy, uses a finite task-risk threshold and
    evaluates each action at its own delivery horizon. Covariances are diagonal
    proxies; missing cross terms and miscalibration must be tested separately.
    """
    p, v = np.asarray(position_variance), np.asarray(velocity_variance)
    if (
        p.shape != (2,)
        or v.shape != (2,)
        or not np.isfinite(p).all()
        or not np.isfinite(v).all()
        or (p < 0).any()
        or (v < 0).any()
    ):
        raise ValueError("Expected non-negative finite two-axis variances")
    for value, name in ((tokens, "tokens"), (extra_delay_s, "extra_delay_s"), (cost_weight, "cost_weight")):
        finite(value, name)
    finite(room_m, "room_m", positive=True)
    finite(target_radius_m, "target_radius_m", positive=True)
    target = target_radius_m**2
    best, best_score = 0, 0.0
    for a in (1, 2):
        spec = ACTIONS[a]
        if spec.airtime_fraction > tokens + 1e-10:
            continue
        horizon = spec.delay_s + extra_delay_s
        before = p + horizon * horizon * v + 0.8 * horizon**3 / 3
        noise = spec.range_sigma_m**2 + (room_m * 0.5 * spec.angle_sigma_rad) ** 2
        after = before * noise / (before + noise)
        avoided_excess = max(0.0, float(before.sum()) - target) - max(0.0, float(after.sum()) - target)
        score = avoided_excess - cost_weight * spec.airtime_fraction
        if score > best_score:
            best, best_score = a, score
    return best


def guarded_risk_action(
    position_variance: np.ndarray,
    velocity_variance: np.ndarray,
    tokens: float,
    room_m: float,
    valid: np.ndarray,
    age: np.ndarray,
    *,
    extra_delay_s: float = 0.0,
    max_radio_age_s: float = 0.25,
) -> int:
    """Heuristic with a freshness guard, not a probabilistic safety bound.

    Experimental variance policies can stop measuring when wrong but confident.
    This guard requests quick sensing when radio evidence is absent or stale.
    The airtime budget still has final authority and can deny the request.
    """
    finite(max_radio_age_s, "max_radio_age_s", positive=True)
    valid, age = np.asarray(valid), np.asarray(age)
    if (
        valid.shape != (2,)
        or age.shape != (2,)
        or not np.isin(valid, [0, 1]).all()
        or not np.isfinite(age).all()
        or (age < 0).any()
    ):
        raise ValueError("Invalid freshness metadata")
    request = risk_budget_action(
        position_variance, velocity_variance, tokens, room_m, extra_delay_s=extra_delay_s
    )
    if not valid[1] or age[1] >= max_radio_age_s:
        return 1 if tokens + 1e-10 >= ACTIONS[1].airtime_fraction else 0
    return request
