"""Classical replay baseline and a transparent budgeted sensing heuristic."""
from __future__ import annotations
import math
import numpy as np
from .events import Observation
from .simulator import ACTIONS


def transition(dt: float, q: float = .8) -> tuple[np.ndarray, np.ndarray]:
    if dt < -1e-9:
        raise ValueError("Negative propagation interval")
    dt = max(0., dt)
    f = np.eye(4)
    f[:2, 2:] = np.eye(2)*dt
    noise = q*np.block([[np.eye(2)*dt**3/3, np.eye(2)*dt**2/2],
                        [np.eye(2)*dt**2/2, np.eye(2)*dt]])
    return f, noise


def measurement_xy(obs: Observation, room_m: float) -> tuple[np.ndarray, np.ndarray] | None:
    if obs.modality == "vision":
        im = obs.payload
        score = np.maximum(im[0]-np.maximum(im[1], im[2])-.2, 0)
        if score.sum() < .1:
            return None
        yy, xx = np.indices(score.shape)
        p = np.array([(xx*score).sum(), (yy*score).sum()])/score.sum()
        p = p/(im.shape[-1]-1)*room_m
        # Conservative pixel-grid proxy, not inferred from hidden target truth.
        sigma = room_m/(im.shape[-1]-1)*.4
        return p, np.eye(2)*sigma**2
    r, cs, sn, _ = map(float, obs.payload)
    r *= room_m*math.sqrt(2)
    spec = ACTIONS[obs.action]
    jacobian = np.array([[cs, -r*sn], [sn, r*cs]])
    cov = jacobian @ np.diag([spec.range_sigma_m**2, spec.angle_sigma_rad**2]) @ jacobian.T
    return np.array([r*cs, r*sn]), cov+np.eye(2)*1e-6


class ReplayKalman:
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
            if obs.available_s > now+1e-9:
                raise ValueError("Undelivered measurement")
            if obs.uid in self.observations:
                raise ValueError("Duplicate evidence")
            self.observations[obs.uid] = obs

    def estimate(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([self.room_m/2, self.room_m/2, 0., 0.])
        p = np.diag([self.room_m**2/12, self.room_m**2/12, 1., 1.])
        h = np.c_[np.eye(2), np.zeros((2, 2))]
        time = 0.
        for obs in sorted(self.observations.values(), key=lambda o: (o.capture_s, o.uid)):
            if obs.available_s > now+1e-9:
                raise ValueError("Query precedes evidence availability")
            measurement = measurement_xy(obs, self.room_m)
            if measurement is None:
                continue
            f, q = transition(obs.capture_s-time)
            x, p = f@x, f@p@f.T+q
            z, r = measurement
            innovation = h@p@h.T+r
            gain = np.linalg.solve(innovation, h@p).T
            x = x+gain@(z-h@x)
            ikh = np.eye(4)-gain@h
            p = ikh@p@ikh.T+gain@r@gain.T  # Joseph form
            p = (p+p.T)/2
            time = obs.capture_s
        f, q = transition(now-time)
        return f@x, f@p@f.T+q


class AirtimeBudget:
    """Token bucket: cumulative use never exceeds per_frame*(step+1)."""
    def __init__(self, per_frame: float = .1, burst: float = .6):
        if not 0 < per_frame <= 1 or burst < per_frame:
            raise ValueError("Invalid budget")
        self.rate, self.burst = per_frame, burst
        self.tokens, self.spent = 0., 0.

    def refill(self) -> None:
        self.tokens = min(self.burst, self.tokens+self.rate)

    def authorize(self, action: int) -> int:
        if action not in (0, 1, 2):
            raise ValueError("Invalid action")
        cost = ACTIONS[action].airtime_fraction
        if cost <= self.tokens+1e-10:
            self.tokens = max(0., self.tokens-cost)
            self.spent += cost
            return action
        return 0


def predictive_action(position_variance: np.ndarray, expected_age_s: float,
                      room_m: float, tokens: float) -> int:
    """Explicit scalar covariance-reduction PROXY, not learned counterfactual VoI.

    Uses forecast risk at an availability horizon. Delay discounts the benefit;
    sensing duration consumes communication airtime. The first release does not
    claim this heuristic is optimal or that the model's variance is calibrated.
    """
    variance = np.maximum(np.asarray(position_variance), 1e-8)
    best_action, best_score = 0, 0.
    for a in (1, 2):
        spec = ACTIONS[a]
        if spec.airtime_fraction > tokens+1e-10:
            continue
        # Mid-range angular-to-Cartesian approximation; deliberately documented.
        r = spec.range_sigma_m**2+(room_m*.5*spec.angle_sigma_rad)**2
        posterior = variance*r/(variance+r)
        gain = float((variance-posterior).sum())
        score = gain*math.exp(-(expected_age_s+spec.delay_s))-.12*spec.airtime_fraction
        if score > best_score:
            best_action, best_score = a, score
    return best_action
