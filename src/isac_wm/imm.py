"""Causal interacting multiple-model (IMM) replay tracking baseline.

Mode bank: constant-velocity dynamics with LOW / NOMINAL / HIGH continuous
acceleration spectral densities. This matches the *form* of the synthetic
piecewise-acceleration scenario but not its actual unknown mode transitions.
Optional 2D innovation gating rejects grossly inconsistent observations.

This implementation is a classical comparator, not a learned world model,
not a globally optimal Bayesian smoother and not a quantum/ISAC waveform model.
Every cached state contains only data delivered by the query time. An OOS
measurement invalidates and recomputes the affected capture-time suffix.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from .baselines import measurement_xy, transition
from .events import Observation
from .validation import finite, integer


@dataclass(frozen=True)
class IMMConfig:
    """Fixed before testing; tuning must use a separate validation split."""

    process_noises: tuple[float, ...] = (0.05, 0.8, 6.0)
    switch_rate_hz: float = 0.6
    gate_threshold: float | None = None
    max_observations: int = 100_000

    def __post_init__(self):
        if not 2 <= len(self.process_noises) <= 8:
            raise ValueError("IMM requires between 2 and 8 modes")
        for noise in self.process_noises:
            finite(noise, "mode process noise", positive=True)
        finite(self.switch_rate_hz, "mode switch rate")
        if self.switch_rate_hz < 0:
            raise ValueError("Mode switch rate must be non-negative")
        if self.gate_threshold is not None:
            finite(self.gate_threshold, "IMM gate threshold", positive=True)
        integer(self.max_observations, "max_observations", 1)


class ReplayIMM:
    """IMM with Gaussian likelihoods, interaction, Joseph updates, and OOSM replay.

    The mode transition kernel depends on elapsed physical time, not the number
    of packets: P_ij(dt)=(1-e^-lambda*dt)/M + e^-lambda*dt*[i=j].
    The global NIS gate uses the *mixed predictive covariance*, including
    between-mode disagreement. Rejected measurements do not update modes.
    """

    def __init__(self, room_m: float, *, config: IMMConfig | None = None):
        self.room_m = finite(room_m, "room_m", positive=True)
        self.config = config or IMMConfig()
        self.observations: dict[int, Observation] = {}
        self._keys: list[tuple[float, int]] = []
        self._ordered: list[Observation] = []
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, int]] = []
        self._now = 0.0
        self.replayed_updates = 0
        self.rejected_measurements = 0
        self._h = np.c_[np.eye(2), np.zeros((2, 2))]

    def _initial(self):
        count = len(self.config.process_noises)
        initial = np.array([self.room_m / 2, self.room_m / 2, 0.0, 0.0])
        cov = np.diag([self.room_m**2 / 12, self.room_m**2 / 12, 1.0, 1.0])
        return (
            np.broadcast_to(initial, (count, 4)).copy(),
            np.broadcast_to(cov, (count, 4, 4)).copy(),
            np.full(count, 1.0 / count),
            0.0,
            0,
        )

    @staticmethod
    def _combine(xs: np.ndarray, ps: np.ndarray, weights: np.ndarray):
        x = weights @ xs
        d = xs - x
        p = np.sum(weights[:, None, None] * (ps + d[:, :, None] * d[:, None, :]), axis=0)
        return x, (p + p.T) / 2

    def _predict_modes(self, xs, ps, weights, dt):
        if dt < -1e-9:
            raise ValueError("Negative propagation interval")
        dt = max(0.0, dt)
        count = len(weights)
        stay = math.exp(-self.config.switch_rate_hz * dt)
        transition_modes = np.full((count, count), (1 - stay) / count)
        transition_modes[np.diag_indices(count)] += stay
        prior = weights @ transition_modes
        mix = weights[:, None] * transition_modes / prior[None, :]
        mixed_x = mix.T @ xs
        delta = xs[:, None, :] - mixed_x[None, :, :]
        mixed_p = np.einsum("ij,ijkl->jkl", mix, ps[:, None] + delta[..., :, None] * delta[..., None, :])
        pred_x = np.empty_like(xs)
        pred_p = np.empty_like(ps)
        for idx, q in enumerate(self.config.process_noises):
            f, noise = transition(dt, q)
            pred_x[idx] = f @ mixed_x[idx]
            pred_p[idx] = f @ mixed_p[idx] @ f.T + noise
        return pred_x, pred_p, prior

    def _update(self, xs, ps, weights, z, r):
        combined_x, combined_p = self._combine(xs, ps, weights)
        residual = z - self._h @ combined_x
        innovation = self._h @ combined_p @ self._h.T + r
        gate = self.config.gate_threshold
        if gate is not None and float(residual @ np.linalg.solve(innovation, residual)) > gate:
            return xs, ps, weights, True
        logs = np.empty(len(weights))
        updated_x, updated_p = xs.copy(), ps.copy()
        identity = np.eye(4)
        for j in range(len(weights)):
            residual_j = z - self._h @ xs[j]
            s = self._h @ ps[j] @ self._h.T + r
            sign, logdet = np.linalg.slogdet(s)
            if sign <= 0:
                raise FloatingPointError("IMM innovation covariance not positive definite")
            nis = float(residual_j @ np.linalg.solve(s, residual_j))
            logs[j] = math.log(max(weights[j], 1e-300)) - 0.5 * (nis + logdet + 2 * math.log(2 * math.pi))
            k = np.linalg.solve(s, self._h @ ps[j]).T
            updated_x[j] = xs[j] + k @ residual_j
            kh = identity - k @ self._h
            updated_p[j] = kh @ ps[j] @ kh.T + k @ r @ k.T
            updated_p[j] = (updated_p[j] + updated_p[j].T) / 2
        probs = np.exp(logs - np.max(logs))
        probs /= probs.sum()
        return updated_x, updated_p, probs, False

    def ingest(self, ready: list[Observation], now: float) -> None:
        finite(now, "now")
        if now < self._now:
            raise ValueError("Ingest time must be monotonic")
        ids = [o.uid for o in ready]
        if len(ids) != len(set(ids)) or set(ids).intersection(self.observations):
            raise ValueError("Duplicate evidence")
        if len(self.observations) + len(ready) > self.config.max_observations:
            raise OverflowError("IMM replay evidence budget exhausted")
        for obs in ready:
            if not isinstance(obs, Observation):
                raise TypeError("Expected Observation")
            if obs.available_s > now + 1e-9 or obs.capture_s < 0:
                raise ValueError("Future or negative acquisition time")
        self._now = now
        for obs in ready:
            key = (obs.capture_s, obs.uid)
            index = bisect_left(self._keys, key)
            self._keys.insert(index, key)
            self._ordered.insert(index, obs)
            self.observations[obs.uid] = obs
            del self._cache[index:]

    def estimate(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        finite(now, "now")
        if now < self._now:
            raise ValueError("Query precedes delivered evidence")
        self._now = now
        if self._cache:
            xs, ps, weights, time_s, rejected = self._cache[-1]
            xs, ps, weights = xs.copy(), ps.copy(), weights.copy()
        else:
            xs, ps, weights, time_s, rejected = self._initial()
        for obs in self._ordered[len(self._cache) :]:
            xs, ps, weights = self._predict_modes(xs, ps, weights, obs.capture_s - time_s)
            time_s = obs.capture_s
            measurement = measurement_xy(obs, self.room_m)
            if measurement is not None:
                xs, ps, weights, gated = self._update(xs, ps, weights, *measurement)
                rejected += int(gated)
            self.replayed_updates += 1
            self._cache.append((xs.copy(), ps.copy(), weights.copy(), time_s, rejected))
        self.rejected_measurements = rejected
        predicted_x, predicted_p, weights = self._predict_modes(xs, ps, weights, now - time_s)
        return self._combine(predicted_x, predicted_p, weights)

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Last delivered-evidence posterior, not a future-truth mode label."""
        return self._cache[-1][2].copy() if self._cache else self._initial()[2]
