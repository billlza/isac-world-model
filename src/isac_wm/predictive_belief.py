"""Causal multi-hypothesis radial dynamics memory; model-based research baseline.

The class updates hypotheses from actual arriving RF range / radial speed evidence,
never from hidden simulated truth. Vision bearings can flag contradictions, but
cannot establish object identity or material. Predictive likelihoods are assumed
Gaussian and not calibrated safety probabilities.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .evidence_reasoning import RFPoint, VisionBearing
from .simulator import ACTIONS


@dataclass(frozen=True)
class Hypothesis:
    name: str
    range_m: float
    speed_mps: float
    range_var: float
    speed_var: float
    log_weight: float

    def predict(self, dt: float, *, acceleration_var: float) -> "Hypothesis":
        if self.name == "static_reflector":
            return Hypothesis(
                self.name, self.range_m, 0.0, self.range_var + 0.03 * dt, self.speed_var, self.log_weight
            )
        return Hypothesis(
            self.name,
            max(0.0, self.range_m + dt * self.speed_mps),
            self.speed_mps,
            self.range_var + dt * dt * self.speed_var + acceleration_var * dt**3 / 3,
            self.speed_var + acceleration_var * dt,
            self.log_weight,
        )


class PredictiveBelief:
    """Two competing *dynamics* hypotheses with bounded evidence and abstention.

    A stationary reflector and a constant-radial-velocity scatterer compete.
    Outlier innovations cannot silently reset the state. History is bounded;
    unique IDs are tracked up to max_ids, after which a new episode is required.
    Updates are based on evidence AVAILABLE by now, not future measurements.
    """

    def __init__(
        self,
        *,
        max_history: int = 48,
        max_ids: int = 10000,
        acceleration_var: float = 0.5,
        innovation_gate: float = 16.0,
    ):
        if type(max_history) is not int or not 2 <= max_history <= 4096:
            raise ValueError("Invalid memory capacity")
        if type(max_ids) is not int or not max_history <= max_ids <= 1_000_000:
            raise ValueError("Invalid evidence ID budget")
        for x in (acceleration_var, innovation_gate):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("Dynamics and gate parameters must be finite positive")
        self.history = deque(maxlen=max_history)
        self.max_ids = max_ids
        self.seen: set[int] = set()
        self.acceleration_var = acceleration_var
        self.innovation_gate = innovation_gate
        self.now_s = 0.0
        self.hypotheses: tuple[Hypothesis, Hypothesis] | None = None
        self.last_capture_s = -math.inf
        self.last_rf: RFPoint | None = None
        self.contradictions = 0

    def _advance(self, now_s: float) -> None:
        if not math.isfinite(now_s) or now_s < self.now_s:
            raise ValueError("Non-monotone or nonfinite time")
        dt = now_s - self.now_s
        if self.hypotheses is not None:
            self.hypotheses = tuple(
                h.predict(dt, acceleration_var=self.acceleration_var) for h in self.hypotheses
            )  # type: ignore[assignment]
        self.now_s = now_s

    @staticmethod
    def _likelihood(residual: float, var: float) -> float:
        return -0.5 * (residual**2 / var + math.log(2 * math.pi * var))

    def ingest(self, item: RFPoint | VisionBearing, *, now_s: float) -> dict:
        if not isinstance(item, (RFPoint, VisionBearing)):
            raise TypeError("Only validated RF/visual observations are allowed")
        if not math.isfinite(now_s) or now_s < self.now_s:
            raise ValueError("Invalid or non-monotone reasoning time")
        if item.available_s > now_s + 1e-10:
            raise ValueError("Future evidence is not available")
        if item.uid in self.seen:
            raise ValueError("Duplicate observation id")
        if len(self.seen) >= self.max_ids:
            raise OverflowError("Evidence ID capacity reached; start new episode")
        # Range/speed updates describe CAPTURE time, not delivery time.
        # We do not pretend to implement a full out-of-sequence Bayesian smoother.
        # A packet captured before our existing state time is recorded but not fused.
        previous_clock = self.now_s
        self.seen.add(item.uid)
        if isinstance(item, VisionBearing):
            self._advance(now_s)
            if self.last_rf is not None and abs(item.capture_s - self.last_rf.capture_s) <= 0.15:
                sigma2 = item.bearing_sigma_rad**2 + self.last_rf.bearing_sigma_rad**2
                if (item.bearing_rad - self.last_rf.bearing_rad) ** 2 > self.innovation_gate * sigma2:
                    self.contradictions += 1
                    outcome = "visual_rf_bearing_conflict"
                else:
                    outcome = "angularly_compatible_identity_unknown"
            else:
                outcome = "unpaired_vision"
            self.history.append((item.uid, item.capture_s, now_s, outcome))
            return self.explain(last_event=outcome)
        if item.capture_s < max(self.last_capture_s, previous_clock) - 1e-10:
            self._advance(now_s)
            self.history.append((item.uid, item.capture_s, now_s, "late_rf_ignored"))
            return self.explain(last_event="late_rf_ignored")
        self.last_capture_s = item.capture_s
        self._advance(item.capture_s)
        if self.hypotheses is None:
            self.hypotheses = (
                Hypothesis(
                    "static_reflector", item.range_m, 0.0, item.range_sigma_m**2, item.speed_sigma_mps**2, 0.0
                ),
                Hypothesis(
                    "moving_scatterer",
                    item.range_m,
                    item.radial_speed_mps,
                    item.range_sigma_m**2,
                    item.speed_sigma_mps**2,
                    0.0,
                ),
            )
            outcome = "initialized_two_motion_hypotheses"
        else:
            candidates = []
            gated = 0
            for h in self.hypotheses:
                rv = h.range_var + item.range_sigma_m**2
                vv = h.speed_var + item.speed_sigma_mps**2
                dr = item.range_m - h.range_m
                dv = item.radial_speed_mps - h.speed_mps
                nis = dr * dr / rv + dv * dv / vv
                if nis > self.innovation_gate:
                    gated += 1
                    candidates.append(
                        Hypothesis(
                            h.name, h.range_m, h.speed_mps, h.range_var, h.speed_var, h.log_weight - 4.0
                        )
                    )
                    continue
                kr, kv = h.range_var / rv, h.speed_var / vv
                candidates.append(
                    Hypothesis(
                        h.name,
                        h.range_m + kr * dr,
                        0.0 if h.name == "static_reflector" else h.speed_mps + kv * dv,
                        max((1 - kr) * h.range_var, 1e-8),
                        max((1 - kv) * h.speed_var, 1e-8),
                        h.log_weight + self._likelihood(dr, rv) + self._likelihood(dv, vv),
                    )
                )
            if gated == 2:
                self.contradictions += 1
                outcome = "rf_outlier_both_hypotheses_rejected"
            else:
                outcome = "rf_corrected_with_prediction_residual"
            largest = max(h.log_weight for h in candidates)
            # Renormalization avoids numerical overflow during long episodes.
            self.hypotheses = tuple(
                Hypothesis(h.name, h.range_m, h.speed_mps, h.range_var, h.speed_var, h.log_weight - largest)
                for h in candidates
            )  # type: ignore[assignment]
        self._advance(now_s)
        if outcome != "rf_outlier_both_hypotheses_rejected":
            self.last_rf = item
        self.history.append((item.uid, item.capture_s, now_s, outcome))
        return self.explain(last_event=outcome)

    def predict_observation(
        self, horizon_s: float, *, range_sigma_m: float = 3.0, speed_sigma_mps: float = 0.5
    ) -> dict:
        if not math.isfinite(horizon_s) or not 0 <= horizon_s <= 60:
            raise ValueError("Invalid prediction horizon")
        if (
            not math.isfinite(range_sigma_m)
            or range_sigma_m <= 0
            or not math.isfinite(speed_sigma_mps)
            or speed_sigma_mps <= 0
        ):
            raise ValueError("Invalid measurement noise assumptions")
        if self.hypotheses is None:
            return {"status": "unknown_no_evidence", "hypotheses": []}
        weights = [math.exp(h.log_weight) for h in self.hypotheses]
        denom = sum(weights)
        predictions = []
        for h, w in zip(self.hypotheses, weights):
            p = h.predict(horizon_s, acceleration_var=self.acceleration_var)
            predictions.append(
                {
                    "hypothesis": h.name,
                    "range_m": p.range_m,
                    "radial_speed_mps": p.speed_mps,
                    "range_sigma_m_assumed": math.sqrt(p.range_var + range_sigma_m**2),
                    "speed_sigma_mps_assumed": math.sqrt(p.speed_var + speed_sigma_mps**2),
                    "relative_weight_uncalibrated": w / denom,
                }
            )
        return {
            "status": "predicted_conditional_on_hypotheses",
            "horizon_s": horizon_s,
            "hypotheses": predictions,
            "identity": "unknown",
            "material": "unknown",
        }

    def forecast_actions(self, horizon_s: float, *, actions: tuple[int, ...] = (0, 1, 2)) -> dict:
        """Predict measurements for each resource action, *not* their realized values.

        Action 0 means no packet. Quick/precise inherit the toy action noise and
        nominal delivery delay; this is not a calibrated RF latency model.
        No action is executed or read from future simulation truth.
        """
        if not isinstance(actions, tuple) or len(set(actions)) != len(actions):
            raise ValueError("Actions must be unique tuple entries")
        output = {}
        for action in actions:
            if type(action) is not int or action not in (0, 1, 2):
                raise ValueError("Unknown sensing action")
            if action == 0:
                output["skip"] = {"packet": None, "resource_fraction": 0.0}
                continue
            spec = ACTIONS[action]
            predicted = self.predict_observation(
                horizon_s,
                range_sigma_m=spec.range_sigma_m,
                speed_sigma_mps=spec.velocity_sigma_mps,
            )
            output[spec.name] = {
                "prediction": predicted,
                "capture_time_s": self.now_s + horizon_s,
                "earliest_available_s": self.now_s + horizon_s + spec.delay_s,
                "resource_fraction": spec.airtime_fraction,
                "bearing_prediction": "unknown_without_angular_dynamics",
            }
        return {"hypotheses_by_action": output, "future_truth_used": False}

    def explain(self, *, last_event: str | None = None) -> dict:
        return {
            "clock_s": self.now_s,
            "evidence_count": len(self.history),
            "last_event": last_event,
            "contradictions": self.contradictions,
            "prediction": self.predict_observation(0.2),
            "object_identity": "unknown",
            "object_material": "unknown",
            "safety_note": "Assumed Gaussian motion hypotheses, not certified or calibrated; no semantic recognition.",
        }
