"""Small action/observation experiment, separate from the ToyScene checkpoints.

Only the simulator owns world truth. A hypothesis is a proposed depth, never an
inferred material or identity. Scores and rejection thresholds are experimental,
not calibrated probabilities or safety guarantees. Camera motion does not alter
the exogenous radar/visual target trajectories.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from torch import nn

from .evidence_reasoning import detect_visual_bearing, from_phy_result, render_synthetic_camera
from .observation_planning import PLANNERS, EvidenceState, verification_action
from .phy_ofdm import OFDMConfig, evaluate_frame

SCENARIOS = ("same", "different", "unresolvable", "outside_candidates", "camera_loss", "late")
METHODS = ("geometry_voi", "learned_memory", "learned_no_memory", "fixed", "random", "always_unknown")
HYPOTHESES = ("same_depth", "farther_object")


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int = 4
    slot_s: float = 2.0
    camera_speed_mps: float = 8.0
    delivery_s: float = 0.25
    exposure_s: float = 0.05
    max_baseline_m: float = 4.0
    travel_budget_m: float = 12.0
    packet_bits: int = 1024
    alternative_depth_m: float = 7.5
    camera_sigma_deg: float = 0.45
    confirmation_score: float = 0.95
    rejection_sigma: float = 3.0

    def __post_init__(self):
        if type(self.steps) is not int or not 2 <= self.steps <= 16:
            raise ValueError("steps must be in [2,16]")
        if type(self.packet_bits) is not int or not 1 <= self.packet_bits <= 8192:
            raise ValueError("packet_bits must be in [1,8192]")
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0.5 < self.confirmation_score < 1 or self.rejection_sigma < 2:
            raise ValueError("Invalid evidence thresholds")
        if 2 * self.max_baseline_m / self.camera_speed_mps + self.exposure_s >= self.slot_s:
            raise ValueError("A camera movement must finish before the next decision slot")


@dataclass(frozen=True)
class Anchor:
    range_m: float
    bearing_deg: float
    radial_speed_mps: float
    available_s: float
    range_sigma_m: float = 1.2
    bearing_sigma_deg: float = 0.6
    speed_sigma_mps: float = 0.3

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Non-finite anchor")
        if self.range_m <= 0 or self.available_s < 0 or abs(self.bearing_deg) > 60:
            raise ValueError("Invalid anchor geometry/time")
        if min(self.range_sigma_m, self.bearing_sigma_deg, self.speed_sigma_mps) <= 0:
            raise ValueError("Anchor uncertainty must be positive")


def observation_geometry(anchor: Anchor, lateral_m: float, capture_s: float, cfg: ExperimentConfig):
    """Candidate means and uncertainty from the observed RF anchor only.

    The camera translates perpendicular to the anchor's measured ray. Static
    range/bearing errors are shared across frames; the classical comparator's
    repeated likelihood product is an approximation, explicitly not calibrated.
    """
    if not math.isfinite(lateral_m) or not math.isfinite(capture_s) or capture_s < anchor.available_s:
        raise ValueError("Prediction must follow RF availability")
    ranges = anchor.range_m + anchor.radial_speed_mps * capture_s + np.array([0, cfg.alternative_depth_m])
    if np.any(ranges <= 1):
        raise ValueError("Candidate outside the forward geometry")
    mean = anchor.bearing_deg - np.degrees(np.arctan2(lateral_m, ranges))
    derivative = np.degrees(lateral_m / (ranges**2 + lateral_m**2))
    sigma = np.sqrt(
        cfg.camera_sigma_deg**2
        + anchor.bearing_sigma_deg**2
        + derivative**2 * (anchor.range_sigma_m**2 + (capture_s * anchor.speed_sigma_mps) ** 2)
    )
    features = np.stack(
        (mean / 60, ranges / 40, np.full(2, lateral_m / 4), np.full(2, capture_s / 10)), axis=-1
    ).astype(np.float32)
    return mean, sigma, features


class ObservationNetwork(nn.Module):
    """Random-initialized residual distribution with optional recurrent evidence.

    This learns from extracted perspective bearings, not RGB semantic features;
    it is a small test of conditional observation learning, not a complete neural
    world model. Each candidate has its own recurrent memory.
    """

    def __init__(self, use_memory: bool = True):
        super().__init__()
        self.use_memory = use_memory
        self.hidden_size = 12
        self.head = nn.Sequential(nn.Linear(16, 24), nn.SiLU(), nn.Linear(24, 2))
        self.update = nn.GRUCell(3, self.hidden_size)

    def predict(self, features, base_mean, base_sigma, memory):
        hidden = memory if self.use_memory else torch.zeros_like(memory)
        raw = self.head(torch.cat((features, hidden), -1))
        return base_mean + 1.5 * raw[..., 0].tanh(), base_sigma * (0.7 * raw[..., 1].tanh()).exp()

    def observe(self, features, base_mean, observation, memory):
        inputs = torch.stack((((observation - base_mean) / 5).tanh(), features[..., 2], features[..., 3]), -1)
        shape = memory.shape
        updated = self.update(inputs.reshape(-1, 3), memory.reshape(-1, self.hidden_size)).reshape(shape)
        return updated if self.use_memory else torch.zeros_like(updated)


class HypothesisMemory:
    def __init__(self, anchor: Anchor, cfg: ExperimentConfig, network: ObservationNetwork | None = None):
        self.anchor, self.cfg, self.network = anchor, cfg, network
        # Deliberately biased starting belief, shared by every comparator.
        self.log_weights = np.log([0.8, 0.2])
        self.hidden = torch.zeros(2, 12)
        self.now_s = anchor.available_s
        self.seen: set[int] = set()
        self.evidence = EvidenceState()
        self.predictive_nll: list[float] = []

    @property
    def accepted(self):
        return self.evidence.accepted

    @property
    def informative(self):
        return self.evidence.informative

    @property
    def contradictions(self):
        return self.evidence.contradictions

    def weights(self):
        p = np.exp(self.log_weights - self.log_weights.max())
        return p / p.sum()

    def predict(self, lateral_m: float, capture_s: float):
        if capture_s < self.now_s:
            raise ValueError("Cannot predict a past observation")
        mean, sigma, features = observation_geometry(self.anchor, lateral_m, capture_s, self.cfg)
        if self.network is not None:
            with torch.no_grad():
                mean_t, sigma_t = self.network.predict(
                    torch.tensor(features),
                    torch.tensor(mean, dtype=torch.float32),
                    torch.tensor(sigma, dtype=torch.float32),
                    self.hidden,
                )
            mean, sigma = mean_t.numpy().astype(float), sigma_t.numpy().astype(float)
        if not np.isfinite(mean).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise FloatingPointError("Invalid predictive distribution")
        return mean, sigma

    def release(self, event: dict, now_s: float):
        """Only a released packet may change belief. No simulator labels accepted.

        The saved prediction was made at action selection; delayed packets are
        scored against that prediction, not against a hindsight reconstruction.
        A single outstanding packet is enforced by the experiment scheduler.
        """
        if not np.isfinite([now_s, event["capture_s"], event["available_s"], event["lateral_m"]]).all():
            raise ValueError("Non-finite packet time or viewpoint")
        if now_s < self.now_s or now_s < event["available_s"]:
            raise ValueError("Future packet or backward clock")
        if event["capture_s"] < self.anchor.available_s or event["available_s"] < event["capture_s"]:
            raise ValueError("Invalid packet timing")
        if event["uid"] in self.seen:
            raise ValueError("Duplicate evidence")
        if event["observation_deg"] is None:
            self.seen.add(event["uid"])
            self.now_s = now_s
            return
        obs = float(event["observation_deg"])
        mean, sigma = np.array(event["prediction_mean_deg"]), np.array(event["prediction_sigma_deg"])
        if mean.shape != (2,) or sigma.shape != (2,) or not np.isfinite([obs, *mean, *sigma]).all():
            raise ValueError("Invalid released observation")
        if np.any(sigma <= 0):
            raise ValueError("Invalid predictive sigma")
        base, base_sigma, features = observation_geometry(
            self.anchor, event["lateral_m"], event["capture_s"], self.cfg
        )
        self.seen.add(event["uid"])
        self.now_s = now_s
        log_likelihood = -0.5 * ((obs - mean) / sigma) ** 2 - np.log(sigma * math.sqrt(2 * math.pi))
        terms = np.log(self.weights().clip(1e-30)) + log_likelihood
        self.predictive_nll.append(float(-(terms.max() + np.log(np.exp(terms - terms.max()).sum()))))
        compatible = bool(np.min(np.abs(obs - mean) / sigma) <= self.cfg.rejection_sigma)
        # A learned residual must not manufacture physical identifiability.
        # The proposed views must distinguish candidates under the declared
        # geometric error floors, independently of neural confidence.
        self.evidence = self.evidence.observe(
            compatible=compatible,
            informative=bool(abs(base[0] - base[1]) > 2 * max(base_sigma)),
            lateral_m=event["lateral_m"],
        )
        # An all-candidates-incompatible packet remains in the audit/NLL stream,
        # but must not strengthen the least-wrong candidate or its neural state.
        if not compatible:
            return
        self.log_weights += log_likelihood
        # Normalize immediately, keeping finite values across long evidence sequences.
        self.log_weights -= self.log_weights.max()
        if self.network is not None:
            with torch.no_grad():
                self.hidden = self.network.observe(
                    torch.tensor(features),
                    torch.tensor(base, dtype=torch.float32),
                    torch.tensor(obs, dtype=torch.float32),
                    self.hidden,
                )

    def status(self):
        if self.evidence.rejected:
            return "candidate_set_rejected"
        if not self.evidence.ready_to_confirm:
            return "unknown"
        p = self.weights()
        return HYPOTHESES[int(p.argmax())] if p.max() >= self.cfg.confirmation_score else "unknown"


def expected_error_reduction(prior: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> float:
    """Two-candidate Bayes classification-error reduction by deterministic quadrature.

    Model-based proxy only: this has no calibrated all-candidates-wrong state.
    The separate rejection gate is evaluated on actual released evidence.
    """
    grid = np.linspace(float(np.min(means - 7 * sigmas)), float(np.max(means + 7 * sigmas)), 401)
    density = np.exp(-0.5 * ((grid[:, None] - means) / sigmas) ** 2) / (sigmas * math.sqrt(2 * math.pi))
    error_after = float(np.trapezoid(np.min(density * prior, axis=1), grid))
    return max(0.0, float(prior.min()) - error_after)


def choose_action(
    memory: HypothesisMemory,
    method: str,
    *,
    decision_s: float,
    position_m: float,
    remaining_m: float,
    deadline_s: float,
    step: int,
    seed: int,
    planner: str = "verification-aware",
):
    """Plan from released evidence and public costs; no world/scenario argument."""
    if method not in METHODS:
        raise ValueError("Unknown method")
    if planner not in PLANNERS:
        raise ValueError("Unknown planner")
    if not np.isfinite([decision_s, position_m, remaining_m, deadline_s]).all():
        raise ValueError("Non-finite planning inputs")
    if decision_s < memory.now_s or remaining_m < 0 or deadline_s <= decision_s:
        raise ValueError("Invalid planning time/budget")
    cfg = memory.cfg
    proposals = []
    for lateral in (-cfg.max_baseline_m, 0.0, cfg.max_baseline_m):
        travel = abs(lateral - position_m)
        capture = decision_s + travel / cfg.camera_speed_mps + cfg.exposure_s
        available = capture + cfg.delivery_s
        if travel > remaining_m + 1e-9 or available > deadline_s:
            continue
        mean, sigma = memory.predict(lateral, capture)
        base, base_sigma, _ = observation_geometry(memory.anchor, lateral, capture, cfg)
        gain = expected_error_reduction(memory.weights(), mean, sigma)
        # Arrival-time usefulness over the remaining common evaluation horizon.
        gain *= max(0, deadline_s - available) / max(1e-9, deadline_s - decision_s)
        proposals.append(
            {
                "lateral_m": lateral,
                "travel_m": travel,
                "capture_s": capture,
                "available_s": available,
                "prediction_mean_deg": mean.tolist(),
                "prediction_sigma_deg": sigma.tolist(),
                "expected_error_reduction_proxy": gain,
                "score": gain - 0.002 - 0.001 * travel,
                "geometrically_informative": bool(abs(base[0] - base[1]) > 2 * max(base_sigma)),
            }
        )
    if method == "always_unknown" or not proposals:
        return None, proposals
    if method == "fixed":
        target = cfg.max_baseline_m if step % 2 == 0 else -cfg.max_baseline_m
        chosen = min(proposals, key=lambda p: abs(p["lateral_m"] - target))
    elif method == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 981]))
        chosen = proposals[int(rng.integers(len(proposals)))]
    else:
        if memory.status() == "candidate_set_rejected":
            return None, proposals
        if planner == "verification-aware" and memory.evidence.pending_verification is not None:
            return verification_action(memory.evidence, proposals), proposals
        chosen = max(proposals, key=lambda p: p["score"])
        if chosen["score"] <= 0:
            return None, proposals
    return {
        **chosen,
        "selection_reason": method if method in {"fixed", "random"} else "positive_voi",
        "verification_view_separation_m": None,
        "verification_same_view": None,
    }, proposals


@dataclass(frozen=True)
class SyntheticWorld:
    """Simulation/evaluation-only data; never pass this object to a predictor."""

    seed: int
    scenario: str
    phy: OFDMConfig
    visual_depth_offset_m: float
    camera_bias_deg: float

    @classmethod
    def create(cls, seed: int, scenario: str):
        if scenario not in SCENARIOS or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Invalid world seed/scenario")
        rng = np.random.default_rng(seed)
        phy = OFDMConfig(
            target_range_m=float(rng.uniform(17, 23)), target_bearing_deg=float(rng.uniform(-20, 20))
        )
        offset = 0.0 if scenario == "same" else (-9.0 if scenario == "outside_candidates" else 7.5)
        return cls(seed, scenario, phy, offset, float(rng.uniform(-0.6, 0.6)))

    def initial_rf(self):
        report = evaluate_frame(self.phy, probe_fraction=0.2, seed=self.seed)
        available = self.phy.symbols * self.phy.pulse_period_s + 0.035
        rf = from_phy_result(report, uid=0, frame=0, capture_s=0, available_s=available)
        return Anchor(rf.range_m, math.degrees(rf.bearing_rad), rf.radial_speed_mps, available), report

    def measure(self, anchor: Anchor, action: dict, step: int, cfg: ExperimentConfig):
        capture = action["capture_s"]
        angle = math.radians(self.phy.target_bearing_deg)
        r = self.phy.target_range_m + self.phy.target_radial_speed_mps * capture + self.visual_depth_offset_m
        target = r * np.array([math.cos(angle), math.sin(angle)])
        ray = math.radians(anchor.bearing_deg)
        camera = action["lateral_m"] * np.array([-math.sin(ray), math.cos(ray)])
        delta = target - camera
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 191]))
        bearing = (
            math.degrees(math.atan2(delta[1], delta[0])) + self.camera_bias_deg + float(rng.normal(0, 0.25))
        )
        lost = self.scenario == "camera_loss"
        image = render_synthetic_camera(bearing, seed=int(rng.integers(2**32)), occluded=lost)
        detection = detect_visual_bearing(
            image,
            uid=step + 1,
            frame=step,
            capture_s=capture,
            available_s=action["available_s"],
            assumed_sigma_deg=cfg.camera_sigma_deg,
        )
        return None if detection is None else math.degrees(detection.bearing_rad)


def run_episode(
    world: SyntheticWorld,
    method: str,
    cfg: ExperimentConfig,
    *,
    anchor: Anchor,
    phy_report: dict,
    network: ObservationNetwork | None = None,
    planner: str = "verification-aware",
):
    if method not in METHODS:
        raise ValueError("Unknown method")
    if planner not in PLANNERS:
        raise ValueError("Unknown planner")
    if method.startswith("learned") and network is None:
        raise ValueError("Learned method requires an explicitly trained model")
    effective = replace(cfg, max_baseline_m=0.1) if world.scenario == "unresolvable" else cfg
    effective = replace(effective, delivery_s=3.0) if world.scenario == "late" else effective
    memory = HypothesisMemory(anchor, effective, network)
    start, position, travelled = anchor.available_s, 0.0, 0.0
    deadline = start + effective.steps * effective.slot_s
    pending = None
    trace, captures = [], []
    initial_bits = int(phy_report["communication"]["payload_bits"])
    if effective.packet_bits > initial_bits:
        raise ValueError("Camera packet exceeds one reserved downlink payload frame")

    def snapshot(t, phase, event=None):
        trace.append(
            {
                "time_s": t,
                "phase": phase,
                "step": -1 if event is None else event["uid"] - 1,
                "weights": memory.weights().tolist(),
                "status": memory.status(),
                "accepted_observations": memory.accepted,
                "contradictions": memory.contradictions,
                "contradiction_streak": memory.evidence.contradiction_streak,
                "supporting_informative": memory.evidence.supporting_informative,
                "pending_verification": memory.evidence.pending_verification,
                "lateral_m": position,
                "travelled_m": travelled,
                "observation_deg": None if phase != "arrival" or event is None else event["observation_deg"],
            }
        )

    snapshot(start, "initial")
    for step in range(effective.steps):
        now = start + step * effective.slot_s
        if pending is not None and pending["available_s"] <= now:
            memory.release(pending, pending["available_s"])
            snapshot(pending["available_s"], "arrival", pending)
            pending = None
        if pending is not None:
            snapshot(now, "waiting")
            continue
        chosen, proposals = choose_action(
            memory,
            method,
            decision_s=now,
            position_m=position,
            remaining_m=effective.travel_budget_m - travelled,
            deadline_s=deadline,
            step=step,
            seed=world.seed,
            planner=planner,
        )
        if chosen is None:
            snapshot(now, "skip")
            continue
        chosen.update(
            uid=step + 1, decision_s=now, candidates=proposals, weights_before=memory.weights().tolist()
        )
        # Prediction exists before the simulator is invoked. The returned
        # observation stays embargoed in pending until its availability time.
        snapshot(now, "prediction", chosen)
        chosen["observation_deg"] = world.measure(anchor, chosen, step, effective)
        chosen["missed_next_decision_deadline"] = chosen["available_s"] > now + effective.slot_s
        travelled += chosen["travel_m"]
        position = chosen["lateral_m"]
        snapshot(chosen["capture_s"], "capture", chosen)
        captures.append(chosen)
        pending = chosen
    if pending is not None and pending["available_s"] <= deadline:
        memory.release(pending, pending["available_s"])
        snapshot(pending["available_s"], "arrival", pending)
    snapshot(deadline, "end")
    truth = (
        "same_depth"
        if world.visual_depth_offset_m == 0
        else (
            "farther_object"
            if world.visual_depth_offset_m == effective.alternative_depth_m
            else "outside_candidates"
        )
    )
    final = memory.status()
    confirmed = final in HYPOTHESES
    correct_times = [r["time_s"] for r in trace if r["status"] == truth]
    nll = memory.predictive_nll
    metrics = {
        "confirmed": confirmed,
        "wrong_confirmation": confirmed and final != truth,
        "ever_wrong_confirmation": any(r["status"] in HYPOTHESES and r["status"] != truth for r in trace),
        "abstained": not confirmed,
        "candidate_set_rejected": final == "candidate_set_rejected",
        "first_correct_confirmation_s": min(correct_times) if correct_times else None,
        "mean_predictive_nll": float(np.mean(nll)) if nll else None,
        "observations_received": memory.accepted,
        "missing_detections": sum(c["observation_deg"] is None for c in captures),
        "late_packets": sum(c["missed_next_decision_deadline"] for c in captures),
        "capture_count": len(captures),
        "travel_m": travelled,
        "camera_transport_bits_reserved": len(captures) * effective.packet_bits,
        "remaining_raw_payload_bits_proxy": initial_bits * effective.steps
        - len(captures) * effective.packet_bits,
        "final_status": final,
    }
    return {
        "scenario": world.scenario,
        "method": method,
        "planner": planner,
        "seed": world.seed,
        "config": asdict(effective),
        "anchor": asdict(anchor),
        "evaluation_only": {"true_hypothesis": truth, "visual_depth_offset_m": world.visual_depth_offset_m},
        "metrics": metrics,
        "trace": trace,
        "measurements": captures,
    }
