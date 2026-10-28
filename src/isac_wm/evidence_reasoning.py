"""Auditable, causal multi-hypothesis evidence memory for synthetic RF/vision ISAC.

This is an interpretable, *model-based* reasoning baseline. The likelihoods and
measurement-error floors are assumptions, NOT calibrated posteriors. It is NOT
an object/material classifier, learned world-model, or autonomous safety system.

The API accepts only observation evidence: the RF adapter deliberately ignores
simulator-only target truth, and the camera returns bearing, not invented depth.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

MAX_ANGLE_RAD = math.radians(60.0)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _identity(uid: int, frame: int, capture_s: float, available_s: float) -> None:
    if type(uid) is not int or uid < 0 or type(frame) is not int or frame < 0:
        raise ValueError("Evidence IDs and frame indexes must be nonnegative integers")
    capture_s = _finite(capture_s, "capture_s")
    available_s = _finite(available_s, "available_s")
    if capture_s < 0 or available_s < capture_s:
        raise ValueError("Evidence cannot be available before nonnegative acquisition")


def _bearing(angle: float, sigma: float) -> None:
    if abs(_finite(angle, "bearing_rad")) > MAX_ANGLE_RAD + 1e-8:
        raise ValueError("Outside the modeled forward-looking 120-degree field of view")
    if not 0 < _finite(sigma, "bearing_sigma_rad") <= math.pi / 3:
        raise ValueError("Bearing uncertainty must be positive and no more than 60 degrees")


@dataclass(frozen=True)
class RFPoint:
    uid: int
    frame: int
    capture_s: float
    available_s: float
    range_m: float
    bearing_rad: float
    radial_speed_mps: float
    bearing_sigma_rad: float = math.radians(8.0)
    range_sigma_m: float = 3.0
    speed_sigma_mps: float = 0.5
    provenance: str = "synthetic-mimo-ofdm-matched-peak"

    def __post_init__(self) -> None:
        _identity(self.uid, self.frame, self.capture_s, self.available_s)
        _bearing(self.bearing_rad, self.bearing_sigma_rad)
        if not 0 < _finite(self.range_m, "range_m") <= 100_000:
            raise ValueError("Range must be finite and positive")
        _finite(self.radial_speed_mps, "radial_speed_mps")
        for name in ("range_sigma_m", "speed_sigma_mps"):
            if not 0 < _finite(getattr(self, name), name) < 100_000:
                raise ValueError(f"Invalid uncertainty {name}")
        if self.provenance != "synthetic-mimo-ofdm-matched-peak":
            raise ValueError("This first-stage adapter requires declared synthetic RF provenance")


@dataclass(frozen=True)
class VisionBearing:
    uid: int
    frame: int
    capture_s: float
    available_s: float
    bearing_rad: float
    bearing_sigma_rad: float = math.radians(5.0)
    provenance: str = "synthetic-perspective-rgb-blob"

    def __post_init__(self) -> None:
        _identity(self.uid, self.frame, self.capture_s, self.available_s)
        _bearing(self.bearing_rad, self.bearing_sigma_rad)
        if self.provenance != "synthetic-perspective-rgb-blob":
            raise ValueError("Unknown camera calibration/provenance")


Evidence = RFPoint | VisionBearing


def from_phy_result(
    result: dict,
    *,
    uid: int,
    frame: int,
    capture_s: float,
    available_s: float,
    bearing_sigma_deg: float = 8.0,
    range_sigma_m: float = 3.0,
    speed_sigma_mps: float = 0.5,
) -> RFPoint:
    """Convert *only* PHY sensing output; simulator truth never enters belief.

    The sigma arguments are externally assumed floors, NOT estimates calibrated
    by the raw matched-filter peak. Caller must calibrate on held-out evidence.
    """
    if result.get("schema_version") != 1:
        raise ValueError("Unknown PHY report schema")
    sensing = result["sensing"]
    return RFPoint(
        uid=uid,
        frame=frame,
        capture_s=capture_s,
        available_s=available_s,
        range_m=float(sensing["range_m"]),
        bearing_rad=math.radians(float(sensing["bearing_deg"])),
        radial_speed_mps=float(sensing["radial_speed_mps"]),
        bearing_sigma_rad=math.radians(bearing_sigma_deg),
        range_sigma_m=range_sigma_m,
        speed_sigma_mps=speed_sigma_mps,
    )


def _x_pixel_from_bearing(bearing_rad: float, width: int, fov_deg: float) -> float:
    tan_half = math.tan(math.radians(fov_deg) / 2)
    return ((math.tan(bearing_rad) / tan_half) + 1) * (width - 1) / 2


def render_synthetic_camera(
    bearing_deg: float,
    *,
    size: int = 64,
    fov_deg: float = 120.0,
    seed: int = 0,
    occluded: bool = False,
    distractor_bearing_deg: float | None = None,
) -> np.ndarray:
    """SIMULATOR ONLY: render an unlabelled bearing cue; no depth or materials.

    The target bearing is simulation truth and never passed to the extractor.
    The image is not interchangeable with the existing top-down ToyScene images.
    """
    if type(size) is not int or not 32 <= size <= 512 or not 10 <= fov_deg <= 120:
        raise ValueError("Invalid camera image size/FOV")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid image noise seed")
    b = math.radians(_finite(bearing_deg, "bearing_deg"))
    if abs(b) > math.radians(fov_deg / 2) - 1e-8:
        raise ValueError("Target outside synthetic camera field of view")
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.0, 0.02, (3, size, size)).astype(np.float32)
    if not occluded:
        yy, xx = np.mgrid[:size, :size]
        locations = [b]
        if distractor_bearing_deg is not None:
            d = math.radians(_finite(distractor_bearing_deg, "distractor_bearing_deg"))
            if abs(d) > math.radians(fov_deg / 2):
                raise ValueError("Distractor outside field of view")
            locations.append(d)
        for angle in locations:
            x_center = _x_pixel_from_bearing(angle, size, fov_deg)
            blob = np.exp(-((xx - x_center) ** 2 + (yy - (size - 1) / 2) ** 2) / (2 * 1.3**2))
            image[0] += 0.9 * blob.astype(np.float32)
            image[1] += 0.08 * blob.astype(np.float32)
    return np.clip(image, 0, 1)


def detect_visual_bearing(
    image: np.ndarray,
    *,
    uid: int,
    frame: int,
    capture_s: float,
    available_s: float,
    fov_deg: float = 120.0,
    assumed_sigma_deg: float = 5.0,
) -> VisionBearing | None:
    """Extract a single colored synthetic bearing cue, or *abstain*.

    No category, identity, size, distance or material is inferred. Multiple
    isolated blobs count as ambiguous, not as a forced nearest-neighbour match.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[0] != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("Expected (3,N,N) image")
    if not 32 <= array.shape[1] <= 512 or not np.isfinite(array).all():
        raise ValueError("Invalid/non-finite image")
    if not 10 <= _finite(fov_deg, "fov_deg") <= 120:
        raise ValueError("Invalid FOV")
    _identity(uid, frame, capture_s, available_s)
    # This red-excess threshold is a test for the synthetic renderer ONLY.
    mask = array[0] - array[1] > 0.28
    if int(mask.sum()) < 3:
        return None
    # Connected components stop a distractor from being mistaken for certainty.
    unvisited = set(map(tuple, np.argwhere(mask)))
    components: list[list[tuple[int, int]]] = []
    while unvisited:
        start = unvisited.pop()
        stack, points = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    point = (y + dy, x + dx)
                    if point in unvisited:
                        unvisited.remove(point)
                        points.append(point)
                        stack.append(point)
        if len(points) >= 3:
            components.append(points)
    if len(components) != 1:
        return None
    points = np.asarray(components[0])
    weights = (array[0, points[:, 0], points[:, 1]] - array[1, points[:, 0], points[:, 1]]).clip(0)
    if float(weights.sum()) < 0.5:
        return None
    x = float(np.average(points[:, 1], weights=weights))
    tan_half = math.tan(math.radians(fov_deg) / 2)
    bearing_rad = math.atan((2 * x / (array.shape[2] - 1) - 1) * tan_half)
    return VisionBearing(
        uid=uid,
        frame=frame,
        capture_s=capture_s,
        available_s=available_s,
        bearing_rad=bearing_rad,
        bearing_sigma_rad=math.radians(assumed_sigma_deg),
    )


@dataclass(frozen=True)
class PairResult:
    frame: int
    rf_uid: int
    vision_uid: int
    residual_deg: float
    evidence_log_ratio: float
    outcome: str

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "rf_uid": self.rf_uid,
            "vision_uid": self.vision_uid,
            "bearing_residual_deg": self.residual_deg,
            "log_support_ratio": self.evidence_log_ratio,
            "outcome": self.outcome,
        }


class CausalEvidenceMemory:
    """Bounded, transparent *hypothesis comparison*, not a semantic oracle.

    H_same: RF and camera cues correspond to the same object.
    H_other: RF and camera cues come from unrelated objects.
    The score is a finite-window likelihood ratio under stated toy assumptions,
    **not a calibrated probability** of object identity. Never hallucinate labels.
    """

    def __init__(
        self,
        *,
        max_frames: int = 12,
        max_ids: int = 10_000,
        max_lag_s: float = 0.15,
        max_evidence_age_s: float = 1.0,
        min_corrob_pairs: int = 2,
        fov_deg: float = 120.0,
    ) -> None:
        if type(max_frames) is not int or not 2 <= max_frames <= 4096:
            raise ValueError("Memory frame count must be between 2 and 4096")
        if type(max_ids) is not int or max_ids < max_frames * 2 or max_ids > 1_000_000:
            raise ValueError("Invalid bounded evidence-ID budget")
        if type(min_corrob_pairs) is not int or not 2 <= min_corrob_pairs <= max_frames:
            raise ValueError("Invalid corroboration requirement")
        if not 0 < _finite(max_lag_s, "max_lag_s") <= 10:
            raise ValueError("Invalid cross-modal time tolerance")
        if not 0 < _finite(max_evidence_age_s, "max_evidence_age_s") <= 3600:
            raise ValueError("Invalid maximum evidence age")
        if not 10 <= _finite(fov_deg, "fov_deg") <= 120:
            raise ValueError("Invalid field of view")
        self.max_frames = max_frames
        self.max_ids = max_ids
        self.max_lag_s = max_lag_s
        self.max_evidence_age_s = max_evidence_age_s
        self.min_corrob_pairs = min_corrob_pairs
        self.fov_half_rad = math.radians(fov_deg) / 2
        self.now_s = 0.0
        self.seen: set[int] = set()
        self.frames: OrderedDict[int, dict[str, Evidence]] = OrderedDict()
        self.pairs: OrderedDict[int, PairResult] = OrderedDict()
        self.ignored_stale = 0
        self.incomparable = 0
        self.last_rf: RFPoint | None = None
        self.last_pair: PairResult | None = None

    def ingest(self, evidence: Evidence, *, now_s: float) -> None:
        if not isinstance(evidence, (RFPoint, VisionBearing)):
            raise TypeError("Evidence must be a validated RFPoint or VisionBearing")
        now = _finite(now_s, "now_s")
        if now + 1e-10 < self.now_s or evidence.available_s > now + 1e-10:
            raise ValueError("Cannot process future evidence or rewind reasoning clock")
        if evidence.uid in self.seen:
            raise ValueError("Duplicate evidence UID: refusing double-counted support")
        if len(self.seen) >= self.max_ids:
            raise OverflowError("Evidence identity budget exhausted; start a new episode")
        modality = "rf" if isinstance(evidence, RFPoint) else "vision"
        old_slot = self.frames.get(evidence.frame)
        if old_slot is not None and modality in old_slot:
            raise ValueError("More than one same-modality detection per frame; association unresolved")
        self.now_s = max(self.now_s, now)
        self.seen.add(evidence.uid)
        if self.frames and evidence.frame < next(iter(self.frames)):
            self.ignored_stale += 1
            return
        slot = self.frames.setdefault(evidence.frame, {})
        slot[modality] = evidence
        # Delayed packets may arrive out of capture order; expire oldest frame,
        # never the newest frame merely because its counterpart arrived late.
        self.frames = OrderedDict(sorted(self.frames.items()))
        if isinstance(evidence, RFPoint):
            if self.last_rf is None or (evidence.capture_s, evidence.uid) > (
                self.last_rf.capture_s,
                self.last_rf.uid,
            ):
                self.last_rf = evidence
        if "rf" in slot and "vision" in slot and evidence.frame not in self.pairs:
            rf, vision = slot["rf"], slot["vision"]
            assert isinstance(rf, RFPoint) and isinstance(vision, VisionBearing)
            lag = abs(rf.capture_s - vision.capture_s)
            if lag <= self.max_lag_s:
                residual = vision.bearing_rad - rf.bearing_rad
                # A conservative angular-motion allowance for asynchronous capture.
                sigma = math.sqrt(
                    rf.bearing_sigma_rad**2 + vision.bearing_sigma_rad**2 + (math.radians(12) * lag) ** 2
                )
                log_shared = -0.5 * (residual / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))
                log_unrelated = -math.log(2 * self.fov_half_rad)
                # Relative evidence is clipped so a single erroneous packet cannot dominate indefinitely.
                log_ratio = max(-6.0, min(6.0, log_shared - log_unrelated))
                result = PairResult(
                    frame=rf.frame,
                    rf_uid=rf.uid,
                    vision_uid=vision.uid,
                    residual_deg=math.degrees(residual),
                    evidence_log_ratio=log_ratio,
                    outcome="compatible" if abs(residual) <= 2.5 * sigma else "contradictory",
                )
                self.pairs[rf.frame] = result
                self.last_pair = result
            else:
                self.incomparable += 1
        while len(self.frames) > self.max_frames:
            old, _ = self.frames.popitem(last=False)
            self.pairs.pop(old, None)
        # Only unexpired comparisons contribute to the belief; no infinite evidence accumulation.
        for index in list(self.pairs):
            if index not in self.frames:
                del self.pairs[index]

    def advance(self, now_s: float) -> None:
        now = _finite(now_s, "now_s")
        if now < self.now_s:
            raise ValueError("Reasoning clock cannot run backwards")
        self.now_s = now

    def counterfactual_parallax_test(
        self,
        *,
        alternative_depth_offset_m: float = 7.5,
        max_lateral_baseline_m: float = 4.0,
        assumed_camera_sigma_deg: float = 5.0,
    ) -> dict | None:
        """Ask which *new camera viewpoint* could distinguish collinear worlds.

        This is a geometry-only counterfactual: H_same places the visual object
        at the measured radar range; H_other places an object on the same ray
        at a user-declared *hypothetical* additional depth. It does NOT infer
        that alternative's true depth or control a real camera/robot.
        """
        if self.last_rf is None:
            return None
        offset = _finite(alternative_depth_offset_m, "alternative_depth_offset_m")
        baseline = _finite(max_lateral_baseline_m, "max_lateral_baseline_m")
        sigma = _finite(assumed_camera_sigma_deg, "assumed_camera_sigma_deg")
        if not 0 < offset <= 1000 or not 0 <= baseline <= 1000 or not 0 < sigma <= 60:
            raise ValueError("Invalid hypothetical viewpoint comparison parameters")
        rf = self.last_rf
        direction = np.array([math.cos(rf.bearing_rad), math.sin(rf.bearing_rad)])
        perp = np.array([-direction[1], direction[0]])
        same = rf.range_m * direction
        other = (rf.range_m + offset) * direction
        proposals = []
        for fraction in (0.0, 0.25, 0.5, 1.0):
            displacement = baseline * fraction
            camera = perp * displacement
            same_bearing = math.atan2(*(same - camera)[::-1])
            other_bearing = math.atan2(*(other - camera)[::-1])
            if abs(same_bearing) > self.fov_half_rad or abs(other_bearing) > self.fov_half_rad:
                continue
            separation = abs(math.degrees(same_bearing - other_bearing))
            proposals.append((separation, displacement, same_bearing, other_bearing))
        if not proposals:
            return {
                "recommendation": "no_available_viewpoint_in_forward_fov",
                "hypothetical_depth_offset_m": offset,
                "identity_confirmed": False,
            }
        separation, displacement, sa, sb = max(proposals)
        discriminability = separation / sigma
        return {
            "hypothesis_same": "camera object at the radar measured range",
            "hypothesis_other": "distinct camera object at a hypothetical farther range",
            "hypothetical_depth_offset_m": offset,
            "candidate_lateral_shift_m": displacement,
            "predicted_bearing_same_deg": math.degrees(sa),
            "predicted_bearing_other_deg": math.degrees(sb),
            "separation_deg": separation,
            "assumed_new_camera_sigma_deg": sigma,
            "separation_in_assumed_sigma": discriminability,
            "recommendation": (
                "hypothetical_viewpoint_could_help_disambiguate"
                if discriminability >= 2.0
                else "parallax_too_weak_use_more_depth_or_viewpoint_evidence"
            ),
            "identity_confirmed": False,
            "note": "Counterfactual geometry under assumed alternative depth; not measured or calibrated.",
        }

    def explanation(self) -> dict:
        values = list(self.pairs.values())
        total_log_ratio = float(sum(item.evidence_log_ratio for item in values))
        same_score = 1 / (1 + math.exp(-max(-30, min(30, total_log_ratio))))
        compatible = sum(p.outcome == "compatible" for p in values)
        contradictory = sum(p.outcome == "contradictory" for p in values)
        status = "insufficient_evidence"
        newest_pair_capture = None
        if values:
            newest_pair_capture = max(max(o.capture_s for o in self.frames[p.frame].values()) for p in values)
        stale = newest_pair_capture is not None and self.now_s - newest_pair_capture > self.max_evidence_age_s
        if stale:
            status = "stale_evidence"
        elif len(values) >= self.min_corrob_pairs:
            if same_score > 0.95 and contradictory == 0:
                status = "angularly_consistent_identity_unresolved"
            elif same_score < 0.05:
                status = "cross_modal_conflict"
            else:
                status = "ambiguous"
        rf = self.last_rf
        prediction = None
        if rf is not None:
            age = max(0.0, self.now_s - rf.capture_s)
            # Only radial position can be extrapolated from one radial Doppler value.
            prediction = {
                "range_m": max(0.0, rf.range_m + age * rf.radial_speed_mps),
                "assumed_sigma_range_m": math.sqrt(
                    rf.range_sigma_m**2 + (age * rf.speed_sigma_mps) ** 2 + (0.5 * age**2) ** 2
                ),
                "last_measured_bearing_deg": math.degrees(rf.bearing_rad),
                "bearing_at_current_time": "unknown_without_angular_motion_model",
                "seconds_since_capture": age,
                "source_rf_uid": rf.uid,
            }
        return {
            "time_s": self.now_s,
            "association_status": status,
            "identity_confirmed": False,
            "independent_visual_depth_available": False,
            "counterfactual_parallax_test": self.counterfactual_parallax_test(),
            "alternative_explanations": [
                "same_physical_object",
                "distinct_objects_aligned_in_bearing",
                "reflection_or_false_radar_peak",
            ],
            "relative_same_object_score_uncalibrated": same_score,
            "pair_count": len(values),
            "latest_corrob_capture_s": newest_pair_capture,
            "evidence_expired": stale,
            "supporting_pairs": compatible,
            "contradicting_pairs": contradictory,
            "comparisons": [p.to_dict() for p in values],
            "radial_prediction": prediction,
            "object_category": "unknown",
            "shape": "unknown",
            "material": "unknown",
            "observation_request": (
                "seek_independent_depth_or_viewpoint_for_identity"
                if status == "angularly_consistent_identity_unresolved"
                else "reobserve_with_independent_view_or_frequency"
            ),
            "ignored_stale_events": self.ignored_stale,
            "incomparable_timestamps": self.incomparable,
            "assumptions": [
                "Synthetic perspective camera, no genuine semantic vision model",
                "Synthetic monostatic radar matched-peak, not a calibrated measurement covariance",
                "Two candidate association hypotheses only; Gaussian angular errors vs uniform unrelated bearing",
                "Sliding window and freshness check limit evidence recycling; pairwise independence is a modeling assumption",
                "Angular agreement never proves physical identity without independent depth/scene evidence",
                "Relative plausibility is NOT calibrated confidence or a material/object classifier",
            ],
        }
