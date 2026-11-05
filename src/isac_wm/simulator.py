"""Documented TOY sensor scene. Not raw-I/Q, calibrated ISAC or quantum physics.

An orthographic synthetic RGB camera and a range/bearing/radial-velocity sensor
observe one target. Sensing consumes an explicit fraction of a shared frame.
Truth is exposed only to the training/evaluation harness, never the policy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .events import EventQueue, Observation, SnapshotBuffer
from .validation import finite, integer


@dataclass(frozen=True)
class SensorAction:
    name: str
    airtime_fraction: float
    delay_s: float
    range_sigma_m: float
    angle_sigma_rad: float
    velocity_sigma_mps: float


ACTIONS = (
    SensorAction("skip", 0.0, 0.0, 0.0, 0.0, 0.0),
    SensorAction("quick", 0.10, 0.06, 0.28, 0.045, 0.22),
    SensorAction("precise", 0.30, 0.14, 0.08, 0.014, 0.08),
)


@dataclass(frozen=True)
class SceneConfig:
    steps: int = 48
    dt: float = 0.1
    image_size: int = 24
    room_m: float = 10.0
    camera_period: int = 2
    camera_delay_s: float = 0.13
    delay_jitter_s: float = 0.04
    dropout: float = 0.05
    acceleration_std: float = 0.5
    snr_linear: float = 10.0
    bandwidth_hz: float = 1_000_000.0
    camera_dropout: float | None = None
    radio_dropout: float | None = None
    radio_outlier_probability: float = 0.0
    radio_outlier_scale_m: float = 4.0
    occlusion_fraction: float = 1 / 6
    radio_delay_extra_s: float = 0.0
    radio_range_bias_m: float = 0.0
    radio_angle_bias_rad: float = 0.0

    def __post_init__(self) -> None:
        integer(self.steps, "steps", 8, 100_000)
        integer(self.image_size, "image_size", 16, 1024)
        integer(self.camera_period, "camera_period", 1, 100_000)
        for name in ("camera_dropout", "radio_dropout", "radio_outlier_probability", "occlusion_fraction"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be in [0,1]")
        finite(self.radio_outlier_scale_m, "radio_outlier_scale_m")
        finite(self.radio_delay_extra_s, "radio_delay_extra_s")
        finite(self.radio_range_bias_m, "radio_range_bias_m")
        finite(self.radio_angle_bias_rad, "radio_angle_bias_rad")
        if self.steps < 8 or self.image_size < 16 or self.camera_period < 1:
            raise ValueError("Invalid sequence, image size or sensor period")
        for name in ("dt", "room_m", "bandwidth_hz"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("camera_delay_s", "delay_jitter_s", "acceleration_std", "snr_linear"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    @property
    def state_scale(self) -> np.ndarray:
        return np.array([self.room_m, self.room_m, 2.0, 2.0], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


def communication_bits(config: SceneConfig, action: int) -> float:
    """AWGN time-sharing proxy: not throughput, BLER or an ISAC capacity theorem."""
    if action not in (0, 1, 2):
        raise ValueError("Unknown sensing action")
    return (
        (1 - ACTIONS[action].airtime_fraction)
        * config.dt
        * config.bandwidth_hz
        * math.log2(1 + config.snr_linear)
    )


def ideal_radio(state: np.ndarray, room_m: float) -> np.ndarray:
    x, y, vx, vy = map(float, state)
    r = max(math.hypot(x, y), 1e-6)
    return np.array([r / (room_m * math.sqrt(2)), x / r, y / r, (x * vx + y * vy) / (2 * r)], np.float32)


class ToyScene:
    def __init__(self, seed: int, config: SceneConfig):
        integer(seed, "seed", 0, 2**32 - 1)
        self.seed, self.config = seed, config
        rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
        self._truth = np.zeros((config.steps, 4), np.float32)
        self._truth[0, :2] = rng.uniform(0.15, 0.85, 2) * config.room_m
        self._truth[0, 2:] = rng.uniform(-1.2, 1.2, 2)
        turn = rng.normal(0, config.acceleration_std, 2)
        for k in range(1, config.steps):
            if k % 8 == 0:
                turn = rng.normal(0, config.acceleration_std, 2)
            prev = self._truth[k - 1]
            accel = turn + rng.normal(0, 0.08, 2)
            v = np.clip(prev[2:] + accel * config.dt, -1.7, 1.7)
            p = prev[:2] + v * config.dt
            # Reflection boundaries; exogenous to the sensing policy.
            for j in range(2):
                lo, hi = 0.08 * config.room_m, 0.92 * config.room_m
                if p[j] < lo:
                    p[j], v[j] = 2 * lo - p[j], abs(v[j])
                elif p[j] > hi:
                    p[j], v[j] = 2 * hi - p[j], -abs(v[j])
            self._truth[k] = np.r_[p, v]
        self.occlusion_start = int(rng.integers(config.steps // 4, config.steps // 2))
        self.occlusion_length = (
            0 if config.occlusion_fraction == 0 else max(3, int(config.steps * config.occlusion_fraction))
        )

    def truth_for_evaluation(self) -> np.ndarray:
        return self._truth.copy()

    def _rng(self, step: int, modality: int, action: int = 0) -> np.random.Generator:
        # Keyed streams: policy choices cannot change the underlying trajectory,
        # camera, or noise for the same (step, action) across paired evaluations.
        return np.random.default_rng(np.random.SeedSequence([self.seed, step, modality, action]))

    def camera(self, step: int) -> Observation | None:
        c = self.config
        integer(step, "step", 0, c.steps - 1)
        if step % c.camera_period:
            return None
        rng = self._rng(step, 21)
        if rng.random() < (c.dropout if c.camera_dropout is None else c.camera_dropout):
            return None
        n = c.image_size
        img = rng.uniform(0.01, 0.055, (3, n, n)).astype(np.float32)
        img[:, :1, :] = 0.16
        img[:, -1:, :] = 0.16
        img[:, :, :1] = 0.16
        img[:, :, -1:] = 0.16
        # The synthetic object is a red Gaussian in a calibrated top-down view.
        # This is a toy renderer, not a realistic or multi-view camera simulator.
        hidden = self.occlusion_start <= step < self.occlusion_start + self.occlusion_length
        if not hidden:
            x, y = self._truth[step, :2] / c.room_m * (n - 1)
            yy, xx = np.mgrid[0:n, 0:n]
            blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.85**2))
            img[0] += 0.9 * blob
            img[1] += 0.12 * blob
        capture = step * c.dt
        delay = c.camera_delay_s + rng.uniform(0, c.delay_jitter_s)
        return Observation(step * 3, "vision", capture, capture + delay, img)

    def radio(self, step: int, action: int) -> Observation | None:
        integer(step, "step", 0, self.config.steps - 1)
        integer(action, "action", 0, 2)
        if action not in (0, 1, 2):
            raise ValueError("Unknown sensing action")
        if action == 0:
            return None
        c, spec = self.config, ACTIONS[action]
        rng = self._rng(step, 22, action)
        if rng.random() < (c.dropout if c.radio_dropout is None else c.radio_dropout):
            return None
        s = self._truth[step]
        r = max(np.linalg.norm(s[:2]), 1e-6)
        angle = math.atan2(float(s[1]), float(s[0]))
        noisy_r = max(0.0, r + c.radio_range_bias_m + rng.normal(0, spec.range_sigma_m))
        noisy_a = angle + c.radio_angle_bias_rad + rng.normal(0, spec.angle_sigma_rad)
        vr = float(np.dot(s[:2], s[2:]) / r) + rng.normal(0, spec.velocity_sigma_mps)
        # Fault stream is separate: toggling outliers does not change scene/camera/noise.
        fault_rng = self._rng(step, 99, action)
        if fault_rng.random() < c.radio_outlier_probability:
            noisy_r = max(0.0, noisy_r + fault_rng.choice([-1.0, 1.0]) * c.radio_outlier_scale_m)
            noisy_a += float(fault_rng.normal(0, 0.2))
        features = np.array(
            [noisy_r / (c.room_m * math.sqrt(2)), math.cos(noisy_a), math.sin(noisy_a), vr / 2], np.float32
        )
        capture = step * c.dt
        delay = spec.delay_s + c.radio_delay_extra_s + rng.uniform(0, c.delay_jitter_s)
        return Observation(step * 3 + action, "radio", capture, capture + delay, features, action)


def recorded_episode(seed: int, config: SceneConfig) -> dict[str, np.ndarray]:
    """Generate a randomized-policy trace with separately named training labels.

    A visual label belongs to the actual image event accepted by the snapshot
    buffer. It describes that event's capture time, never the delivery-time
    target. Cached frames and occluded captures have no localization target.
    None of the ``vision_*`` label fields is an online model input.
    """
    scene, queue = ToyScene(seed, config), EventQueue()
    buffer = SnapshotBuffer(config.image_size)
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 33]))
    snapshots = []
    actions, radio_targets, radio_target_valid = [], [], []
    truth = scene.truth_for_evaluation()
    # Acquisition metadata is keyed by event identity, not reconstructed from a
    # rounded delivery time. This also handles reordering and multiple arrivals.
    camera_labels: dict[int, tuple[float, np.ndarray, bool]] = {}
    visual_xy, visual_mask, visual_capture_s, visual_uid = [], [], [], []
    for k in range(config.steps):
        t = k * config.dt
        # Decisions use only previously delivered evidence. New samples are
        # acquired after the decision and cannot enter that decision's snapshot.
        snap = buffer.update(queue.release(t), t, config.dt)
        snapshots.append(snap)
        target_xy, mask, captured, uid = np.zeros(2, np.float32), 0.0, -1.0, -1
        if snap["new"][0] and snap["valid"][0]:
            delivered = buffer.latest["vision"]
            captured, capture_xy, visible = camera_labels[delivered.uid]
            if captured != delivered.capture_s:
                raise RuntimeError("Visual capture metadata does not match the delivered event")
            uid = delivered.uid
            mask = float(visible)
            if visible:
                target_xy = capture_xy.copy()
        visual_xy.append(target_xy)
        visual_mask.append(mask)
        visual_capture_s.append(captured)
        visual_uid.append(uid)
        a = int(policy_rng.choice([0, 1, 2], p=[0.30, 0.45, 0.25]))
        actions.append(a)
        camera, radio = scene.camera(k), scene.radio(k, a)
        if camera is not None:
            visible = not (scene.occlusion_start <= k < scene.occlusion_start + scene.occlusion_length)
            camera_labels[camera.uid] = (camera.capture_s, truth[k, :2] / config.room_m, visible)
        for obs in (camera, radio):
            if obs is not None:
                queue.push(obs)
        # Training labels may use observations not yet available; NEVER inputs.
        radio_targets.append(np.zeros(4, np.float32) if radio is None else radio.payload)
        radio_target_valid.append(float(radio is not None))
    out = {key: np.stack([s[key] for s in snapshots]) for key in snapshots[0]}
    out["state_target"] = truth / config.state_scale
    out["action_target"] = np.array(actions, np.int64)
    out["radio_target"] = np.stack(radio_targets)
    out["radio_target_valid"] = np.array(radio_target_valid, np.float32)
    out["vision_capture_target_xy"] = np.stack(visual_xy).astype(np.float32)
    out["vision_target_mask"] = np.array(visual_mask, np.float32)
    out["vision_capture_s"] = np.array(visual_capture_s, np.float64)
    out["vision_event_uid"] = np.array(visual_uid, np.int64)
    return out
