"""Deterministic pretraining traces for the existing single-target toy scene.

The catalogue varies sensing conditions and target motion without changing the
coordinate system or tensor sizes. Optional RGB appearance augmentation adds
background texture and non-target visual clutter; this is still synthetic
single-target tracking, not multi-target perception or real-image reconstruction.

Reference quality belongs to the supervision contract. The simulator exposes
exact state truth, so its reference variance is zero even when its *sensors* are
noisy, delayed, occluded, or biased. External datasets must supply independently
assessed reference variance/validity instead of copying sensor noise or a model's
own confidence into these fields.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .simulator import SceneConfig, recorded_episode


def scene_profiles(base: SceneConfig) -> dict[str, SceneConfig]:
    """Return a fixed-order training catalogue with common geometry and timing.

    ``steps``, ``dt``, ``image_size``, and ``room_m`` are identical to ``base``
    for every profile. Complete sensor blackout is not a training success case:
    it belongs in an observability/failure evaluation, with uncertainty and
    failure reported rather than a claim of recovered localization.
    """
    camera_dropout = base.dropout if base.camera_dropout is None else base.camera_dropout
    radio_dropout = base.dropout if base.radio_dropout is None else base.radio_dropout
    if camera_dropout == 1 and radio_dropout == 1:
        raise ValueError("Complete sensor blackout is an observability evaluation, not a training profile")

    motion = {"acceleration_std": max(base.acceleration_std, 2.0)}
    delay = {
        "camera_delay_s": max(base.camera_delay_s, 0.65),
        "delay_jitter_s": max(base.delay_jitter_s, 0.35),
        "radio_delay_extra_s": max(base.radio_delay_extra_s, 0.25),
    }
    occlusion = {"occlusion_fraction": max(base.occlusion_fraction, 0.40)}
    outliers = {
        "radio_outlier_probability": max(base.radio_outlier_probability, 0.25),
        "radio_outlier_scale_m": max(base.radio_outlier_scale_m, 4.0),
    }
    bias = {
        "radio_range_bias_m": (base.radio_range_bias_m if abs(base.radio_range_bias_m) >= 0.8 else 0.8),
        "radio_angle_bias_rad": (
            base.radio_angle_bias_rad if abs(base.radio_angle_bias_rad) >= 0.06 else 0.06
        ),
    }
    sparse = {
        "camera_period": max(base.camera_period, 5),
        "camera_delay_s": max(base.camera_delay_s, 0.35),
        "delay_jitter_s": max(base.delay_jitter_s, 0.08),
    }
    return {
        "nominal": base,
        "maneuver": replace(base, **motion),
        "delay": replace(base, **delay),
        "occlusion": replace(base, **occlusion),
        "radio_outliers": replace(base, **outliers),
        "radio_bias": replace(base, **bias),
        "sparse_camera": replace(base, **sparse),
        "maneuver_delay": replace(base, **motion, **delay),
        "occlusion_outliers": replace(base, **occlusion, **outliers),
        "maneuver_bias": replace(base, **motion, **bias),
        "combined": replace(base, **motion, **delay, **occlusion, **outliers, **bias),
    }


def _augment_captured_image(image: np.ndarray, seed: int, event_uid: int) -> np.ndarray:
    """Transform one acquired image using only that image and exogenous RNG.

    Episode-level colour/background/clutter parameters stay fixed. Exposure and
    small pixel perturbations are keyed by capture identity, never delivery tick,
    sequence length, state labels, or future observations. No target is drawn or
    moved by this function; an occluded target remains absent.
    """
    episode_rng = np.random.default_rng(np.random.SeedSequence([seed, 70]))
    event_rng = np.random.default_rng(np.random.SeedSequence([seed, 71, event_uid]))
    _, height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    channel_order = episode_rng.permutation(3)
    channel_gain = episode_rng.uniform(0.75, 1.15, (3, 1, 1)).astype(np.float32)
    transformed = image[channel_order].astype(np.float32, copy=True) * channel_gain

    phase = episode_rng.uniform(0, 2 * np.pi, 2)
    background = (
        0.018 * np.sin(xx / max(width - 1, 1) * 2 * np.pi + phase[0])
        + 0.018 * np.cos(yy / max(height - 1, 1) * 2 * np.pi + phase[1])
        + 0.04
    ).astype(np.float32)
    transformed += background[None]

    # These static, lower-contrast patches have no state, identity or dynamics
    # labels. They are nuisance clutter, not additional tracked targets.
    for _ in range(int(episode_rng.integers(3, 7))):
        x, y = episode_rng.uniform(0.08, 0.92, 2) * np.array([width - 1, height - 1])
        sigma = float(episode_rng.uniform(0.8, 2.2))
        amplitude = float(episode_rng.uniform(0.06, 0.24))
        patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        # Nearly achromatic clutter cannot create the target's coloured blob.
        colour = episode_rng.uniform(0.80, 1.0, (3, 1, 1))
        transformed += (amplitude * colour * patch[None]).astype(np.float32)

    transformed *= np.float32(event_rng.uniform(0.85, 1.15))
    transformed += event_rng.normal(0, 0.005, transformed.shape).astype(np.float32)
    return np.clip(transformed, 0.0, 1.0).astype(np.float32)


def make_episode(seed: int, scene: SceneConfig, *, appearance: bool = False) -> dict[str, np.ndarray]:
    """Record one trace and attach external-reference quality supervision.

    All fields are NumPy arrays and work with the existing ``batchify`` helper.
    The only new fields are ``state_reference_variance`` and
    ``state_reference_validity``, both float32 ``[time, 4]`` arrays in normalized
    state coordinates. Variances therefore use squared normalized units. A zero
    validity entry means that coordinate cannot supervise any state/future or
    displacement target involving that reference.

    Augmentation preserves the six existing online input keys and all original
    labels. A cached image is the *same augmented capture* at every stale tick;
    empty camera placeholders remain zero. Reference/provenance fields are never
    added to the model's online input contract.
    """
    if not isinstance(appearance, bool):
        raise ValueError("appearance must be a boolean")
    episode = recorded_episode(seed, scene)
    episode["state_reference_variance"] = np.zeros_like(episode["state_target"], dtype=np.float32)
    episode["state_reference_validity"] = np.ones_like(episode["state_target"], dtype=np.float32)
    if not appearance:
        return episode

    images = np.zeros_like(episode["image"])
    captures: dict[int, np.ndarray] = {}
    latest_uid: int | None = None
    for tick in range(scene.steps):
        if not episode["valid"][tick, 0]:
            continue
        if episode["new"][tick, 0]:
            uid = int(episode["vision_event_uid"][tick])
            if uid < 0:
                raise ValueError("Fresh visual evidence requires its capture event UID")
            if uid not in captures:
                captures[uid] = _augment_captured_image(episode["image"][tick], seed, uid)
            latest_uid = uid
        if latest_uid is None:
            raise ValueError("A stale image must have a previously delivered capture")
        images[tick] = captures[latest_uid]
    episode["image"] = images
    return episode
