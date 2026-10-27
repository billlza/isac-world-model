"""Explicit fault-injection configurations; nothing is silently relabelled nominal."""

from __future__ import annotations

from dataclasses import replace

from .simulator import SceneConfig

SCENARIOS = (
    "nominal",
    "delay",
    "camera_loss",
    "blackout",
    "radio_outliers",
    "radio_bias",
    "sparse_camera",
    "maneuver",
)


def scenario_config(name: str, base: SceneConfig) -> SceneConfig:
    if name == "nominal":
        return base
    if name == "delay":
        return replace(base, camera_delay_s=0.65, delay_jitter_s=0.35, radio_delay_extra_s=0.25)
    if name == "camera_loss":
        return replace(base, camera_dropout=1.0)
    if name == "blackout":
        return replace(base, camera_dropout=1.0, radio_dropout=1.0)
    if name == "radio_outliers":
        return replace(base, radio_outlier_probability=0.25, radio_outlier_scale_m=4.0)
    if name == "radio_bias":
        return replace(base, radio_range_bias_m=0.8, radio_angle_bias_rad=0.06)
    if name == "sparse_camera":
        return replace(base, camera_period=5, camera_delay_s=0.35, delay_jitter_s=0.08)
    if name == "maneuver":
        return replace(base, acceleration_std=2.0, occlusion_fraction=0.4)
    raise ValueError(f"Unknown scenario {name!r}; choose from {SCENARIOS}")
