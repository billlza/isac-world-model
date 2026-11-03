"""Matched blackout/reacquisition challenge; separate simulator, observations, and scorer."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from isac_wm.events import EventQueue, SnapshotBuffer
from isac_wm.simulator import ACTIONS, ToyScene

START, TURN, END = 16, 24, 32
DT, WINDOW, TOLERANCE, PERSIST = 0.1, 10, 0.25, 3


class ChallengeScene(ToyScene):
    """Override the exogenous path only. ToyScene renders the identical camera and RF interfaces."""

    def __init__(self, seed, config, maneuver):
        if config.steps != 64 or config.dt != DT or config.room_m != 10:
            raise ValueError("Protocol requires 64 ticks, dt=0.1s and a 10m scene")
        super().__init__(
            seed, replace(config, occlusion_fraction=0, camera_dropout=0, camera_delay_s=0.2, delay_jitter_s=0)
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 204]))
        start = rng.uniform(4, 6, 2)
        angle = rng.uniform(-math.pi, math.pi)
        speed = rng.uniform(0.32, 0.48)
        v0 = speed * np.array([math.cos(angle), math.sin(angle)])
        rotation = rng.choice([-1, 1])
        v1 = rotation * np.array([-v0[1], v0[0]]) if maneuver else v0
        t = np.arange(config.steps)[:, None] * DT
        pivot = TURN * DT
        self._truth[:, :2] = start + np.minimum(t, pivot) * v0 + np.maximum(0, t - pivot) * v1
        self._truth[:, 2:] = np.where(t < pivot, v0, v1)
        if np.any((self._truth[:, :2] < 0.8) | (self._truth[:, :2] > 9.2)):
            raise ValueError("Challenge trajectory escaped ToyScene reflection domain")

    def camera(self, step):
        if START <= step < END:
            return None
        return super().camera(step)


def observed_trace(seed, cfg, maneuver):
    scene = ChallengeScene(seed, cfg, maneuver)
    q, buffer = EventQueue(), SnapshotBuffer(cfg.image_size)
    actions = np.random.default_rng(np.random.SeedSequence([seed, 33]))
    anchor, airtime, requested, missed, suppressed = None, 0.0, 0, 0, 0
    snaps, deliveries = [], []
    for k in range(cfg.steps):
        t = k * cfg.dt
        ready = q.release(t)
        snapshot = buffer.update(ready, t, cfg.dt)
        snaps.append(snapshot)
        deliveries.append(ready)
        if anchor is None and any(o.modality == "vision" and o.capture_s >= END * cfg.dt for o in ready):
            anchor = k
        action = int(actions.choice([0, 1, 2], p=[0.30, 0.45, 0.25]))
        camera, rf = scene.camera(k), scene.radio(k, action)
        requested += int(action != 0)
        missed += int(action != 0 and rf is None)
        suppressed += int(START <= k < END and k % cfg.camera_period == 0)
        airtime += cfg.dt * ACTIONS[action].airtime_fraction
        for obs in (camera, rf):
            if obs is not None:
                q.push(obs)
    if anchor is None or anchor + WINDOW >= cfg.steps:
        raise ValueError("Recovery packet not delivered with a full follow-up window")
    metadata = {
        "camera_blackout_start_tick": START,
        "camera_first_possible_recapture_tick": END,
        "first_post_blackout_camera_delivery_tick": anchor,
        "rf_requests": requested,
        "rf_capture_drops": missed,
        "suppressed_camera_captures": suppressed,
        "rf_airtime_s": float(airtime),
        "undelivered_at_end": len(q),
        "older_snapshots_ignored": buffer.ignored_older,
    }
    return snaps, deliveries, scene.truth_for_evaluation(), metadata


def score(estimated_xy, truth_xy, anchor):
    est, truth = np.asarray(estimated_xy, dtype=float), np.asarray(truth_xy, dtype=float)
    if est.shape != truth.shape or est.ndim != 2 or est.shape[1] != 2:
        raise ValueError("Need paired XY positions in metres")
    if not (np.isfinite(est).all() and np.isfinite(truth).all()):
        raise ValueError("Non-finite trajectory cannot be silently skipped")
    if not (START < anchor < len(est) - WINDOW):
        raise ValueError("No complete causal recovery window")
    errors = np.linalg.norm(est - truth, axis=-1)
    post = errors[anchor : anchor + WINDOW + 1]
    # Recovery time is when the third consecutive in-tolerance estimate becomes OBSERVABLE.
    confirmation = next((j for j in range(PERSIST - 1, len(post)) if max(post[j-2:j+1]) <= TOLERANCE), None)
    return {
        "position_error_auc_m_s": float(np.trapezoid(post, dx=DT)),
        "blackout_rmse_m": float(np.sqrt(np.mean(errors[START:END] ** 2))),
        "post_errors_m": post.tolist(),
        "recovery_confirmed_s": None if confirmation is None else confirmation * DT,
        "not_recovered_by_1s": confirmation is None,
    }
