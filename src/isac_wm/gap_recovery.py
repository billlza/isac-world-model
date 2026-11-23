"""Causal RF capture-gap traces for the existing bounded 2-D toy scene.

This protocol changes only acquisition: the exogenous ``ToyScene`` trajectory
is unchanged. No camera is sampled, and radial velocity is zeroed in every RF
payload so position-only replay Kalman/IMM baselines receive the same evidence.
The capture gap is not the delivery gap: already captured packets may arrive
after acquisitions stop. Both intervals are recorded explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .events import EventQueue, Observation, SnapshotBuffer
from .simulator import SceneConfig, ToyScene
from .validation import integer


@dataclass(frozen=True)
class GapRecoverySpec:
    gap_start_step: int = 12
    gap_steps: int = 6
    action: int = 1

    def __post_init__(self) -> None:
        integer(self.gap_start_step, "gap_start_step", 1)
        integer(self.gap_steps, "gap_steps", 0)
        integer(self.action, "action", 1, 2)


@dataclass(frozen=True)
class GapRecoveryEpisode:
    """Separate supervision, accepted online events, and protocol metadata.

    ``arrays`` supports the existing ``batchify`` helper. Online callers must
    still select only ``training.INPUT_KEYS``; phase/target/count arrays are
    evaluation or supervision, never estimator inputs. ``delivered[t]`` holds
    exactly the final fresh event accepted by the neural snapshot at tick t,
    not all raw arrivals. Baselines using this tuple therefore have no extra
    same-tick or out-of-order measurements. Exact times live in ``metadata``
    and in the immutable ``Observation`` objects, outside model inputs.
    """

    arrays: dict[str, np.ndarray]
    delivered: tuple[tuple[Observation, ...], ...]
    metadata: dict


def generate_gap_episode(seed: int, scene: SceneConfig, spec: GapRecoverySpec) -> GapRecoveryEpisode:
    """Record history, a capture gap, and resumed fixed-action RF acquisitions.

    The gap is the integer capture interval ``[start, start + gap_steps)``;
    acquisitions outside it use ``spec.action``. Sensor dropout, jitter, bias
    and outliers are inherited explicitly from ``scene``. Camera settings are
    irrelevant because this RF-only protocol never calls ``camera``.

    Phase boundaries use accepted arrivals: history ends at the last accepted
    pre-gap packet, the no-new-evidence gap lasts until the first accepted
    post-gap packet, and recovery begins at that packet's update tick. A
    configuration/seed lacking pre-gap or recovery evidence fails explicitly.
    A nonzero capture gap must produce at least one no-new-evidence tick.
    """
    if not isinstance(scene, SceneConfig) or not isinstance(spec, GapRecoverySpec):
        raise TypeError("Expected SceneConfig and GapRecoverySpec")
    end_step = spec.gap_start_step + spec.gap_steps
    if end_step >= scene.steps - 1:
        raise ValueError("The capture gap must leave history and recovery ticks")

    world = ToyScene(seed, scene)
    queue, buffer = EventQueue(), SnapshotBuffer(scene.image_size)
    truth = world.truth_for_evaluation()
    snapshots, delivered, captured_steps = [], [], {}
    actions, radio_targets, radio_valid = [], [], []
    raw_arrivals, ignored_older, superseded = [], [], []
    accepted_capture_s, accepted_available_s, accepted_uid = [], [], []
    pre_gap_ticks, recovery_ticks = [], []
    acquired_count = 0

    for tick in range(scene.steps):
        now = tick * scene.dt
        ready = queue.release(now)
        ignored_before = buffer.ignored_older
        snap = buffer.update(ready, now, scene.dt)
        snapshots.append(snap)
        accepted = buffer.latest.get("radio") if snap["new"][1] else None
        delivered.append(() if accepted is None else (accepted,))
        raw_arrivals.append(len(ready))
        ignored_count = buffer.ignored_older - ignored_before
        ignored_older.append(ignored_count)
        superseded.append(len(ready) - ignored_count - int(accepted is not None))
        accepted_capture_s.append(None if accepted is None else accepted.capture_s)
        accepted_available_s.append(None if accepted is None else accepted.available_s)
        accepted_uid.append(None if accepted is None else accepted.uid)
        if accepted is not None:
            capture_step = captured_steps[accepted.uid]
            if capture_step < spec.gap_start_step:
                pre_gap_ticks.append(tick)
            elif capture_step >= end_step:
                recovery_ticks.append(tick)
            else:
                raise RuntimeError("A packet captured inside the gap entered the online trace")

        # Acquire only after building this tick's snapshot. Suppression is keyed
        # by integer CAPTURE step; queued pre-gap evidence remains deliverable.
        action = 0 if spec.gap_start_step <= tick < end_step else spec.action
        actions.append(action)
        observation = world.radio(tick, action)
        if observation is not None:
            payload = observation.payload.copy()
            payload[3] = 0.0
            observation = Observation(
                observation.uid,
                "radio",
                observation.capture_s,
                observation.available_s,
                payload,
                observation.action,
            )
            queue.push(observation)
            captured_steps[observation.uid] = tick
            acquired_count += 1
        radio_targets.append(np.zeros(4, np.float32) if observation is None else observation.payload)
        radio_valid.append(float(observation is not None))

    if not pre_gap_ticks:
        raise ValueError("No pre-gap RF packet was accepted; increase history or reduce delay/dropout")
    if not recovery_ticks:
        raise ValueError("No recovery RF packet was accepted; extend the episode or reduce delay/dropout")
    anchor, recovery = pre_gap_ticks[-1], recovery_ticks[0]
    if anchor >= recovery:
        raise RuntimeError("Accepted acquisition ordering violated the gap protocol")
    gap_ticks = list(range(anchor + 1, recovery))
    if spec.gap_steps and not gap_ticks:
        raise ValueError("This capture gap produced no no-new-evidence interval")
    if any(snapshots[tick]["new"][1] for tick in gap_ticks):
        raise RuntimeError("The declared delivery gap contains fresh RF evidence")

    arrays = {key: np.stack([snapshot[key] for snapshot in snapshots]) for key in snapshots[0]}
    arrays.update(
        state_target=truth / scene.state_scale,
        action_target=np.array(actions, np.int64),
        radio_target=np.stack(radio_targets),
        radio_target_valid=np.array(radio_valid, np.float32),
        vision_capture_target_xy=np.zeros((scene.steps, 2), np.float32),
        vision_target_mask=np.zeros(scene.steps, np.float32),
        vision_capture_s=np.full(scene.steps, -1.0, np.float64),
        vision_event_uid=np.full(scene.steps, -1, np.int64),
        state_reference_variance=np.zeros_like(truth, dtype=np.float32),
        state_reference_validity=np.ones_like(truth, dtype=np.float32),
    )
    ticks = np.arange(scene.steps)
    arrays["phase_history"] = ticks <= anchor
    arrays["phase_gap"] = (ticks > anchor) & (ticks < recovery)
    arrays["phase_recovery"] = ticks >= recovery
    recovered_packets = np.zeros(scene.steps, np.int64)
    recovered_packets[recovery_ticks] = 1
    arrays["recovery_packet_count"] = np.cumsum(recovered_packets)

    metadata = {
        "schema_version": 1,
        "protocol": "rf-capture-gap-position-only-v1",
        "scope": "existing single-target bounded 2-D ToyScene; not outdoor channel or beam simulation",
        "seed": int(seed),
        "scene": scene.to_dict(),
        "spec": asdict(spec),
        "camera_sampled": False,
        "radial_velocity_masked": True,
        "capture_gap_start_s": spec.gap_start_step * scene.dt,
        "capture_gap_end_s": end_step * scene.dt,
        "capture_gap_steps": list(range(spec.gap_start_step, end_step)),
        "forecast_anchor_step": anchor,
        "forecast_anchor_state_time_s": anchor * scene.dt,
        "forecast_anchor_capture_s": accepted_capture_s[anchor],
        "forecast_anchor_available_s": accepted_available_s[anchor],
        "first_recovery_step": recovery,
        "first_recovery_state_time_s": recovery * scene.dt,
        "first_recovery_capture_s": accepted_capture_s[recovery],
        "first_recovery_available_s": accepted_available_s[recovery],
        "gap_no_new_steps": gap_ticks,
        "gap_no_new_start_s": (anchor + 1) * scene.dt,
        "gap_no_new_end_s": recovery * scene.dt,
        "gap_no_new_duration_s": len(gap_ticks) * scene.dt,
        "state_time_s": (ticks * scene.dt).tolist(),
        "accepted_radio_capture_s": accepted_capture_s,
        "accepted_radio_available_s": accepted_available_s,
        "accepted_radio_uid": accepted_uid,
        "recovery_packet_steps": recovery_ticks,
        "rf_attempted_captures": scene.steps - spec.gap_steps,
        "rf_suppressed_capture_steps": spec.gap_steps,
        "rf_capture_dropouts": scene.steps - spec.gap_steps - acquired_count,
        "rf_acquired_packets": acquired_count,
        "rf_pending_after_final_tick": len(queue),
        "rf_raw_arrivals_per_tick": raw_arrivals,
        "rf_ignored_older_per_tick": ignored_older,
        "rf_superseded_same_tick": superseded,
        "rf_snapshot_accepted_packets": sum(len(events) for events in delivered),
    }
    return GapRecoveryEpisode(arrays, tuple(delivered), metadata)
