"""Read-only correction observers for the paired recovery challenge.

The replay receives only delivered snapshots. Evaluation truth is joined AFTER
the complete forward pass, at each packet's capture tick. Diagnostic collection
adds synchronization/copying work and must not be timed as ordinary inference.
Learned measurement variances and covariance are model quantities, not calibrated
uncertainty certificates. Vision corrects first, then RF, exactly as in the model.
"""

from __future__ import annotations

import time

import numpy as np
from recovery_challenge_core import WINDOW

from isac_wm.events import SnapshotBuffer

MODALITIES = ("vision", "radio")


def capture_state_for_evaluation(truth, capture_s, dt):
    """Look up grid-aligned simulator truth; never substitute delivery-time truth."""
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[1] != 4 or not np.isfinite(truth).all():
        raise ValueError("Expected finite evaluation states with shape [ticks,4]")
    if not np.isfinite([capture_s, dt]).all() or capture_s < 0 or dt <= 0:
        raise ValueError("Invalid capture time or simulator interval")
    tick = int(round(capture_s / dt))
    if not 0 <= tick < len(truth) or not np.isclose(capture_s, tick * dt, rtol=0, atol=1e-8):
        raise ValueError("Evaluation requires an actual grid-aligned capture time")
    return truth[tick].copy()


def summarize_events(events):
    """Per-trajectory summaries; absent fresh evidence is explicit, never zero error."""
    summary = {}
    for name, selected in (
        ("all_fresh", events),
        ("recovery_window_fresh", [e for e in events if e["within_recovery_window"]]),
    ):
        panel = {}
        for modality in MODALITIES:
            rows = [e for e in selected if e["modality"] == modality]
            if not rows:
                panel[modality] = {"event_count": 0, "metrics": None}
                continue
            errors = np.asarray([e["measurement_error_at_capture_m"] for e in rows])
            panel[modality] = {
                "event_count": len(rows),
                "metrics": {
                    "measurement_error_at_capture_mean_m": float(errors.mean()),
                    "measurement_error_at_capture_rmse_m": float(np.sqrt(np.mean(errors**2))),
                    "measurement_error_at_capture_p90_m": float(np.quantile(errors, 0.9)),
                    "age_mean_s": float(np.mean([e["age_s"] for e in rows])),
                    "measurement_variance_xy_mean_m2": np.mean(
                        [e["measurement_variance_xy_m2"] for e in rows], axis=0
                    ).tolist(),
                    "innovation_norm_mean_m": float(np.mean([e["innovation_norm_m"] for e in rows])),
                    "position_gain_trace_mean": float(np.mean([np.trace(e["position_gain"]) for e in rows])),
                    "position_gain_frobenius_mean": float(
                        np.mean([np.linalg.norm(e["position_gain"]) for e in rows])
                    ),
                    "velocity_gain_frobenius_mean_s_inv": float(
                        np.mean([np.linalg.norm(e["velocity_gain_s_inv"]) for e in rows])
                    ),
                    "applied_position_correction_norm_mean_m": float(
                        np.mean([np.linalg.norm(e["applied_position_correction_m"]) for e in rows])
                    ),
                    "immediate_position_error_change_mean_m": float(
                        np.mean([e["position_error_after_m"] - e["position_error_before_m"] for e in rows])
                    ),
                    "immediate_position_correction_improved_count": sum(
                        e["position_error_after_m"] < e["position_error_before_m"] for e in rows
                    ),
                },
            }
            if modality == "vision":
                panel[modality]["metrics"].update(
                    heatmap_entropy_mean=float(np.mean([e["heatmap_entropy"] for e in rows])),
                    learned_measurement_variance_xy_mean_m2=np.mean(
                        [e["learned_measurement_variance_xy_m2"] for e in rows], axis=0
                    ).tolist(),
                )
        summary[name] = panel
    return summary


def _event_record(observed, packet, tick, cfg, anchor):
    """Convert a detached one-example observer result without access to truth."""
    raw = {key: value.cpu().numpy()[0] for key, value in observed.items()}
    if any(not np.isfinite(value).all() for value in raw.values()):
        raise ValueError("Non-finite correction diagnostics")
    now = tick * cfg.dt
    if not bool(raw["active"][0]) or not np.isclose(
        float(raw["age_s"][0]), now - packet.capture_s, rtol=1e-6, atol=2e-7
    ):
        raise ValueError("Observer active mask/age does not match delivered packet")
    scale = np.asarray([cfg.room_m, cfg.room_m, 2.0, 2.0])
    gain = raw["gain"]
    before = raw["state_mean_before"] * scale
    after = raw["state_mean_after"] * scale
    row = {
        "modality": packet.modality,
        "observation_uid": packet.uid,
        "sensor_action": packet.action,
        "delivery_tick": tick,
        "capture_s": packet.capture_s,
        "available_s": packet.available_s,
        "decision_time_s": now,
        "estimated_state_time_s": now,
        "age_s": now - packet.capture_s,
        "within_recovery_window": anchor <= tick <= anchor + WINDOW,
        "measurement_xy_m": (raw["measurement_xy"] * cfg.room_m).tolist(),
        "predicted_measurement_xy_m": (raw["predicted_measurement_xy"] * cfg.room_m).tolist(),
        "measurement_variance_xy_m2": (raw["measurement_variance_xy"] * cfg.room_m**2).tolist(),
        "innovation_xy_m": (raw["innovation_xy"] * cfg.room_m).tolist(),
        "innovation_norm_m": float(np.linalg.norm(raw["innovation_xy"]) * cfg.room_m),
        "innovation_covariance_xy_m2": (raw["innovation_covariance_xy"] * cfg.room_m**2).tolist(),
        "position_gain": gain[:2].tolist(),
        "velocity_gain_s_inv": (gain[2:] * (2 / cfg.room_m)).tolist(),
        "state_before_xy_m_velocity_mps": before.tolist(),
        "state_after_xy_m_velocity_mps": after.tolist(),
        "applied_position_correction_m": (raw["applied_state_correction"][:2] * cfg.room_m).tolist(),
        "applied_velocity_correction_mps": (raw["applied_state_correction"][2:] * 2).tolist(),
    }
    if packet.modality == "vision":
        row["heatmap_entropy"] = float(raw["heatmap_entropy"][0])
        row["learned_measurement_variance_xy_m2"] = (
            raw["learned_measurement_variance_xy"] * cfg.room_m**2
        ).tolist()
    return row


def replay_with_diagnostics(model, cfg, device, snapshots, deliveries, truth, anchor):
    """Return ``(estimated_xy_m, forecast_xy_m, JSONable_diagnostics)``.

    Consume the same snapshots and selected packets as ``observed_trace``. Only
    new-and-valid corrections produce event records. Cached valid observations
    contribute to model state as usual, but are never counted again as evidence.
    No observation, model parameter, caller memory, or checkpoint is modified.
    The absolute state/decision time is the trace's ``tick * dt``; the model's
    relative internal recurrent clock is deliberately not used as a timestamp.
    """
    import torch

    from isac_wm.artifacts import model_fingerprint
    from isac_wm.structured import StructuredEventModel
    from isac_wm.training import INPUT_KEYS

    if not isinstance(model, StructuredEventModel) or model.training:
        raise ValueError("Read-only correction replay requires a structured model in eval mode")
    if (
        len(snapshots) != cfg.steps
        or len(deliveries) != cfg.steps
        or not isinstance(anchor, int)
        or not 0 <= anchor < cfg.steps - WINDOW
        or model.config.room_m != cfg.room_m
    ):
        raise ValueError("Mismatched recovery trace, scene, or anchor")
    actual_device = next(model.parameters()).device
    requested = torch.device(device)
    if actual_device.type != requested.type or (
        requested.index is not None and requested.index != actual_device.index
    ):
        raise ValueError("Diagnostic device differs from the loaded model device")
    start = time.perf_counter()
    fingerprint = model_fingerprint(model)
    buffer = SnapshotBuffer(cfg.image_size)
    estimates, forecasts, events = [], [], []
    ticks = {modality: {"new_valid": 0, "valid_cached": 0, "unavailable": 0} for modality in MODALITIES}
    with torch.inference_mode():
        memory = model.initial(1, device)
        for k, (snap, packets) in enumerate(zip(snapshots, deliveries, strict=True)):
            # Use the actual selection contract, including rejection of older
            # arrivals. Matching the whole snapshot prevents UID/time guessing.
            rebuilt = buffer.update(packets, k * cfg.dt, cfg.dt)
            if set(snap) != set(rebuilt) or any(
                not np.array_equal(snap[key], rebuilt[key]) for key in rebuilt
            ):
                raise ValueError("Delivered packets do not reproduce the estimator snapshot")
            inputs = {key: torch.from_numpy(snap[key])[None].to(device) for key in INPUT_KEYS}
            diagnostic = {}
            current, memory = model.step(inputs, memory, diagnostics=diagnostic)
            future = model.forecast(current["hidden"], 3 * cfg.dt)
            estimates.append(current["mean"][0, :2])
            forecasts.append(future["mean"][0, :2])
            for index, modality in enumerate(MODALITIES):
                observed = diagnostic[modality]
                active = bool(snap["valid"][index] and snap["new"][index])
                if bool(observed["active"][0, 0]) != active:
                    raise ValueError("Diagnostic active mask differs from estimator inputs")
                if active:
                    ticks[modality]["new_valid"] += 1
                    events.append(_event_record(observed, buffer.latest[modality], k, cfg, anchor))
                else:
                    category = "valid_cached" if snap["valid"][index] else "unavailable"
                    ticks[modality][category] += 1
                    if bool(observed["applied_state_correction"].ne(0).any()):
                        raise RuntimeError("A cached or unavailable modality applied a correction")
        estimated = torch.stack(estimates).cpu().numpy() * cfg.room_m
        forecast = torch.stack(forecasts).cpu().numpy() * cfg.room_m
    if model_fingerprint(model) != fingerprint:
        raise RuntimeError("Read-only diagnostic replay changed model parameters")

    # This is the first access to truth. The entire causal rollout is complete.
    truth = np.asarray(truth)
    if truth.shape != (cfg.steps, 4) or not np.isfinite(truth).all():
        raise ValueError("Expected a finite complete evaluation trajectory")
    for event in events:
        captured = capture_state_for_evaluation(truth, event["capture_s"], cfg.dt)
        now = truth[event["delivery_tick"]]
        event["capture_truth_xy_m"] = captured[:2].tolist()
        event["measurement_error_at_capture_m"] = float(
            np.linalg.norm(np.asarray(event["measurement_xy_m"]) - captured[:2])
        )
        event["position_error_before_m"] = float(
            np.linalg.norm(np.asarray(event["state_before_xy_m_velocity_mps"][:2]) - now[:2])
        )
        event["position_error_after_m"] = float(
            np.linalg.norm(np.asarray(event["state_after_xy_m_velocity_mps"][:2]) - now[:2])
        )
    return (
        estimated,
        forecast,
        {
            "schema_version": 1,
            "status": "completed",
            "parameter_sha256_before_and_after": fingerprint,
            "diagnostic_replay_wall_s": time.perf_counter() - start,
            "recovery_anchor_tick": anchor,
            "recovery_window_end_tick": anchor + WINDOW,
            "correction_order": list(MODALITIES),
            "units": (
                "XY in metres, velocities in m/s, variances/covariances in m^2; position gain "
                "dimensionless and velocity gain in 1/s. Variances are model outputs, not calibrated "
                "uncertainty. Measurement errors use capture-time truth; before/after state errors "
                "use delivery-decision-time truth. Immediate correction changes are not intervention effects."
            ),
            "snapshot_counts": ticks,
            "older_packets_ignored": buffer.ignored_older,
            "summary": summarize_events(events),
            "events": events,
        },
    )
