"""Causal virtual-time rollout with a single non-preemptive inference worker.

In-flight computation is evaluated immediately offline but its result, sensing
command, and monitor decision are embargoed until its declared finish time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .baselines import AirtimeBudget, ReplayKalman, guarded_risk_action, predictive_action, risk_budget_action
from .events import EventQueue, SnapshotBuffer
from .imm import IMMConfig, ReplayIMM
from .safety import MonitorConfig, assess
from .simulator import ToyScene, communication_bits
from .training import INPUT_KEYS


@dataclass
class _Result:
    start_s: float
    finish_s: float
    measured_s: float
    state: np.ndarray
    covariance: np.ndarray
    valid: np.ndarray
    age: np.ndarray
    requested_action: int
    observation_ids: list[int]


def propagate(state: np.ndarray, covariance: np.ndarray, dt: float, noise: float):
    """Constant velocity; preserve structured neural cross-covariance when supplied."""
    if dt < -1e-10:
        raise ValueError("Cannot publish an estimate before its state time")
    dt = max(0.0, dt)
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    q = noise * np.array(
        [
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt, 0],
            [0, dt**2 / 2, 0, dt],
        ]
    )
    cov = f @ covariance @ f.T + q
    return f @ state, (cov + cov.T) / 2


def streaming_rollout(
    seed,
    config,
    estimator,
    policy,
    model,
    device,
    *,
    runtime,
    monitor=None,
    quantile=None,
    sensor_budget=0.1,
    gate_threshold=None,
    baseline_process_noise=0.8,
    imm_config: IMMConfig | None = None,
):
    from .evaluation import summarize_trace, synchronize

    monitor = monitor or MonitorConfig()
    scene = ToyScene(seed, config)
    queue = EventQueue(capacity=runtime.queue_capacity)
    buffer = SnapshotBuffer(config.image_size)
    if estimator.startswith("imm"):
        from dataclasses import replace

        setup = imm_config or IMMConfig()
        gate = 9.210340371976184 if estimator == "imm-gated" else None
        tracker = ReplayIMM(config.room_m, config=replace(setup, gate_threshold=gate))
    else:
        gate = 9.210340371976184 if estimator == "kalman-gated" else gate_threshold
        tracker = ReplayKalman(config.room_m, gate_threshold=gate, process_noise=baseline_process_noise)
    budget = AirtimeBudget(sensor_budget, max(0.6, sensor_budget))
    memory = model.initial(1, device) if estimator == "model" else None
    truth = scene.truth_for_evaluation()
    pending = published = None
    last_start = None
    rows, timings, service_times, jobs = [], [], [], []
    overflow_total = 0
    for k in range(config.steps):
        now = k * config.dt
        budget.refill()
        action, requested, just_published = 0, 0, False
        ready_ids = []
        if pending is not None and pending.finish_s <= now + 1e-12:
            published = pending
            pending = None
            requested = published.requested_action
            action = budget.authorize(requested)
            just_published = True
            jobs[-1]["published_s"] = now
            jobs[-1]["executed_action"] = action
        busy_at_tick = pending is not None
        if not busy_at_tick:
            ready = queue.release(now)
            ready_ids = [o.uid for o in ready]
            dt = config.dt if last_start is None else now - last_start
            snap = buffer.update(ready, now, dt)
            if estimator == "model":
                synchronize(device)
            started = time.perf_counter()
            if estimator != "model":
                tracker.ingest(ready, now)
                state, covariance = tracker.estimate(now)
                var = np.diag(covariance)
                decision_var = var[:2] + (3 * config.dt) ** 2 * var[2:]
            else:
                batch = {key: torch.from_numpy(snap[key])[None].to(device) for key in INPUT_KEYS}
                with torch.inference_mode():
                    pred, memory = model.step(batch, memory)
                    future = model.forecast(pred["hidden"], 3 * config.dt)
                    state = pred["mean"][0].cpu().numpy() * config.state_scale
                    var = pred["logvar"][0].exp().cpu().numpy() * config.state_scale**2
                    covariance = (
                        pred["covariance"][0].cpu().numpy() * np.outer(config.state_scale, config.state_scale)
                        if "covariance" in pred
                        else np.diag(var)
                    )
                    decision_var = future["logvar"][0, :2].exp().cpu().numpy() * config.room_m**2
                synchronize(device)
            if policy == "fixed":
                request = 1
            elif policy == "predictive":
                request = predictive_action(
                    decision_var, float(snap["age"].max()), config.room_m, budget.tokens
                )
            elif policy == "guarded":
                request = guarded_risk_action(
                    var[:2],
                    var[2:],
                    budget.tokens,
                    config.room_m,
                    snap["valid"],
                    snap["age"],
                    extra_delay_s=config.radio_delay_extra_s + config.delay_jitter_s / 2,
                )
            else:
                request = risk_budget_action(
                    var[:2],
                    var[2:],
                    budget.tokens,
                    config.room_m,
                    extra_delay_s=config.radio_delay_extra_s + config.delay_jitter_s / 2,
                )
            elapsed = time.perf_counter() - started
            service = runtime.service_seconds(elapsed)
            pending = _Result(
                now,
                now + service,
                elapsed,
                state.copy(),
                covariance.copy(),
                snap["valid"].copy(),
                snap["age"].copy(),
                request,
                ready_ids,
            )
            jobs.append(
                {
                    "start_s": now,
                    "input_cutoff_s": now,
                    "finish_s": pending.finish_s,
                    "service_s": service,
                    "measured_compute_s": elapsed,
                    "input_ids": ready_ids,
                    "max_input_available_s": max((o.available_s for o in ready), default=None),
                    "published_s": None,
                    "requested_action": request,
                    "executed_action": None,
                }
            )
            timings.append(elapsed)
            service_times.append(service)
            last_start = now
            if pending.finish_s <= now + 1e-12:
                # Fixed zero is a useful reference-equivalence test. No future read.
                published, pending = pending, None
                requested = published.requested_action
                action = budget.authorize(requested)
                just_published = True
                jobs[-1]["published_s"] = now
                jobs[-1]["executed_action"] = action
        overflow_ids = []
        captured_ids = []
        # Camera is independent of inference. Radio is acquired only AFTER the
        # completed decision is actually executed, never at the old job start.
        for obs in (scene.camera(k), scene.radio(k, action)):
            if obs is not None:
                captured_ids.append(obs.uid)
                try:
                    queue.push(obs)
                except OverflowError:
                    overflow_total += 1
                    overflow_ids.append(obs.uid)  # explicit drop-new policy, not hidden
        if published is None:
            state = np.array([config.room_m / 2, config.room_m / 2, 0.0, 0.0])
            covariance = np.diag([config.room_m**2 / 12] * 2 + [1.0, 1.0])
            valid = np.zeros(2)
            age = np.full(2, 20.0)
            verdict = {"accepted": False, "status": "no_output", "radius_m": None}
            output_start = output_finish = None
            compute = 0.0
        else:
            output_start, output_finish = published.start_s, published.finish_s
            state, covariance = propagate(
                published.state,
                published.covariance,
                now - published.start_s,
                runtime.output_acceleration_noise,
            )
            valid = published.valid
            age = published.age.astype(float) + (now - published.start_s)
            compute = published.measured_s
            verdict = assess(
                state,
                np.diag(covariance),
                valid,
                age,
                (published.finish_s - published.start_s) * 1000,
                config.room_m,
                monitor,
                quantile,
            )
        variance = np.diag(covariance)
        error = float(np.linalg.norm(state[:2] - truth[k, :2]))
        rows.append(
            {
                "step": k,
                "time_s": now,
                "estimate": state.tolist(),
                "truth": truth[k].tolist(),
                "position_variance": variance[:2].tolist(),
                "error_m": error,
                "heuristic_radius_m": 2.448 * math.sqrt(float(max(variance[:2]))),
                "accepted": verdict["accepted"],
                "status": verdict["status"],
                "radius_m": verdict["radius_m"],
                "action": action,
                "requested_action": requested,
                "budget_tokens": budget.tokens,
                "communication_bits_proxy": communication_bits(config, action),
                "age_s": age.tolist(),
                "valid": valid.tolist(),
                "arrived_ids": ready_ids,
                "queue_depth": len(queue),
                "compute_s": compute,
                "output_available": published is not None,
                "output_input_cutoff_s": output_start,
                "output_ready_s": output_finish,
                "output_state_time_s": now,
                "new_output": just_published,
                "busy_at_tick": busy_at_tick,
                "captured_ids": captured_ids,
                "overflow_dropped_ids": overflow_ids,
            }
        )
    metrics, statuses = summarize_trace(
        rows, timings, monitor, quantile, budget, config, tracker, buffer, queue
    )
    scored = rows[5:]
    metrics.update(
        {
            "no_output_fraction": sum(not r["output_available"] for r in scored) / len(scored),
            "busy_tick_fraction": sum(r["busy_at_tick"] for r in rows) / len(rows),
            "completed_jobs": sum(j["published_s"] is not None for j in jobs),
            "started_jobs": len(jobs),
            "unfinished_jobs": int(pending is not None),
            "overflow_dropped_count": overflow_total,
            "service_p95_ms": float(np.quantile(service_times, 0.95) * 1000),
            "service_deadline_miss_fraction": sum(
                t * 1000 > monitor.compute_deadline_ms for t in service_times
            )
            / len(service_times),
        }
    )
    return {
        "seed": seed,
        "estimator": estimator,
        "policy": policy,
        "runtime": runtime.to_dict(),
        "metrics": metrics,
        "status_counts": statuses,
        "trace": rows,
        "jobs": jobs,
    }
