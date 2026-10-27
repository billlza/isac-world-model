from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import write_json
from .baselines import AirtimeBudget, ReplayKalman, guarded_risk_action, predictive_action, risk_budget_action
from .events import EventQueue, SnapshotBuffer
from .imm import IMMConfig, ReplayIMM
from .model import EventBeliefModel
from .runtime import RuntimeConfig
from .safety import MonitorConfig, assess
from .simulator import SceneConfig, ToyScene, communication_bits
from .training import INPUT_KEYS
from .validation import integer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _reference_rollout(
    seed: int,
    config: SceneConfig,
    estimator: str,
    policy: str,
    model: EventBeliefModel | None = None,
    device: torch.device | None = None,
    *,
    monitor: MonitorConfig | None = None,
    quantile: float | None = None,
    sensor_budget: float = 0.1,
    gate_threshold: float | None = None,
    baseline_process_noise: float = 0.8,
    imm_config: IMMConfig | None = None,
) -> dict:
    if estimator not in {"kalman", "kalman-gated", "imm", "imm-gated", "model"} or policy not in {
        "fixed",
        "predictive",
        "risk",
        "guarded",
    }:
        raise ValueError("Unknown estimator/policy")
    if estimator == "model" and (model is None or device is None):
        raise ValueError("Neural estimator needs a loaded model/device")
    monitor = monitor or MonitorConfig()
    scene, queue, buffer = ToyScene(seed, config), EventQueue(), SnapshotBuffer(config.image_size)
    if estimator.startswith("imm"):
        from dataclasses import replace

        setup = imm_config or IMMConfig()
        threshold = 9.210340371976184 if estimator == "imm-gated" else None
        tracker = ReplayIMM(config.room_m, config=replace(setup, gate_threshold=threshold))
    else:
        threshold = 9.210340371976184 if estimator == "kalman-gated" else gate_threshold
        tracker = ReplayKalman(config.room_m, gate_threshold=threshold, process_noise=baseline_process_noise)
    budget = AirtimeBudget(sensor_budget, max(0.6, sensor_budget))
    memory = model.initial(1, device) if estimator == "model" else None
    rows, timings = [], []
    truth = scene.truth_for_evaluation()  # confined to evaluator; not supplied to model/policy
    for k in range(config.steps):
        now = k * config.dt
        ready = queue.release(now)
        snap = buffer.update(ready, now, config.dt)
        tick = time.perf_counter()
        if estimator != "model":
            tracker.ingest(ready, now)
            state, cov = tracker.estimate(now)
            var = np.diag(cov)
            decision_var = var[:2] + (3 * config.dt) ** 2 * var[2:]
        else:
            batch = {key: torch.from_numpy(snap[key])[None].to(device) for key in INPUT_KEYS}
            with torch.inference_mode():
                pred, memory = model.step(batch, memory)
                future = model.forecast(pred["hidden"], 3 * config.dt)
                state = pred["mean"][0].cpu().numpy() * config.state_scale
                var = pred["logvar"][0].exp().cpu().numpy() * config.state_scale**2
                decision_var = future["logvar"][0, :2].exp().cpu().numpy() * config.room_m**2
            synchronize(device)
        elapsed = time.perf_counter() - tick
        timings.append(elapsed)
        budget.refill()
        if policy == "fixed":
            requested = 1
        elif policy == "predictive":
            requested = predictive_action(
                decision_var, float(snap["age"].max()), config.room_m, budget.tokens
            )
        elif policy == "guarded":
            requested = guarded_risk_action(
                var[:2],
                var[2:],
                budget.tokens,
                config.room_m,
                snap["valid"],
                snap["age"],
                extra_delay_s=config.radio_delay_extra_s + config.delay_jitter_s / 2,
            )
        else:
            requested = risk_budget_action(
                var[:2],
                var[2:],
                budget.tokens,
                config.room_m,
                extra_delay_s=config.radio_delay_extra_s + config.delay_jitter_s / 2,
            )
        action = budget.authorize(requested)
        # Measurement acquisition AFTER decision, regardless of processing delay.
        for obs in (scene.camera(k), scene.radio(k, action)):
            if obs is not None:
                queue.push(obs)
        error = float(np.linalg.norm(state[:2] - truth[k, :2]))
        # Uncalibrated diagnostic acceptance heuristic; not a risk certificate.
        radius = 2.448 * math.sqrt(float(max(var[:2])))
        verdict = assess(
            state, var, snap["valid"], snap["age"], elapsed * 1000, config.room_m, monitor, quantile
        )
        accepted = verdict["accepted"]
        rows.append(
            {
                "step": k,
                "time_s": now,
                "estimate": state.tolist(),
                "truth": truth[k].tolist(),
                "position_variance": var[:2].tolist(),
                "error_m": error,
                "heuristic_radius_m": radius,
                "accepted": accepted,
                "status": verdict["status"],
                "radius_m": verdict["radius_m"],
                "requested_action": requested,
                "budget_tokens": budget.tokens,
                "action": action,
                "communication_bits_proxy": communication_bits(config, action),
                "age_s": snap["age"].tolist(),
                "valid": snap["valid"].tolist(),
                "arrived_ids": [o.uid for o in ready],
                "queue_depth": len(queue),
                "compute_s": elapsed,
            }
        )
    metrics, statuses = summarize_trace(
        rows, timings, monitor, quantile, budget, config, tracker, buffer, queue
    )
    if estimator == "model":
        metrics["replayed_filter_updates"] = 0
        metrics["gated_radio_or_vision_measurements"] = 0
    return {
        "seed": seed,
        "estimator": estimator,
        "policy": policy,
        "runtime": RuntimeConfig().to_dict(),
        "metrics": metrics,
        "status_counts": statuses,
        "trace": rows,
    }


def summarize_trace(rows, timings, monitor, quantile, budget, config, tracker, buffer, queue):
    scored = rows[5:]  # startup is retained in trace, explicitly excluded from summary
    errors = np.array([r["error_m"] for r in scored])
    accepted_rows = [r for r in scored if r["accepted"]]
    statuses = {s: sum(r["status"] == s for r in scored) for s in sorted({r["status"] for r in scored})}
    metrics = {
        "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "position_p95_m": float(np.quantile(errors, 0.95)),
        "acceptance_fraction": len(accepted_rows) / len(scored),
        "accepted_error_gt_0_75m": (
            None
            if not accepted_rows
            else sum(r["error_m"] > 0.75 for r in accepted_rows) / len(accepted_rows)
        ),
        "accepted_error_exceeds_tolerance": (
            None
            if not accepted_rows
            else sum(r["error_m"] > monitor.error_tolerance_m for r in accepted_rows) / len(accepted_rows)
        ),
        "accepted_count": len(accepted_rows),
        "scored_count": len(scored),
        "sensing_airtime_fraction": budget.spent / config.steps,
        "communication_bits_proxy": sum(r["communication_bits_proxy"] for r in rows),
        "compute_p50_ms": float(np.quantile(timings, 0.5) * 1000),
        "compute_p95_ms": float(np.quantile(timings, 0.95) * 1000),
        "compute_p99_ms": float(np.quantile(timings, 0.99) * 1000),
        "deadline_miss_fraction": sum(t * 1000 > monitor.compute_deadline_ms for t in timings) / len(timings),
        "queue_peak": max(r["queue_depth"] for r in rows),
        "maximum_error_m": float(errors.max()),
        "replayed_filter_updates": tracker.replayed_updates,
        "gated_radio_or_vision_measurements": tracker.rejected_measurements,
        "action_skip_count": sum(r["action"] == 0 for r in rows),
        "action_quick_count": sum(r["action"] == 1 for r in rows),
        "action_precise_count": sum(r["action"] == 2 for r in rows),
        "calibrated_episode_covered": (
            None
            if quantile is None
            else float(
                all(
                    np.sum(
                        (np.asarray(r["estimate"][:2]) - np.asarray(r["truth"][:2])) ** 2
                        / np.asarray(r["position_variance"])
                    )
                    <= quantile**2
                    for r in scored
                )
            )
        ),
        "ignored_older_snapshot_packets": buffer.ignored_older,
        "undelivered_at_end": len(queue),
    }
    return metrics, statuses


def rollout(
    seed: int,
    config: SceneConfig,
    estimator: str,
    policy: str,
    model: EventBeliefModel | None = None,
    device: torch.device | None = None,
    *,
    monitor: MonitorConfig | None = None,
    quantile: float | None = None,
    sensor_budget: float = 0.1,
    gate_threshold: float | None = None,
    runtime: RuntimeConfig | None = None,
    baseline_process_noise: float = 0.8,
    imm_config: IMMConfig | None = None,
) -> dict:
    integer(seed, "seed", 0, 2**32 - 1)
    if estimator not in {"kalman", "kalman-gated", "imm", "imm-gated", "model"} or policy not in {
        "fixed",
        "predictive",
        "risk",
        "guarded",
    }:
        raise ValueError("Unknown estimator/policy")
    if estimator == "model" and (model is None or device is None):
        raise ValueError("Neural estimator needs a loaded model/device")
    runtime = runtime or RuntimeConfig()
    if runtime.mode == "measured" and quantile is not None:
        raise ValueError("Measured timing is diagnostic; use fixed timing for calibration")
    options = dict(
        monitor=monitor,
        quantile=quantile,
        sensor_budget=sensor_budget,
        gate_threshold=gate_threshold,
        baseline_process_noise=baseline_process_noise,
        imm_config=imm_config,
    )
    if runtime.mode == "reference":
        return _reference_rollout(seed, config, estimator, policy, model, device, **options)
    from .streaming import streaming_rollout

    return streaming_rollout(seed, config, estimator, policy, model, device, runtime=runtime, **options)


def evaluate(
    outdir: Path,
    model: EventBeliefModel,
    scene: SceneConfig,
    device: torch.device,
    episodes: int = 8,
    seed: int = 30_000_000,
    *,
    policies: tuple[str, ...] = ("fixed", "predictive"),
    calibration: dict | None = None,
    monitor: MonitorConfig | None = None,
    sensor_budget: float = 0.1,
    runtime: RuntimeConfig | None = None,
    estimators: tuple[str, ...] = ("kalman", "model"),
    baseline_process_noise: float = 0.8,
    imm_config: IMMConfig | None = None,
) -> dict:
    runtime = runtime or RuntimeConfig()
    integer(episodes, "episodes", 1, 10_000)
    integer(seed, "seed", 0, 2**32 - 1 - episodes)
    if (
        not policies
        or len(policies) != len(set(policies))
        or set(policies) - {"fixed", "predictive", "risk", "guarded"}
    ):
        raise ValueError("Invalid policies")
    if (
        not estimators
        or len(estimators) != len(set(estimators))
        or set(estimators) - {"kalman", "kalman-gated", "imm", "imm-gated", "model"}
    ):
        raise ValueError("Invalid estimators")
    if calibration is not None:
        from .calibration import validate_calibration

        validate_calibration(calibration, model, scene, list(range(seed, seed + episodes)), runtime=runtime)
        if "kalman" in estimators and baseline_process_noise != 0.8:
            raise ValueError(
                "Calibration was collected with Kalman process noise 0.8; changed process noise is unsupported"
            )
        if sensor_budget != 0.1:
            raise ValueError(
                "Calibration was collected at default budget; changed budget needs new calibration"
            )
    if set(range(seed, seed + episodes)).intersection(getattr(model, "provenance_seeds", set())):
        raise ValueError("Evaluation episodes overlap training/validation")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; use another directory")
    runs = []
    model.eval()
    for estimator in estimators:
        for policy in policies:
            for s in range(seed, seed + episodes):
                quantile = None
                if calibration is not None:
                    key = f"{estimator}+{policy}"
                    if key not in calibration["variants"]:
                        raise ValueError("No calibration for the selected estimator/policy")
                    quantile = calibration["variants"][key]["quantile"]
                runs.append(
                    rollout(
                        s,
                        scene,
                        estimator,
                        policy,
                        model,
                        device,
                        quantile=quantile,
                        monitor=monitor,
                        sensor_budget=sensor_budget,
                        runtime=runtime,
                        baseline_process_noise=baseline_process_noise,
                        imm_config=imm_config,
                    )
                )
    summary = {}
    for estimator in estimators:
        for policy in policies:
            metrics = [r["metrics"] for r in runs if r["estimator"] == estimator and r["policy"] == policy]
            summary[f"{estimator}+{policy}"] = {
                key: (
                    None
                    if not (values := [m[key] for m in metrics if m[key] is not None])
                    else float(np.mean(values))
                )
                for key in metrics[0]
            }
    report = {
        "status": "engineering smoke experiment; no superiority claim",
        "schema_version": 2,
        "runtime": runtime.to_dict(),
        "scored_from_step": 5,
        "scene": scene.to_dict(),
        "test_seed_start": seed,
        "test_episodes": episodes,
        "summary": summary,
        "runs": runs,
        "monitor": (monitor or MonitorConfig()).to_dict(),
        "calibration": "none"
        if calibration is None
        else {"fingerprint": calibration["model_fingerprint"], "scope": calibration["scope"]},
        "limitations": [
            "synthetic orthographic images and processed radio features",
            "AWGN time-sharing proxy, no waveform-level ISAC or quantum model",
            (
                "reference: measured compute does not advance the simulation clock"
                if runtime.mode == "reference"
                else "single-worker tick-quantized timing; no GPU/OS scheduler or physical latency validation"
            ),
            "outputs between completions use an explicit constant-velocity covariance model",
            "all methods share an airtime cap; actual usage may differ",
            "neural late older packets are ignored; Kalman replays all delivered evidence",
            (
                "no calibration: results are not accepted by the default monitor"
                if calibration is None
                else "episode-marginal conformal diagnostic; no conditional accepted-risk/OOD safety guarantee"
            ),
        ],
    }
    write_json(outdir / "evaluation.json", report)
    return report
