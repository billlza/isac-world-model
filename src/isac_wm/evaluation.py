from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .baselines import AirtimeBudget, ReplayKalman, predictive_action
from .events import EventQueue, SnapshotBuffer
from .model import EventBeliefModel
from .simulator import SceneConfig, ToyScene, communication_bits
from .training import INPUT_KEYS


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def rollout(seed: int, config: SceneConfig, estimator: str, policy: str,
            model: EventBeliefModel | None = None, device: torch.device | None = None) -> dict:
    if estimator not in {"kalman", "model"} or policy not in {"fixed", "predictive"}:
        raise ValueError("Unknown estimator/policy")
    if estimator == "model" and (model is None or device is None):
        raise ValueError("Neural estimator needs a loaded model/device")
    scene, queue, buffer = ToyScene(seed, config), EventQueue(), SnapshotBuffer(config.image_size)
    tracker, budget = ReplayKalman(config.room_m), AirtimeBudget()
    memory = model.initial(1, device) if estimator == "model" else None
    rows, timings = [], []
    truth = scene.truth_for_evaluation()  # confined to evaluator; not supplied to model/policy
    for k in range(config.steps):
        now = k*config.dt
        ready = queue.release(now)
        snap = buffer.update(ready, now, config.dt)
        tick = time.perf_counter()
        if estimator == "kalman":
            tracker.ingest(ready, now)
            state, cov = tracker.estimate(now)
            var = np.diag(cov)
            decision_var = var[:2]+.3*var[2:]
        else:
            batch = {key: torch.from_numpy(snap[key])[None].to(device) for key in INPUT_KEYS}
            with torch.no_grad():
                pred, memory = model.step(batch, memory)
                future = model.forecast(pred["hidden"], .3)
                state = pred["mean"][0].cpu().numpy()*config.state_scale
                var = pred["logvar"][0].exp().cpu().numpy()*config.state_scale**2
                decision_var = future["logvar"][0, :2].exp().cpu().numpy()*config.room_m**2
            synchronize(device)
        elapsed = time.perf_counter()-tick
        timings.append(elapsed)
        budget.refill()
        requested = (1 if policy == "fixed" else predictive_action(
            decision_var, float(snap["age"].max()), config.room_m, budget.tokens))
        action = budget.authorize(requested)
        # Measurement acquisition AFTER decision, regardless of processing delay.
        for obs in (scene.camera(k), scene.radio(k, action)):
            if obs is not None:
                queue.push(obs)
        error = float(np.linalg.norm(state[:2]-truth[k, :2]))
        # Uncalibrated diagnostic acceptance heuristic; not a risk certificate.
        radius = 2.448*math.sqrt(float(max(var[:2])))
        accepted = radius <= .75
        rows.append({"step": k, "time_s": now, "estimate": state.tolist(),
                     "truth": truth[k].tolist(), "position_variance": var[:2].tolist(),
                     "error_m": error, "heuristic_radius_m": radius, "accepted": accepted,
                     "action": action, "communication_bits_proxy": communication_bits(config, action),
                     "age_s": snap["age"].tolist(), "valid": snap["valid"].tolist(),
                     "arrived_ids": [o.uid for o in ready], "queue_depth": len(queue),
                     "compute_s": elapsed})
    scored = rows[5:]  # startup is retained in trace, explicitly excluded from summary
    errors = np.array([r["error_m"] for r in scored])
    accepted_rows = [r for r in scored if r["accepted"]]
    metrics = {"position_rmse_m": float(np.sqrt(np.mean(errors**2))),
               "position_p95_m": float(np.quantile(errors, .95)),
               "acceptance_fraction": len(accepted_rows)/len(scored),
               "accepted_error_gt_0_75m": (None if not accepted_rows else
                    sum(r["error_m"] > .75 for r in accepted_rows)/len(accepted_rows)),
               "sensing_airtime_fraction": budget.spent/config.steps,
               "communication_bits_proxy": sum(r["communication_bits_proxy"] for r in rows),
               "compute_p50_ms": float(np.quantile(timings, .5)*1000),
               "compute_p95_ms": float(np.quantile(timings, .95)*1000),
               "ignored_older_snapshot_packets": buffer.ignored_older,
               "undelivered_at_end": len(queue)}
    return {"seed": seed, "estimator": estimator, "policy": policy,
            "metrics": metrics, "trace": rows}


def evaluate(outdir: Path, model: EventBeliefModel, scene: SceneConfig, device: torch.device,
             episodes: int = 8, seed: int = 30_000_000) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir/"evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; use another directory")
    runs = []
    model.eval()
    for estimator in ("kalman", "model"):
        for policy in ("fixed", "predictive"):
            for s in range(seed, seed+episodes):
                runs.append(rollout(s, scene, estimator, policy, model, device))
    summary = {}
    for estimator in ("kalman", "model"):
        for policy in ("fixed", "predictive"):
            metrics = [r["metrics"] for r in runs if r["estimator"] == estimator and r["policy"] == policy]
            summary[f"{estimator}+{policy}"] = {
                key: (None if not (values := [m[key] for m in metrics if m[key] is not None])
                      else float(np.mean(values))) for key in metrics[0]}
    report = {"status": "engineering smoke experiment; no superiority claim",
              "scored_from_step": 5, "scene": scene.to_dict(), "test_seed_start": seed,
              "test_episodes": episodes, "summary": summary, "runs": runs,
              "limitations": ["synthetic orthographic images and processed radio features",
                              "AWGN time-sharing proxy, no waveform-level ISAC or quantum model",
                              "reported compute time is measured; it does not advance simulation clock",
                              "all methods share an airtime cap; actual usage may differ",
                              "neural late older packets are ignored; Kalman replays all delivered evidence",
                              "variances and acceptance thresholds have not been calibrated"]}
    (outdir/"evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
