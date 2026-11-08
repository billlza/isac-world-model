"""Fixed-update capacity x nested-data experiment, with two explicit initializations.

The formal test set is read only after all eight best checkpoints are frozen.
Development uses different trajectories and short budgets, retaining both real widths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (96, 512)
INITIALIZATIONS = (7, 11)
CONDITIONS = ("control", "maneuver")
ORDER_SEED = 37
PARAMETERS = {96: 172246, 512: 3883798}
METRICS = ("position_error_auc_m_s", "three_step_forecast_rmse_m")


def design(profile):
    common = {
        "widths": list(WIDTHS),
        "initializations": list(INITIALIZATIONS),
        "architecture": "structured-attention",
        "feature_dim": 48,
        "memory_size": 16,
        "heads": 4,
        "minibatch_order_seed": ORDER_SEED,
        "learning_rate": 0.001,
        "steps": 64,
        "forecast_steps": 3,
        "selection_metric": "validation_current_position_rmse_m_after_tick_4",
        "early_stopping_patience": 0,
        "training_initialization": "random; no pretrained weights; no resumed training",
    }
    if profile == "formal":
        return common | {
            "train_episode_counts": [96, 1536],
            "train_seed_base": 700000,
            "batch_size": 8,
            "optimizer_updates": 1536,
            "validation_episodes": 192,
            "validation_seed_base": 74611000,
            "validation_every_updates": 96,
            "challenge_episodes": 24,
            "test_seed_base": 95871000,
            "split_name": "fresh_test",
        }
    if profile == "smoke":
        return common | {
            "train_episode_counts": [8, 16],
            "train_seed_base": 81700000,
            "batch_size": 4,
            "optimizer_updates": 8,
            "validation_episodes": 8,
            "validation_seed_base": 81710000,
            "validation_every_updates": 2,
            "challenge_episodes": 2,
            "test_seed_base": 81720000,
            "split_name": "development",
        }
    raise ValueError("Unknown capacity study profile")


def cells(spec):
    return tuple((w, n, i) for w in WIDTHS for n in spec["train_episode_counts"] for i in INITIALIZATIONS)


def cell_name(width, episodes, initialization):
    return f"a{width}-data-{episodes}-init-{initialization}"


def epoch_budget(spec, episodes):
    if episodes % spec["batch_size"]:
        raise ValueError("Controlled data sizes must contain complete batches")
    updates_per_epoch = episodes // spec["batch_size"]
    if spec["optimizer_updates"] % updates_per_epoch:
        raise ValueError("Fixed update budget must end at an epoch boundary")
    return spec["optimizer_updates"] // updates_per_epoch


def registered_seeds(spec):
    train = set(range(spec["train_seed_base"], spec["train_seed_base"] + max(spec["train_episode_counts"])))
    val = set(range(spec["validation_seed_base"], spec["validation_seed_base"] + spec["validation_episodes"]))
    test = set(range(spec["test_seed_base"], spec["test_seed_base"] + spec["challenge_episodes"]))
    if train & val or train & test or val & test:
        raise ValueError("Training, validation and held-out trajectories must be disjoint")
    return {"train": train, "validation": val, "test": test}


def paired_summary(delta, indices):
    values = np.asarray(delta, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite, paired trajectory differences")
    return {
        "mean": float(values.mean()),
        "trajectory_bootstrap_95": np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist(),
        "negative_pairs": int((values < 0).sum()),
        "positive_pairs": int((values > 0).sum()),
        "tied_pairs": int((values == 0).sum()),
        "paired_trajectories": len(values),
    }


def factorial_contrasts(panel, spec):
    """Paired differences conditional on the two stated initializations, not seed-population CIs."""
    keys = cells(spec)
    if set(panel) != set(keys):
        raise ValueError("Missing or unexpected capacity/data/initialization cell")
    seeds = sorted(panel[keys[0]])
    if len(seeds) < 2 or any(set(p) != set(seeds) for p in panel.values()):
        raise ValueError("Missing or mismatched paired trajectories")
    values = {key: np.array([panel[key][s] for s in seeds], dtype=float) for key in keys}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Non-finite factorial metric")
    narrow, wide = WIDTHS
    small, large = spec["train_episode_counts"]
    differences = {}
    for n in (small, large):
        for i in INITIALIZATIONS:
            differences[f"wide_minus_narrow_at_data_{n}_init_{i}"] = values[wide, n, i] - values[narrow, n, i]
        differences[f"wide_minus_narrow_at_data_{n}_mean_over_inits"] = (
            sum(values[wide, n, i] - values[narrow, n, i] for i in INITIALIZATIONS) / 2
        )
    for w in WIDTHS:
        for i in INITIALIZATIONS:
            differences[f"large_minus_small_data_at_width_{w}_init_{i}"] = (
                values[w, large, i] - values[w, small, i]
            )
        differences[f"large_minus_small_data_at_width_{w}_mean_over_inits"] = (
            sum(values[w, large, i] - values[w, small, i] for i in INITIALIZATIONS) / 2
        )
        for n in (small, large):
            differences[f"init_11_minus_7_at_width_{w}_data_{n}"] = values[w, n, 11] - values[w, n, 7]
    differences["capacity_by_data_interaction_mean_over_inits"] = (
        sum(
            values[wide, large, i]
            - values[narrow, large, i]
            - values[wide, small, i]
            + values[narrow, small, i]
            for i in INITIALIZATIONS
        )
        / 2
    )
    indices = np.random.default_rng(20261104).integers(len(seeds), size=(4000, len(seeds)))
    return {name: paired_summary(delta, indices) for name, delta in differences.items()}


def recovery_and_forecast(estimated, forecast, truth, anchor):
    from recovery_challenge_core import TOLERANCE, score

    result = score(estimated, truth[:, :2], anchor)
    future = np.asarray(forecast)
    if future.shape != truth[:, :2].shape or not np.isfinite(future).all():
        raise ValueError("Missing or non-finite causal future forecast")
    delta = future[5:-3] - truth[8:, :2]
    result["three_step_forecast_rmse_m"] = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
    result["forecast_origin_ticks"] = [5, len(truth) - 4]
    result["forecast_horizon_s"] = 0.3
    confirmation = result["recovery_confirmed_s"]
    result["end_window_relapse"] = confirmation is not None and result["post_errors_m"][-1] > TOLERANCE
    return result


def run_baseline(estimator, cfg, deliveries, truth, anchor):
    estimates, forecasts = [], []
    start = time.perf_counter()
    for tick, ready in enumerate(deliveries):
        now = tick * cfg.dt
        estimator.ingest(ready, now)
        state = estimator.estimate(now)[0]
        estimates.append(state[:2].copy())
        # All existing KF/IMM modes have the same constant-velocity transition.
        # Project a COPY of the current mean; no future query mutates replay time.
        forecasts.append(state[:2] + 3 * cfg.dt * state[2:])
    result = recovery_and_forecast(np.asarray(estimates), np.asarray(forecasts), truth, anchor)
    result["baseline_replay_wall_s"] = time.perf_counter() - start
    result["gated_measurements"] = estimator.rejected_measurements
    return result


def summarize(rows, spec):
    names = [cell_name(*c) for c in cells(spec)]
    required = set(names) | {"kalman", "gated-imm"}
    expected_pairs = {
        (s, condition)
        for s in range(spec["test_seed_base"], spec["test_seed_base"] + spec["challenge_episodes"])
        for condition in CONDITIONS
    }
    actual_pairs = [(r["seed"], r["condition"]) for r in rows]
    if len(set(actual_pairs)) != len(actual_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Incomplete or repeated held-out cases")
    if any(set(r["scores"]) != required for r in rows):
        raise ValueError("Every case must report all eight models and both classical baselines")
    result = {}
    for condition in CONDITIONS:
        selected = sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["seed"])
        indices = np.random.default_rng(20261104).integers(len(selected), size=(4000, len(selected)))
        metrics, contrasts, comparisons = {}, {}, {}
        for metric in METRICS:
            arrays = {name: np.array([r["scores"][name][metric] for r in selected]) for name in required}
            if not np.isfinite(list(arrays.values())).all():
                raise ValueError("Non-finite test metrics cannot be silently excluded")
            panels = {
                key: {r["seed"]: r["scores"][cell_name(*key)][metric] for r in selected}
                for key in cells(spec)
            }
            contrasts[metric] = factorial_contrasts(panels, spec)
            comparisons[metric] = {
                name: {
                    baseline: paired_summary(arrays[name] - arrays[baseline], indices)
                    for baseline in ("kalman", "gated-imm")
                }
                for name in names
            }
            for w in WIDTHS:
                for n in spec["train_episode_counts"]:
                    alias = f"a{w}-data-{n}-mean-over-inits"
                    marginal = sum(arrays[cell_name(w, n, i)] for i in INITIALIZATIONS) / 2
                    comparisons[metric][alias] = {
                        baseline: paired_summary(marginal - arrays[baseline], indices)
                        for baseline in ("kalman", "gated-imm")
                    }
            metrics[metric] = {name: float(values.mean()) for name, values in sorted(arrays.items())}
        result[condition] = {
            "methods": {
                name: {
                    "mean_auc_m_s": metrics[METRICS[0]][name],
                    "mean_episode_forecast_rmse_m": metrics[METRICS[1]][name],
                    "unrecovered_by_1s": sum(int(r["scores"][name]["not_recovered_by_1s"]) for r in selected),
                    "end_window_relapses": sum(
                        int(r["scores"][name]["end_window_relapse"]) for r in selected
                    ),
                }
                for name in sorted(required)
            },
            "factorial_contrasts": contrasts,
            "paired_baseline_comparisons": comparisons,
        }
    return result


def validate_training_manifest(manifest, spec, key):
    width, episodes, initialization = key
    expected_updates = list(
        range(
            spec["validation_every_updates"], spec["optimizer_updates"] + 1, spec["validation_every_updates"]
        )
    )
    expected = {
        "parameters": PARAMETERS[width],
        "completed_epochs": epoch_budget(spec, episodes),
        "optimizer_updates": spec["optimizer_updates"],
        "validation_updates": expected_updates,
        "validation_opportunities": len(expected_updates),
        "validation_episodes": spec["validation_episodes"],
        "validation_every_updates": spec["validation_every_updates"],
        "resume_source": None,
        "train_episode_seeds": list(range(spec["train_seed_base"], spec["train_seed_base"] + episodes)),
        "validation_episode_seeds": list(
            range(spec["validation_seed_base"], spec["validation_seed_base"] + spec["validation_episodes"])
        ),
        "seed_plan": {
            "initialization": initialization,
            "train_seed_base": spec["train_seed_base"],
            "validation_seed_base": spec["validation_seed_base"],
            "batch_order_seed": ORDER_SEED,
        },
    }
    if (
        any(manifest.get(k) != v for k, v in expected.items())
        or manifest.get("best_update") not in expected_updates
    ):
        raise ValueError(f"Training evidence differs from the registered protocol: {cell_name(*key)}")


def run(out, device_name, profile):
    import torch
    from correction_diagnostics import replay_with_diagnostics
    from recovery_challenge_core import observed_trace

    from isac_wm.artifacts import model_fingerprint, sha256_file, write_json
    from isac_wm.baselines import ReplayKalman
    from isac_wm.imm import IMMConfig, ReplayIMM
    from isac_wm.simulator import SceneConfig
    from isac_wm.structured import StructuredConfig
    from isac_wm.training import load_model, resolve_device, runtime_info, train

    spec = design(profile)
    seed_sets = registered_seeds(spec)
    expected_device = "cuda" if profile == "formal" else "cpu"
    if device_name != expected_device:
        raise ValueError(f"{profile} requires --device {expected_device}")
    device = resolve_device(device_name)
    torch.set_num_threads(2)
    cfg = SceneConfig(steps=spec["steps"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "schema_version": 1,
        "profile": profile,
        "git_commit": revision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": runtime_info(device),
        "design": spec,
        "scene": cfg.to_dict(),
        "cell_order": [cell_name(*key) for key in cells(spec)],
        "validation_episode_seeds": sorted(seed_sets["validation"]),
        "test_episode_seeds": sorted(seed_sets["test"]),
        "interpretation": "Paired trajectory intervals are conditional on initializations 7 and 11; they do not estimate variation over arbitrary training seeds. Every outcome is retained.",
        "cells": {cell_name(*key): {"status": "not_started"} for key in cells(spec)},
        "results": {},
    }
    destination = out / "study.json"
    write_json(destination, report)
    wall_start = time.perf_counter()
    try:
        trained = {}
        starts = {}
        for key in cells(spec):
            width, episodes, initialization = key
            name = cell_name(*key)
            folder = out / "cells" / name
            stage = {
                "status": "running",
                "hidden_dim": width,
                "training_episodes": episodes,
                "initialization": initialization,
            }
            report["cells"][name] = stage
            write_json(destination, report)
            print(
                json.dumps({"stage": "training", "cell": name, "updates": spec["optimizer_updates"]}),
                flush=True,
            )
            started = time.perf_counter()
            manifest = train(
                folder,
                epochs=epoch_budget(spec, episodes),
                episodes=episodes,
                batch_size=spec["batch_size"],
                device_name=device_name,
                seed=initialization,
                scene=cfg,
                model_config=StructuredConfig(architecture="structured-attention", hidden_dim=width),
                learning_rate=spec["learning_rate"],
                train_seed_base=spec["train_seed_base"],
                validation_seed_base=spec["validation_seed_base"],
                batch_order_seed=ORDER_SEED,
                validation_episodes=spec["validation_episodes"],
                validation_every_updates=spec["validation_every_updates"],
            )
            validate_training_manifest(manifest, spec, key)
            if manifest["runtime"]["device"] != device_name:
                raise RuntimeError("Training device differs from the requested device")
            fingerprint = manifest["initial_parameters_sha256"]
            if starts.setdefault((width, initialization), fingerprint) != fingerprint:
                raise RuntimeError("Initial weights differed between nested data sizes")
            checkpoint = folder / "model.pt"
            checksum = sha256_file(checkpoint)
            if Path(str(checkpoint) + ".sha256").read_text().strip() != checksum:
                raise ValueError("Best model digest does not match its sidecar")
            stage.update(
                status="completed",
                wall_seconds=time.perf_counter() - started,
                epochs=manifest["completed_epochs"],
                parameters=manifest["parameters"],
                optimizer_updates=manifest["optimizer_updates"],
                validation_opportunities=manifest["validation_opportunities"],
                validation_updates=manifest["validation_updates"],
                best_update=manifest["best_update"],
                best_validation_rmse_m=manifest["best_validation_rmse_m"],
                initial_parameters_sha256=fingerprint,
                checkpoint=str(checkpoint.relative_to(out)),
                checkpoint_sha256=checksum,
                manifest_sha256=sha256_file(folder / "manifest.json"),
                history_sha256=sha256_file(folder / "history.json"),
                epoch_history_sha256=sha256_file(folder / "epoch_history.json"),
            )
            trained[key] = checkpoint
            write_json(destination, report)
        for width in WIDTHS:
            if starts[width, 7] == starts[width, 11]:
                raise RuntimeError("Different initialization settings produced identical models")
        # Freeze all eight validation-selected checkpoints BEFORE reading test outcomes.
        report["checkpoint_selection_frozen"] = True
        write_json(destination, report)
        models, fingerprints = {}, {}
        for key, checkpoint in trained.items():
            if sha256_file(checkpoint) != report["cells"][cell_name(*key)]["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed after training")
            model, scene, loaded_device = load_model(checkpoint, device_name)
            expected_provenance = (
                set(range(spec["train_seed_base"], spec["train_seed_base"] + key[1]))
                | seed_sets["validation"]
            )
            if scene != cfg or device != loaded_device or model.provenance_seeds != expected_provenance:
                raise ValueError("Loaded checkpoint provenance or geometry mismatch")
            if model.provenance_seeds & seed_sets["test"]:
                raise ValueError("Held-out episodes overlap checkpoint provenance")
            models[key], fingerprints[key] = model, model_fingerprint(model)
        evaluation = {
            "status": "running",
            "profile": profile,
            "git_commit": revision,
            "split": spec["split_name"],
            "rows": [],
            "summary": {},
        }
        diagnostic_records = {key: [] for key in cells(spec)}
        eval_path = out / f"{spec['split_name']}.json"
        for seed in sorted(seed_sets["test"]):
            for condition in CONDITIONS:
                snapshots, deliveries, truth, meta = observed_trace(seed, cfg, condition == "maneuver")
                anchor = meta["first_post_blackout_camera_delivery_tick"]
                scores = {}
                for key, model in models.items():
                    estimated, forecast, diagnostic = replay_with_diagnostics(
                        model, cfg, device, snapshots, deliveries, truth, anchor
                    )
                    scores[cell_name(*key)] = recovery_and_forecast(estimated, forecast, truth, anchor)
                    diagnostic_records[key].append(
                        {"seed": seed, "condition": condition, "report": diagnostic}
                    )
                scores["kalman"] = run_baseline(ReplayKalman(cfg.room_m), cfg, deliveries, truth, anchor)
                scores["gated-imm"] = run_baseline(
                    ReplayIMM(cfg.room_m, config=IMMConfig(gate_threshold=9.21)),
                    cfg,
                    deliveries,
                    truth,
                    anchor,
                )
                evaluation["rows"].append(
                    {"seed": seed, "condition": condition, "observations": meta, "scores": scores}
                )
                write_json(eval_path, evaluation)
            print(
                json.dumps(
                    {
                        "stage": "evaluation",
                        "completed_trajectory_pairs": len(evaluation["rows"]) // 2,
                        "total": spec["challenge_episodes"],
                    }
                ),
                flush=True,
            )
        evaluation["summary"] = summarize(evaluation["rows"], spec)
        evaluation["status"] = "completed"
        write_json(eval_path, evaluation)
        for key, model in models.items():
            name = cell_name(*key)
            if model_fingerprint(model) != fingerprints[key]:
                raise RuntimeError("Read-only evaluation changed model weights")
            write_json(
                out / "diagnostics" / f"{name}.json",
                {
                    "status": "completed",
                    "profile": profile,
                    "git_commit": revision,
                    "cell": name,
                    "checkpoint_sha256": report["cells"][name]["checkpoint_sha256"],
                    "episodes": diagnostic_records[key],
                },
            )
        report["results"][spec["split_name"]] = evaluation["summary"]
        report["status"] = "completed"
        report["wall_seconds"] = time.perf_counter() - wall_start
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        for stage in report["cells"].values():
            if stage["status"] == "running":
                stage.update(status="failed", error=report["error"])
        raise
    finally:
        write_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    report = run(args.out, args.device, args.profile)
    print(json.dumps({"status": report["status"], "evidence": str(args.out / "study.json")}), flush=True)


if __name__ == "__main__":
    main()
