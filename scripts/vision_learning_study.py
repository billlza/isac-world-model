"""Phase A: fixed-architecture learning-rate x direct visual-supervision pilot.

All eight cells share data, update/selection budgets and starting weights for a
given initialization. Only training and validation trajectories are read. A
fresh held-out test is deliberately not consumed by this diagnostic study.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNING_RATES = (0.001, 0.0003)
VISION_WEIGHTS = (0.0, 1.0)
INITIALIZATIONS = (7, 11)
PARAMETERS = 3_883_798
VISUAL_PARAMETERS = 1230
ORDER_SEED = 37
DIAGNOSTIC_UPDATES = (0, 1, 2, 4, 8, 16, 32, 64, 96, 192, 384)
MODEL = {
    "architecture": "structured-attention",
    "hidden_dim": 512,
    "feature_dim": 48,
    "memory_size": 16,
    "heads": 4,
    "room_m": 10.0,
}
SCENE = {
    "steps": 64,
    "dt": 0.1,
    "image_size": 24,
    "room_m": 10.0,
    "camera_period": 2,
    "camera_delay_s": 0.13,
    "delay_jitter_s": 0.04,
    "dropout": 0.05,
    "acceleration_std": 0.5,
    "snr_linear": 10.0,
    "bandwidth_hz": 1_000_000.0,
    "camera_dropout": None,
    "radio_dropout": None,
    "radio_outlier_probability": 0.0,
    "radio_outlier_scale_m": 4.0,
    "occlusion_fraction": 1 / 6,
    "radio_delay_extra_s": 0.0,
    "radio_range_bias_m": 0.0,
    "radio_angle_bias_rad": 0.0,
}
SUPERVISION_PROTOCOL = {
    "schema_version": 1,
    "target": "normalized xy at the accepted image event's capture time",
    "mask": "new and valid and visible_at_capture",
    "normalization": "mean squared error over coordinates, then visible captures",
    "empty_support": "zero loss and zero count",
    "online_input_keys_unchanged": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def design(profile):
    common = {
        "phase": "A",
        "learning_rates": list(LEARNING_RATES),
        "vision_loss_weights": list(VISION_WEIGHTS),
        "initializations": list(INITIALIZATIONS),
        "model": MODEL.copy(),
        "scene": SCENE.copy(),
        "parameters": PARAMETERS,
        "visual_encoder_parameters": VISUAL_PARAMETERS,
        "minibatch_order_seed": ORDER_SEED,
        "steps": 64,
        "dt_s": 0.1,
        "image_size": 24,
        "selection_metric": "validation_current_position_rmse_m_after_tick_4",
        "early_stopping_patience": 0,
        "training_initialization": "random; no pretrained weights; no resumed training",
        "visual_supervision": "masked mean squared capture-position error in normalized room coordinates",
        "weight_interpretation": "weight 1.0 is a preregistered diagnostic pilot, not an optimized weight",
        "test_policy": "training and validation only; no fresh test trajectories consumed",
    }
    if profile == "formal":
        spec = common | {
            "training_episodes": 1536,
            "train_seed_base": 700000,
            "batch_size": 8,
            "optimizer_updates": 384,
            "validation_episodes": 192,
            "validation_seed_base": 74611000,
            "validation_every_updates": 96,
        }
    elif profile == "smoke":
        spec = common | {
            "training_episodes": 8,
            "train_seed_base": 82800000,
            "batch_size": 4,
            "optimizer_updates": 8,
            "validation_episodes": 8,
            "validation_seed_base": 82810000,
            "validation_every_updates": 2,
        }
    else:
        raise ValueError("Unknown visual-learning study profile")
    return spec | {"diagnostic_updates": [u for u in DIAGNOSTIC_UPDATES if u <= spec["optimizer_updates"]]}


def cells(spec):
    return tuple(
        (lr, weight, seed)
        for lr in spec["learning_rates"]
        for weight in spec["vision_loss_weights"]
        for seed in spec["initializations"]
    )


def cell_name(learning_rate, weight, initialization):
    require(learning_rate in LEARNING_RATES and weight in VISION_WEIGHTS, "Unregistered study cell")
    require(initialization in INITIALIZATIONS, "Unregistered initialization")
    rate = "1e-3" if learning_rate == 0.001 else "3e-4"
    return f"a512-lr-{rate}-vision-{int(weight)}-init-{initialization}"


def epoch_budget(spec):
    episodes, batch = spec["training_episodes"], spec["batch_size"]
    require(episodes % batch == 0, "Training episodes must contain complete batches")
    per_epoch = episodes // batch
    require(spec["optimizer_updates"] % per_epoch == 0, "Update budget must end at an epoch boundary")
    return spec["optimizer_updates"] // per_epoch


def validation_updates(spec):
    return list(
        range(
            spec["validation_every_updates"], spec["optimizer_updates"] + 1, spec["validation_every_updates"]
        )
    )


def registered_seeds(spec):
    train = set(range(spec["train_seed_base"], spec["train_seed_base"] + spec["training_episodes"]))
    validation = set(
        range(spec["validation_seed_base"], spec["validation_seed_base"] + spec["validation_episodes"])
    )
    require(not train & validation, "Training and validation trajectories must be disjoint")
    return {"train": train, "validation": validation}


def finite_nonnegative(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def validate_training_manifest(manifest, spec, key, device_name):
    lr, weight, initialization = key
    updates = validation_updates(spec)
    pools = registered_seeds(spec)
    expected = {
        "seed": initialization,
        "episodes": spec["training_episodes"],
        "epochs": epoch_budget(spec),
        "completed_epochs": epoch_budget(spec),
        "batch_size": spec["batch_size"],
        "parameters": PARAMETERS,
        "visual_encoder_parameters": VISUAL_PARAMETERS,
        "model": MODEL,
        "learning_rate": lr,
        "vision_loss_weight": weight,
        "early_stopping_patience": 0,
        "optimizer_updates": spec["optimizer_updates"],
        "validation_updates": updates,
        "validation_opportunities": 4,
        "validation_episodes": spec["validation_episodes"],
        "validation_every_updates": spec["validation_every_updates"],
        "resume_source": None,
        "model_selection_status": "selected",
        "train_episode_seeds": sorted(pools["train"]),
        "validation_episode_seeds": sorted(pools["validation"]),
        "seed_plan": {
            "initialization": initialization,
            "train_seed_base": spec["train_seed_base"],
            "validation_seed_base": spec["validation_seed_base"],
            "batch_order_seed": ORDER_SEED,
        },
    }
    for name, value in expected.items():
        require(manifest.get(name) == value, f"Training protocol mismatch in {cell_name(*key)}: {name}")
    require(
        manifest.get("runtime", {}).get("device") == device_name,
        "Training device differs from the registered device",
    )
    require(manifest.get("scene") == spec["scene"], "Training geometry or cadence mismatch")
    require(
        manifest.get("best_update") in updates and finite_nonnegative(manifest.get("best_validation_rmse_m")),
        "Missing validation-selected checkpoint",
    )
    require(finite_nonnegative(manifest.get("training_seconds")), "Missing training duration")
    for name in ("initial_parameters_sha256", "initial_visual_parameters_sha256"):
        require(
            re.fullmatch(r"[0-9a-f]{64}", manifest.get(name, "")) is not None,
            f"Missing initial fingerprint: {name}",
        )
    require(
        manifest.get("vision_supervision_protocol") == SUPERVISION_PROTOCOL,
        "Capture-time supervision protocol mismatch",
    )
    counts = manifest.get("event_counts_seen", {})
    require(isinstance(counts, dict), "Missing actual training event counts")
    count_keys = (
        "snapshots",
        "vision_new_valid",
        "radio_new_valid",
        "evidence_update_ticks",
        "prediction_only_ticks",
        "vision_visible_captures",
    )
    require(
        set(counts) == set(count_keys) and all(type(counts[k]) is int and counts[k] >= 0 for k in count_keys),
        "Missing actual training event counts",
    )
    require(
        counts["snapshots"] == spec["optimizer_updates"] * spec["batch_size"] * spec["steps"],
        "Training event snapshot budget mismatch",
    )
    require(
        counts["evidence_update_ticks"] + counts["prediction_only_ticks"] == counts["snapshots"],
        "Training evidence/prediction tick counts do not sum to all snapshots",
    )
    require(
        counts["vision_visible_captures"] <= counts["vision_new_valid"] <= counts["snapshots"]
        and counts["radio_new_valid"] <= counts["snapshots"],
        "Invalid training event counts",
    )


def validate_training_evidence(manifest, history, epoch_history, diagnostics, spec, key, device_name):
    validate_training_manifest(manifest, spec, key, device_name)
    expected = validation_updates(spec)
    require(
        isinstance(history, list) and [r.get("optimizer_updates") for r in history] == expected,
        "Missing or repeated validation opportunity",
    )
    require(
        all(finite_nonnegative(r.get("validation_position_rmse_m")) for r in history),
        "Non-finite validation RMSE",
    )
    best = min(history, key=lambda r: r["validation_position_rmse_m"])
    require(
        best["optimizer_updates"] == manifest["best_update"]
        and best["validation_position_rmse_m"] == manifest["best_validation_rmse_m"],
        "Selected validation score differs from history",
    )
    epochs = epoch_budget(spec)
    per_epoch = spec["training_episodes"] // spec["batch_size"]
    require(
        isinstance(epoch_history, list)
        and [r.get("epoch") for r in epoch_history] == list(range(1, epochs + 1)),
        "Missing completed epoch history",
    )
    require(
        [r.get("optimizer_updates") for r in epoch_history] == [n * per_epoch for n in range(1, epochs + 1)],
        "Epoch update budget mismatch",
    )
    protocol = diagnostics.get("protocol", {})
    require(
        diagnostics.get("schema_version") == 1 and manifest.get("diagnostic_protocol") == protocol,
        "Diagnostic protocol differs from manifest",
    )
    require(
        protocol.get("updates") == spec["diagnostic_updates"], "Diagnostic updates differ from registration"
    )
    require(
        protocol.get("validation_episode_seeds") == sorted(registered_seeds(spec)["validation"])[:4],
        "Diagnostic subset differs from registered validation trajectories",
    )
    require(
        protocol.get("max_validation_episodes") == 4
        and protocol.get("scope") == "diagnostic-only; not checkpoint selection",
        "Diagnostic subset must not change checkpoint selection",
    )
    require(
        protocol.get("validation_steps") == spec["steps"] and protocol.get("saturation_threshold") == 0.99,
        "Diagnostic time or saturation definition mismatch",
    )
    snapshots = diagnostics.get("snapshots", [])
    require(
        [r.get("optimizer_updates") for r in snapshots] == spec["diagnostic_updates"],
        "Missing or repeated diagnostic snapshot",
    )
    for snapshot in snapshots:
        validation = snapshot.get("validation", {})
        require(
            finite_nonnegative(validation.get("visual_position_rmse_m")),
            "Missing visible-capture visual diagnostic",
        )
        require(
            isinstance(validation.get("visual_visible_capture_count"), int)
            and validation["visual_visible_capture_count"] > 0,
            "No visible captures in diagnostic subset",
        )
        require(isinstance(validation.get("event_counts"), dict), "Missing diagnostic event cadence")
    return {
        "best_update": manifest["best_update"],
        "best_validation_rmse_m": manifest["best_validation_rmse_m"],
        "last_validation_rmse_m": history[-1]["validation_position_rmse_m"],
        "initial_visual_position_rmse_m": snapshots[0]["validation"]["visual_position_rmse_m"],
        "last_visual_position_rmse_m": snapshots[-1]["validation"]["visual_position_rmse_m"],
        "visual_diagnostic_validation_seeds": protocol["validation_episode_seeds"],
        "diagnostic_updates": spec["diagnostic_updates"],
        "event_counts_seen": manifest["event_counts_seen"],
        "diagnostic_event_counts": snapshots[-1]["validation"]["event_counts"],
        "initial_visual_diagnostics": snapshots[0]["validation"],
        "last_visual_diagnostics": snapshots[-1]["validation"],
        "last_training_diagnostics": snapshots[-1].get("training"),
        "training_seconds": manifest["training_seconds"],
        "wall_average_optimizer_updates_per_s": spec["optimizer_updates"]
        / max(manifest["training_seconds"], 1e-12),
        "timing_scope": "training wall time including validation/diagnostic overhead; not online decision latency",
    }


def summarize(cell_reports, spec):
    keys = cells(spec)
    require(
        set(cell_reports) == {cell_name(*k) for k in keys}, "Every registered study cell must be retained"
    )
    require(
        all(cell_reports[cell_name(*k)].get("status") == "completed" for k in keys),
        "Cannot summarize incomplete training cells",
    )
    metrics = ("best_validation_rmse_m", "last_validation_rmse_m", "last_visual_position_rmse_m")
    result = {"cells": {cell_name(*k): cell_reports[cell_name(*k)]["metrics"] for k in keys}, "contrasts": {}}
    for metric in metrics:
        panel = {k: cell_reports[cell_name(*k)]["metrics"][metric] for k in keys}
        require(all(finite_nonnegative(v) for v in panel.values()), "Non-finite cell metric")
        differences = {}
        for lr in LEARNING_RATES:
            for seed in INITIALIZATIONS:
                differences[f"vision_1_minus_0_lr_{lr:g}_init_{seed}"] = (
                    panel[lr, 1.0, seed] - panel[lr, 0.0, seed]
                )
            differences[f"vision_1_minus_0_lr_{lr:g}_mean_over_inits"] = (
                sum(panel[lr, 1.0, i] - panel[lr, 0.0, i] for i in INITIALIZATIONS) / 2
            )
        for weight in VISION_WEIGHTS:
            for seed in INITIALIZATIONS:
                differences[f"lr_3e-4_minus_1e-3_vision_{int(weight)}_init_{seed}"] = (
                    panel[0.0003, weight, seed] - panel[0.001, weight, seed]
                )
        differences["learning_rate_by_supervision_interaction_mean_over_inits"] = (
            sum(
                panel[0.0003, 1.0, i] - panel[0.0003, 0.0, i] - panel[0.001, 1.0, i] + panel[0.001, 0.0, i]
                for i in INITIALIZATIONS
            )
            / 2
        )
        result["contrasts"][metric] = differences
    result["interpretation"] = (
        "Descriptive validation contrasts conditional on initializations 7 and 11; negative means lower error. No population confidence interval, fresh-test claim or automatic causal diagnosis."
    )
    return result


def run(out, device_name, profile):
    import torch

    from isac_wm.artifacts import sha256_file, write_json
    from isac_wm.simulator import SceneConfig
    from isac_wm.structured import StructuredConfig
    from isac_wm.training import resolve_device, runtime_info, train

    spec = design(profile)
    expected_device = "cuda" if profile == "formal" else "cpu"
    require(device_name == expected_device, f"{profile} requires --device {expected_device}")
    device = resolve_device(device_name)
    torch.set_num_threads(2)
    cfg = SceneConfig(**spec["scene"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if profile == "formal":
        require(
            revision == os.environ.get("ISAC_VISION_PIN")
            and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) is not None,
            "Formal study requires the pinned Hamilton launcher",
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "profile": profile,
        "git_commit": revision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": runtime_info(device),
        "design": spec,
        "scene": cfg.to_dict(),
        "fresh_test_consumed": False,
        "test_episode_seeds": [],
        "cell_order": [cell_name(*k) for k in cells(spec)],
        "cells": {cell_name(*k): {"status": "not_started"} for k in cells(spec)},
    }
    destination = out / "study.json"
    write_json(destination, report)
    started = time.perf_counter()
    starts, visual_starts, event_counts = {}, {}, None
    try:
        for key in cells(spec):
            lr, weight, initialization = key
            name = cell_name(*key)
            folder = out / "cells" / name
            stage = {
                "status": "running",
                "learning_rate": lr,
                "vision_loss_weight": weight,
                "initialization": initialization,
            }
            report["cells"][name] = stage
            write_json(destination, report)
            print(
                json.dumps(
                    {"stage": "training", "cell": name, "optimizer_updates": spec["optimizer_updates"]}
                ),
                flush=True,
            )
            cell_started = time.perf_counter()
            manifest = train(
                folder,
                epochs=epoch_budget(spec),
                episodes=spec["training_episodes"],
                batch_size=spec["batch_size"],
                device_name=device_name,
                seed=initialization,
                scene=cfg,
                model_config=StructuredConfig(**MODEL),
                learning_rate=lr,
                train_seed_base=spec["train_seed_base"],
                validation_seed_base=spec["validation_seed_base"],
                batch_order_seed=ORDER_SEED,
                validation_episodes=spec["validation_episodes"],
                validation_every_updates=spec["validation_every_updates"],
                vision_loss_weight=weight,
                diagnostic_updates=tuple(spec["diagnostic_updates"]),
            )
            evidence = {
                n: json.loads((folder / f"{n}.json").read_text())
                for n in ("history", "epoch_history", "visual_diagnostics")
            }
            metrics = validate_training_evidence(
                manifest,
                evidence["history"],
                evidence["epoch_history"],
                evidence["visual_diagnostics"],
                spec,
                key,
                device_name,
            )
            for cache, field in (
                (starts, "initial_parameters_sha256"),
                (visual_starts, "initial_visual_parameters_sha256"),
            ):
                require(
                    cache.setdefault(initialization, manifest[field]) == manifest[field],
                    "Initial weights differ across learning-rate/supervision cells",
                )
            if event_counts is None:
                event_counts = metrics["event_counts_seen"]
            require(
                event_counts == metrics["event_counts_seen"],
                "Training event frequency differs across controlled cells",
            )
            checkpoint = folder / "model.pt"
            checksum = sha256_file(checkpoint)
            require(
                Path(str(checkpoint) + ".sha256").read_text().strip() == checksum,
                "Best checkpoint digest differs from sidecar",
            )
            stage.update(
                status="completed",
                wall_seconds=time.perf_counter() - cell_started,
                parameters=manifest["parameters"],
                visual_encoder_parameters=manifest["visual_encoder_parameters"],
                initial_parameters_sha256=manifest["initial_parameters_sha256"],
                initial_visual_parameters_sha256=manifest["initial_visual_parameters_sha256"],
                checkpoint=str(checkpoint.relative_to(out)),
                checkpoint_sha256=checksum,
                metrics=metrics,
                **{
                    f"{n}_sha256": sha256_file(folder / f"{n}.json")
                    for n in ("manifest", "history", "epoch_history", "visual_diagnostics")
                },
            )
            write_json(destination, report)
        require(
            starts[7] != starts[11] and visual_starts[7] != visual_starts[11],
            "Different initializations produced identical fingerprints",
        )
        report["summary"] = summarize(report["cells"], spec)
        report["status"] = "completed"
        report["wall_seconds"] = time.perf_counter() - started
        write_json(destination, report)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "phase": "A",
                    "fresh_test_consumed": False,
                    "evidence": str(destination),
                }
            ),
            flush=True,
        )
        return report
    except Exception as failure:
        report["status"] = "failed"
        report["error"] = {"type": type(failure).__name__, "message": str(failure)}
        report["wall_seconds"] = time.perf_counter() - started
        write_json(destination, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--profile", required=True, choices=("smoke", "formal"))
    args = parser.parse_args()
    run(args.out, args.device, args.profile)


if __name__ == "__main__":
    main()
