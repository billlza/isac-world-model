"""Paired CPU pilot of frozen-latent versus causal latent-rollout forecasts.

This is an RF-only, position-input toy study. It does not measure beam capture,
communication throughput, outdoor generalization, or calibrated safety.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from isac_wm.artifacts import model_fingerprint, save_checkpoint, sha256_file, write_json
from isac_wm.baselines import ReplayKalman
from isac_wm.gap_recovery import GapRecoverySpec, generate_gap_episode
from isac_wm.imm import IMMConfig, ReplayIMM
from isac_wm.pretraining_objective import pretraining_loss
from isac_wm.simulator import SceneConfig
from isac_wm.structured import StructuredConfig, build_model
from isac_wm.training import INPUT_KEYS, batchify, runtime_info, seed_all

BASE_COMMIT = "9cf8304c7f66c60cf29c889f7b1b98d7627cd293"
PROFILES = tuple((accel, gap) for accel in (0.5, 2.0) for gap in (3, 6, 10, 20))
CHI2_90 = -2.0 * math.log(0.1)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def plan(count, split):
    offset = {"train": 0, "validation": 1_000_000, "test": 2_000_000}[split]
    return [{"seed": 710_000_000 + offset + i, "profile": i % len(PROFILES)} for i in range(count)]


def corpus(rows):
    result = []
    for row in rows:
        acceleration, gap_steps = PROFILES[row["profile"]]
        scene = SceneConfig(
            steps=40,
            dt=0.1,
            camera_dropout=1.0,
            radio_dropout=0.0,
            dropout=0.0,
            delay_jitter_s=0.0,
            acceleration_std=acceleration,
        )
        episode = generate_gap_episode(row["seed"], scene, GapRecoverySpec(gap_steps=gap_steps))
        result.append((row, scene, episode))
    return result


def anchors(data):
    indices, targets, horizons = [], [], []
    for row, scene, episode in data:
        arrays = episode.arrays
        history = arrays["phase_history"].astype(bool) & arrays["new"][:, 1].astype(bool)
        where = np.flatnonzero(history)
        if not where.size:
            raise ValueError("Episode lacks a delivered RF history anchor")
        anchor = int(where[-1])
        gap_steps = PROFILES[row["profile"]][1]
        target = anchor + gap_steps
        if target >= scene.steps or not arrays["phase_gap"][target]:
            raise ValueError("Fixed-horizon endpoint must be inside the observed no-new-RF gap")
        indices.append(anchor)
        targets.append(target)
        horizons.append(gap_steps * scene.dt)
    return np.asarray(indices), np.asarray(targets), np.asarray(horizons, np.float32)


def distribution_metrics(error, covariance):
    if not error.size:
        raise ValueError("Empty metric support")
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0) or not np.isfinite(covariance).all():
        raise FloatingPointError("Non-positive predictive covariance")
    mahal = np.einsum("...i,...i->...", error, np.linalg.solve(covariance, error[..., None])[..., 0])
    return {
        "count": int(error.shape[0]),
        "mse_m2": float(np.mean(np.sum(error**2, axis=-1))),
        "position_nll": float(np.mean(math.log(2 * math.pi) + 0.5 * (logdet + mahal))),
        "ellipse90_coverage": float(np.mean(mahal <= CHI2_90)),
        "ellipse90_area_m2": float(np.mean(math.pi * CHI2_90 * np.exp(0.5 * logdet))),
    }


def score_episode(item, mean, covariance, future_mean, future_covariance, target_index):
    row, scene, episode = item
    arrays = episode.arrays
    truth = arrays["state_target"][:, :2] * scene.room_m
    error = mean[:, :2] - truth
    count = arrays["recovery_packet_count"]
    masks = {
        "history": arrays["phase_history"].astype(bool) & (np.arange(scene.steps) >= 5),
        "gap": arrays["phase_gap"].astype(bool),
        "recovery_first5": arrays["phase_recovery"].astype(bool) & (count >= 1) & (count <= 5),
    }
    masks.update(
        {
            f"recovery_packet{n}": arrays["phase_recovery"].astype(bool)
            & (count == n)
            & arrays["new"][:, 1].astype(bool)
            for n in (1, 3, 5)
        }
    )
    values = {
        name: distribution_metrics(error[mask], covariance[mask, :2, :2]) for name, mask in masks.items()
    }
    values["anchor_forecast"] = distribution_metrics(
        (future_mean[:2] - truth[target_index])[None], future_covariance[None, :2, :2]
    )
    return {**row, "metrics": values}


def aggregate(rows):
    by_profile = {}
    for profile in range(len(PROFILES)):
        selected = [r for r in rows if r["profile"] == profile]
        if not selected:
            raise ValueError("Every evaluation profile needs support")
        metrics = {}
        for phase in selected[0]["metrics"]:
            records = [r["metrics"][phase] for r in selected]
            weights = np.asarray([r["count"] for r in records])
            metrics[phase] = {
                key: float(np.average([r[key] for r in records], weights=weights))
                for key in ("mse_m2", "position_nll", "ellipse90_coverage", "ellipse90_area_m2")
            }
            metrics[phase]["rmse_m"] = math.sqrt(metrics[phase]["mse_m2"])
            metrics[phase]["count"] = int(weights.sum())
        by_profile[str(profile)] = metrics
    macro = {
        phase: {
            key: float(np.mean([p[phase][key] for p in by_profile.values()]))
            for key in ("rmse_m", "position_nll", "ellipse90_coverage", "ellipse90_area_m2")
        }
        for phase in rows[0]["metrics"]
    }
    score = float(np.mean([macro[p]["rmse_m"] for p in ("gap", "recovery_first5", "anchor_forecast")]))
    return {"selection_score_m": score, "macro": macro, "profiles": by_profile, "episodes": rows}


def evaluate_model(model, data, batch_size):
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            items = data[start : start + batch_size]
            batch = batchify([e.arrays for _, _, e in items], torch.device("cpu"))
            anchor, target, horizon = anchors(items)
            memory = model.initial(len(items), "cpu")
            means, covariances, anchor_states = [], [], [None] * len(items)
            for tick in range(items[0][1].steps):
                out, memory = model.step({key: batch[key][:, tick] for key in INPUT_KEYS}, memory)
                means.append(out["mean"].numpy())
                covariances.append(out["covariance"].numpy())
                for j in np.flatnonzero(anchor == tick):
                    anchor_states[j] = out["hidden"][j : j + 1].clone()
            forecast = model.forecast(torch.cat(anchor_states), torch.from_numpy(horizon[:, None]))
            means = np.stack(means, 1)
            covariances = np.stack(covariances, 1)
            for j, item in enumerate(items):
                scale = item[1].state_scale
                rows.append(
                    score_episode(
                        item,
                        means[j] * scale,
                        covariances[j] * scale[:, None] * scale[None, :],
                        forecast["mean"][j].numpy() * scale,
                        forecast["covariance"][j].numpy() * scale[:, None] * scale[None, :],
                        int(target[j]),
                    )
                )
    return aggregate(rows)


def evaluate_baseline(factory, data):
    rows = []
    for item in data:
        _, scene, episode = item
        anchor, target, horizon = anchors([item])
        estimator = factory(scene.room_m)
        means, covariances = [], []
        future_mean = future_covariance = None
        for tick, accepted in enumerate(episode.delivered):
            now = tick * scene.dt
            estimator.ingest(list(accepted), now)
            mean, covariance = estimator.estimate(now)
            means.append(mean)
            covariances.append(covariance)
            if tick == anchor[0]:
                future_mean, future_covariance = copy.deepcopy(estimator).estimate(now + float(horizon[0]))
        rows.append(
            score_episode(
                item,
                np.stack(means),
                np.stack(covariances),
                future_mean,
                future_covariance,
                int(target[0]),
            )
        )
    return aggregate(rows)


def fingerprint_data(data):
    h = hashlib.sha256()
    for row, scene, episode in data:
        h.update(canonical({"row": row, "scene": scene.to_dict(), "metadata": episode.metadata}).encode())
        for key, value in sorted(episode.arrays.items()):
            h.update(key.encode())
            h.update(str((value.dtype, value.shape)).encode())
            h.update(value.tobytes())
    return h.hexdigest()


def train_cell(args, variant, seed, datasets, run_dir, source):
    seed_all(seed)
    config = StructuredConfig(
        latent_state_norm="layernorm", forecast_step_s=0.1 if variant == "rollout" else 0.0
    )
    model = build_model(config)
    initial = model_fingerprint(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    order_rng = np.random.default_rng(117)
    history, best, selected_epoch, updates = [], math.inf, None, 0
    started = time.perf_counter()
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = order_rng.permutation(len(datasets["train"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = batchify(
                [datasets["train"][i][2].arrays for i in order[start : start + args.batch_size]],
                torch.device("cpu"),
            )
            horizon = (3, 6, 10, 20)[updates % 4]
            output = model(batch, forecast_s=horizon * 0.1, batched_encoding=True)
            loss, details = pretraining_loss(
                output, batch, horizon, vision_weight=0, observation_weight=0, process_weight=0
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append({**details, "gradient_norm_before_clip": float(gradient_norm)})
            updates += 1
        validation = evaluate_model(model, datasets["validation"], args.batch_size)
        score = validation["selection_score_m"]
        if score < best:
            best, selected_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "updates": updates, "training": losses, "validation": validation})
        write_json(run_dir / "history.json", history)
        print(f"{variant} seed={seed} epoch={epoch}/{args.epochs} validation={score:.6f}m", flush=True)
    model.load_state_dict(best_state)
    result = {
        "variant": variant,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "updates": updates,
        "validation_score_m": best,
        "initial_weights_sha256": initial,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_config": config.to_dict(),
        "elapsed_training_validation_s": time.perf_counter() - started,
        "test": evaluate_model(model, datasets["test"], args.batch_size),
    }
    save_checkpoint(
        run_dir / "selected.pt",
        {
            "schema_version": 1,
            "state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "scene_config": datasets["train"][0][1].to_dict(),
            "study": "gap-recovery-pilot",
            "seed": seed,
            "source": source,
            "selected_epoch": selected_epoch,
        },
    )
    result["selected_checkpoint_sha256"] = sha256_file(run_dir / "selected.pt")
    write_json(run_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-per-profile", type=int, default=4)
    parser.add_argument("--test-per-profile", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11])
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.validation_per_profile, args.test_per_profile) < 1:
        parser.error("Training and evaluation counts must be positive")
    if args.episodes < len(PROFILES) or args.episodes > 10000 or args.episodes % len(PROFILES):
        parser.error("episodes must be a multiple of eight in [8,10000]")
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 or s > 99 for s in args.seeds):
        parser.error("seeds must be unique integers in [0,99]")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Choose a new empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/isac_wm").glob("*.py")) + [Path(__file__).resolve()]
    source = {
        "base_commit": BASE_COMMIT,
        "files_sha256": {str(p.relative_to(root)): sha256_file(p) for p in paths},
    }
    plans = {
        "train": plan(args.episodes, "train"),
        "validation": plan(len(PROFILES) * args.validation_per_profile, "validation"),
        "test": plan(len(PROFILES) * args.test_per_profile, "test"),
    }
    datasets = {name: corpus(rows) for name, rows in plans.items()}
    manifest = {
        "status": "running",
        "source": source,
        "runtime": runtime_info(torch.device("cpu")),
        "arguments": {**vars(args), "out": str(args.out)},
        "plans": plans,
        "data_sha256": {name: fingerprint_data(data) for name, data in datasets.items()},
        "profiles": [{"acceleration_std": a, "gap_steps": g} for a, g in PROFILES],
        "gap_protocol": asdict(GapRecoverySpec()),
        "selection": "Mean of profile-macro gap RMSE, first-five-recovery RMSE and anchor-forecast endpoint RMSE on validation only",
        "loss": "state NLL + 0.5 future NLL; horizon cycles 3/6/10/20 ticks per update; no vision/observation/process auxiliary",
        "uncertainty": "Uncalibrated learned 2D position covariance; ellipse coverage is empirical, pooled by profile",
        "scope": "Single-target 2D bounded toy motion; fixed quick measurements; same accepted range/bearing packets for every estimator; radial velocity masked for all",
    }
    write_json(args.out / "manifest.json", manifest)
    results = {"cells": [], "baselines": {}}
    try:
        for seed in args.seeds:
            for variant in ("frozen", "rollout"):
                run_dir = args.out / f"{variant}-seed{seed}"
                run_dir.mkdir()
                results["cells"].append(train_cell(args, variant, seed, datasets, run_dir, source))
                write_json(args.out / "results.json", results)
            pair = results["cells"][-2:]
            if pair[0]["initial_weights_sha256"] != pair[1]["initial_weights_sha256"]:
                raise RuntimeError("Paired variants did not start from identical weights")
        for kind, options in (("kalman", (0.05, 0.2, 0.8, 2.0, 6.0)), ("imm", (0.25, 1.0, 4.0))):
            grid = []
            for value in options:
                factory = (
                    (lambda room, q=value: ReplayKalman(room, process_noise=q))
                    if kind == "kalman"
                    else (
                        lambda room, scale=value: ReplayIMM(
                            room, config=IMMConfig(process_noises=tuple(scale * q for q in (0.05, 0.8, 6.0)))
                        )
                    )
                )
                validation = evaluate_baseline(factory, datasets["validation"])
                grid.append({"value": value, "validation": validation})
            chosen = min(grid, key=lambda x: x["validation"]["selection_score_m"])["value"]
            factory = (
                (lambda room: ReplayKalman(room, process_noise=chosen))
                if kind == "kalman"
                else (
                    lambda room: ReplayIMM(
                        room, config=IMMConfig(process_noises=tuple(chosen * q for q in (0.05, 0.8, 6.0)))
                    )
                )
            )
            results["baselines"][kind] = {
                "selected_value": chosen,
                "validation_grid": grid,
                "test": evaluate_baseline(factory, datasets["test"]),
            }
            write_json(args.out / "results.json", results)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(args.out / "manifest.json", manifest)
    print(
        f"Completed {len(results['cells'])} training cells and both validation-selected baselines: {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
