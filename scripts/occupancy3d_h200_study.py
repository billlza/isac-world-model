"""H200 execution variant of the frozen occupancy3d_study.py scientific protocol.

Scientific training/evaluation logic is copied unchanged. Explicit differences:
CUDA defaults; strict or warning determinism; actual CUDA network and RF backward
preflight; original and adapter source hashes recorded together. The warning mode
permits nondeterministic AvgPool3d and trilinear CUDA backward, and is not a promise
of bitwise replay. This runner implements supervised geometry pretraining and
supervised RF consistency continuation, not RL or RLHF. No device fallback.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from isac_wm.occupancy3d_data import GridSpec, RFConfig, generate_scenes, make_operator, observe_scenes
from isac_wm.occupancy3d_evaluation import (
    complex_nmse,
    fista_box_inverse,
    fixed_height_25d,
    matched_filter,
    occupancy_metrics,
    select_fista_configuration,
    select_fixed_height_configuration,
    select_occupancy_threshold,
)
from isac_wm.occupancy3d_model import (
    OccupancyReadout,
    geometry_loss,
    prefix_features,
    rf_consistency_loss,
    weight_fingerprint,
)

LAYOUTS = ("same_height", "height_diverse")
PREFIXES = (3, 6, 12)
SOURCE_BASE = "90e278bffd7af659761404ca5c1c10bb3db4f09d"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts/occupancy3d_study.py",
        *sorted((root / "src/isac_wm").glob("occupancy3d_*.py")),
    ]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def predict(
    model: OccupancyReadout, features: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start : start + batch_size]).to(device)
            values.append(model(x).sigmoid().cpu().numpy())
    return np.concatenate(values)


def validation_loss(
    model: OccupancyReadout, features: np.ndarray, target: np.ndarray, device: torch.device, batch_size: int
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start : start + batch_size]).to(device)
            truth = torch.from_numpy(target[start : start + batch_size]).to(device)
            total += float(geometry_loss(model(x), truth)) * len(x)
    return total / len(features)


def train_stage(
    model: OccupancyReadout,
    *,
    train_features: dict[int, np.ndarray],
    val_features: dict[int, np.ndarray],
    train_truth: np.ndarray,
    val_truth: np.ndarray,
    train_y: np.ndarray,
    operator: np.ndarray,
    n_frequencies: int,
    epochs: int,
    seed: int,
    rf_weight: float,
    learning_rate: float,
    args: argparse.Namespace,
    name: str,
    device: torch.device,
) -> tuple[OccupancyReadout, dict]:
    """All selection uses validation geometry; test data is not accepted here."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    order_rng = np.random.default_rng(seed)
    a = torch.from_numpy(operator).to(device)
    truth = torch.from_numpy(train_truth).to(device)
    y = torch.from_numpy(train_y).to(device)
    features = {key: torch.from_numpy(value).to(device) for key, value in train_features.items()}
    initial_hash = weight_fingerprint(model)
    start_time = time.perf_counter()
    history = []
    initial_val = validation_loss(model, val_features[12], val_truth, device, args.batch_size)
    best_loss, best_epoch = initial_val, 0
    best_state = copy.deepcopy(model.state_dict())
    schedule_hash = hashlib.sha256()
    updates = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(len(train_truth))
        running_geometry, running_rf, seen = 0.0, 0.0, 0
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset : offset + args.batch_size]
            prefix = int(order_rng.choice(PREFIXES))
            schedule_hash.update(indices.astype(np.int64).tobytes())
            schedule_hash.update(np.asarray([prefix], np.int64).tobytes())
            index = torch.as_tensor(indices, dtype=torch.long, device=device)
            logits = model(features[prefix][index])
            geometric = geometry_loss(logits, truth[index])
            rows = prefix * n_frequencies
            physical = rf_consistency_loss(logits.sigmoid(), y[index, :rows], a[:rows])
            objective = geometric + rf_weight * physical
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(f"Non-finite objective in {name}")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(f"Non-finite gradient in {name}")
            optimizer.step()
            running_geometry += float(geometric.detach()) * len(indices)
            running_rf += float(physical.detach()) * len(indices)
            seen += len(indices)
            updates += 1
        val = validation_loss(model, val_features[12], val_truth, device, args.batch_size)
        if val < best_loss:
            best_loss, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            "train_geometry_loss": running_geometry / seen,
            "train_observed_rf_nmse": running_rf / seen,
            "validation_geometry_loss": val,
            "elapsed_s": time.perf_counter() - start_time,
        }
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            print(json.dumps({"stage": name, **row}), flush=True)
    model.load_state_dict(best_state)
    selected_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = args.output / "checkpoints" / f"{name}.pt"
    torch.save(
        {
            "state_dict": selected_state,
            "channels": args.channels,
            "initialization": "random",
            "selected_epoch": best_epoch,
            "selection": "minimum full-prefix validation geometry loss",
            "source_sha256": source_hashes(),
        },
        checkpoint,
    )
    record = {
        "name": name,
        "completed": True,
        "rf_weight": rf_weight,
        "epochs_executed": epochs,
        "optimizer_updates": updates,
        "optimizer": "AdamW reset at stage start",
        "learning_rate": learning_rate,
        "batch_schedule_sha256": schedule_hash.hexdigest(),
        "initial_weight_sha256": initial_hash,
        "selected_weight_sha256": weight_fingerprint(model),
        "selected_epoch": best_epoch,
        "initial_validation_geometry_loss": initial_val,
        "selected_validation_geometry_loss": best_loss,
        "elapsed_s": time.perf_counter() - start_time,
        "history": history,
        "checkpoint": str(checkpoint.relative_to(args.output)),
    }
    write_json(args.output / f"{name}.json", record)
    return model, record


def evaluate_field(
    prob: np.ndarray,
    *,
    truth: np.ndarray,
    observations: np.ndarray,
    operator: np.ndarray,
    query_operator: np.ndarray,
    query_clean: np.ndarray,
    query_noisy: np.ndarray,
    grid: GridSpec,
    threshold: float,
    scene_hashes: tuple[str, ...],
    pair_ids: tuple[str, ...],
) -> list[dict]:
    predicted_query = prob.reshape(len(prob), -1) @ query_operator.T
    hard_query = (prob >= threshold).reshape(len(prob), -1).astype(np.float32) @ query_operator.T
    predicted_observed = prob.reshape(len(prob), -1) @ operator.T
    query_nmse = complex_nmse(predicted_query, query_clean)
    query_noisy_nmse = complex_nmse(predicted_query, query_noisy)
    hard_nmse = complex_nmse(hard_query, query_clean)
    obs_nmse = complex_nmse(predicted_observed, observations)
    records = []
    for i in range(len(prob)):
        records.append(
            {
                "scene_hash": scene_hashes[i],
                "family_id": pair_ids[i],
                **occupancy_metrics(prob[i], truth[i], grid.spacing_m, threshold=threshold),
                "query_rf_nmse_soft": float(query_nmse[i]),
                "query_rf_nmse_hard": float(hard_nmse[i]),
                "query_noisy_rf_nmse_soft": float(query_noisy_nmse[i]),
                "observed_rf_nmse_soft": float(obs_nmse[i]),
            }
        )
    return records


def macro_metrics(records: list[dict]) -> dict:
    keys = (
        "iou_3d",
        "precision",
        "recall",
        "brier",
        "binary_nll",
        "ece",
        "height_column_rmse_m",
        "height_column_mae_m",
        "surface_chamfer_m",
        "surface_f1",
        "empty_predicted_surface",
        "missed_columns",
        "false_positive_columns",
        "query_rf_nmse_soft",
        "query_rf_nmse_hard",
        "query_noisy_rf_nmse_soft",
        "observed_rf_nmse_soft",
    )
    result = {}
    for key in keys:
        values = [row[key] for row in records if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        if len(values) != len(records):
            result[key + "_defined_scenes"] = len(values)
    return result


def load_prior_scene_hashes(path: Path | None) -> tuple[set[str], dict]:
    """Exclude prior experiments from every new split, preserving fresh test scenes."""
    if path is None:
        return set(), {"path": None, "sha256": None, "scene_count": 0}
    content = path.read_bytes()
    data = json.loads(content)
    values = data.get("scene_hashes")
    if not isinstance(values, list) or any(
        not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in values
    ):
        raise ValueError("Prior scene hashes must be a list of SHA256 strings")
    if len(set(values)) != len(values) or data.get("scene_count") != len(values):
        raise ValueError("Prior scene hash count or uniqueness is invalid")
    return set(values), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "scene_count": len(values),
        "policy": "exclude_from_all_new_splits",
    }


def cuda_training_preflight(channels: int, rf_weight: float) -> dict:
    """Exercise the actual model, geometry loss and complex RF gradient on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no device fallback")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before starting CUDA training")
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(101)
        model = OccupancyReadout(channels).cuda()
        before = weight_fingerprint(model)
        features = torch.randn(2, 7, 12, 12, 8, device="cuda")
        truth = (torch.rand(2, 12, 12, 8, device="cuda") > 0.8).float()
        matrix = torch.randn(16, 1152, dtype=torch.complex64, device="cuda")
        observed = truth.flatten(1).to(torch.complex64) @ matrix.T
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            logits = model(features)
            loss = geometry_loss(logits, truth) + rf_weight * rf_consistency_loss(
                logits.sigmoid(), observed, matrix
            )
            loss.backward()
            if not torch.isfinite(loss) or not all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise RuntimeError("Non-finite or missing gradient in actual CUDA model preflight")
            optimizer.step()
            torch.cuda.synchronize()
        if weight_fingerprint(model) == before:
            raise RuntimeError("CUDA preflight did not update model parameters")
        messages = [str(item.message) for item in captured]
        for message in messages:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return {
            "status": "passed",
            "device": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "actual_model_backward": True,
            "complex_rf_backward": True,
            "loss": float(loss.detach()),
            "warnings": messages,
            "parameter_update_verified": True,
        }


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    prior_hashes, exclusion_record = load_prior_scene_hashes(args.exclude_scene_hashes)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=args.determinism == "warn")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no device fallback")
    device = torch.device(args.device)
    preflight = (
        cuda_training_preflight(args.channels, args.rf_weight)
        if args.device == "cuda"
        else {"status": "not_executed_cpu_smoke_only"}
    )
    grid, rf = GridSpec(), RFConfig()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory is nonempty; use a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir()
    protocol = {
        "status": "registered_before_training",
        "scientific_derivation_commit": SOURCE_BASE,
        "execution_source_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "source_sha256": source_hashes(),
        "task": "static_3D_discrete_weak_point_scatterer_support_reconstruction",
        "model_scope": "No continuous furniture, multipath, occlusion, motion, or measured-real-data claim",
        "initialization": "random for every learned main-model component",
        "stages": {
            "A": "supervised_geometry_pretraining_and_continuation_ablation",
            "A_continuation": "paired_supervised_geometry_and_RF_consistency",
            "RL": "not_implemented",
            "RLHF": "not_implemented",
        },
        "cuda_preflight": preflight,
        "determinism": {
            "mode": args.determinism,
            "bitwise_cuda_replay_claim": False,
            "known_warning_operations": ["AvgPool3d backward", "trilinear interpolation backward"],
        },
        "execution_variant": "H200 variant; original scientific runner also retained in source_sha256",
        "model_parameters": sum(p.numel() for p in OccupancyReadout(args.channels).parameters()),
        "grid": asdict(grid),
        "rf": asdict(rf),
        "layouts": LAYOUTS,
        "prefixes": PREFIXES,
        "training": {
            **vars(args),
            "output": str(args.output),
            "exclude_scene_hashes": str(args.exclude_scene_hashes) if args.exclude_scene_hashes else None,
        },
        "prior_scene_exclusion": exclusion_record,
        "paired_finetuning": "same selected warmup weights, reset optimizer, matched data order and update budget",
        "checkpoint_selection": "minimum validation full-prefix BCE plus 0.5 soft Dice loss",
        "threshold_selection": "validation macro scene IoU, per method and prefix",
        "baseline_selection": "validation IoU within .005 of best, then common-query soft RF NMSE",
        "query": "same height-diverse 8 unseen poses for both acquisition layouts; never estimator input",
        "split": "whole cuboid families including both vertical mirrors; duplicate hashes excluded across splits",
        "stress": "test susceptibility exp(0.2*N(0,1)-0.02), model retains nominal unit susceptibility",
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "threads": args.threads,
        },
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "status.json", {"status": "building_data"})
    scenes = {}
    excluded: set[str] = set(prior_hashes)
    for index, (split, count) in enumerate(
        (("train", args.train_scenes), ("validation", args.val_scenes), ("test", args.test_scenes))
    ):
        batch = generate_scenes(count, args.data_seed + index, grid, excluded_hashes=excluded)
        scenes[split] = batch
        excluded.update(batch.hashes)
    operators = {layout: make_operator(grid, rf, layout) for layout in LAYOUTS}
    query = make_operator(grid, rf, "height_diverse", heldout=True)
    observations, query_observations, features = {}, {}, {}
    arrays = {f"{split}_occupancy": batch.occupancy for split, batch in scenes.items()}
    for split_index, (split, batch) in enumerate(scenes.items()):
        query_observations[split] = observe_scenes(
            batch, query, noise_seed=args.data_seed + 400 + split_index
        )
        arrays[f"{split}_query_clean"] = query_observations[split].clean_y
        arrays[f"{split}_query_noisy"] = query_observations[split].y
        for layout_index, (layout, operator) in enumerate(operators.items()):
            obs = observe_scenes(
                batch, operator, noise_seed=args.data_seed + 100 + 10 * split_index + layout_index
            )
            observations[split, layout] = obs
            arrays[f"{split}_{layout}_y"] = obs.y
            features[split, layout] = {
                prefix: prefix_features(
                    obs.y, operator.matrix, grid.coordinates_m, grid.shape, rf.n_frequencies, prefix
                )
                for prefix in PREFIXES
            }
    np.savez_compressed(args.output / "dataset.npz", **arrays)
    dataset_metadata = {
        "splits": {key: value.metadata() for key, value in scenes.items()},
        "operators": {key: value.metadata() for key, value in operators.items()},
        "query_operator": query.metadata(),
        "observations": {
            f"{split}/{layout}": obs.metadata() for (split, layout), obs in observations.items()
        },
        "duplicate_scene_hash_count": sum(len(x.hashes) for x in scenes.values())
        - len(excluded - prior_hashes),
        "prior_scene_exclusion": exclusion_record,
        "prior_scene_overlap_count": len(
            prior_hashes.intersection(scene_hash for batch in scenes.values() for scene_hash in batch.hashes)
        ),
    }
    write_json(args.output / "data_provenance.json", dataset_metadata)
    train, val, test = (scenes[key] for key in ("train", "validation", "test"))
    learned, training_records, val_thresholds = {}, [], {}
    for layout, operator in operators.items():
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = OccupancyReadout(args.channels).to(device)
            common = dict(
                train_features=features["train", layout],
                val_features=features["validation", layout],
                train_truth=train.occupancy,
                val_truth=val.occupancy,
                train_y=observations["train", layout].y,
                operator=operator.matrix,
                n_frequencies=rf.n_frequencies,
                args=args,
                device=device,
            )
            warm_name = f"{layout}_seed{seed}_warmup"
            write_json(args.output / "status.json", {"status": "training", "stage": warm_name})
            model, record = train_stage(
                model,
                **common,
                epochs=args.warmup_epochs,
                seed=seed + 1000,
                rf_weight=0.0,
                learning_rate=0.002,
                name=warm_name,
            )
            training_records.append(record)
            branch_start = copy.deepcopy(model.state_dict())
            pair_records = []
            for variant, weight in (("geometry", 0.0), ("geometry_rf", args.rf_weight)):
                branch = OccupancyReadout(args.channels).to(device)
                branch.load_state_dict(branch_start)
                name = f"{layout}_seed{seed}_{variant}"
                write_json(args.output / "status.json", {"status": "training", "stage": name})
                branch, record = train_stage(
                    branch,
                    **common,
                    epochs=args.finetune_epochs,
                    seed=seed + 2000,
                    rf_weight=weight,
                    learning_rate=0.001,
                    name=name,
                )
                training_records.append(record)
                pair_records.append(record)
                learned[name] = (branch, layout, seed, variant)
                val_thresholds[name] = {}
                for prefix in PREFIXES:
                    probability = predict(
                        branch, features["validation", layout][prefix], device, args.batch_size
                    )
                    val_thresholds[name][prefix] = select_occupancy_threshold(probability, val.occupancy)
            for key in ("initial_weight_sha256", "batch_schedule_sha256", "optimizer_updates"):
                if pair_records[0][key] != pair_records[1][key]:
                    raise AssertionError(f"Unpaired finetuning {key}")
    write_json(args.output / "training_manifest.json", training_records)
    write_json(args.output / "validation_thresholds.json", val_thresholds)
    # Baselines are fully selected on validation before any test metrics are computed.
    baseline_configs = {}
    profile = train.occupancy.mean(axis=(0, 1, 2))
    profiles = {"training_mean_z": profile / profile.max(), "full_height": np.ones(grid.shape[2])}
    for layout, operator in operators.items():
        for prefix in PREFIXES:
            rows = prefix * rf.n_frequencies
            a = operator.matrix[:rows]
            y = observations["validation", layout].y[:, :rows]
            key = f"{layout}_prefix{prefix}"
            write_json(args.output / "status.json", {"status": "selecting_baselines", "cell": key})
            common_args = dict(iteration_candidates=(60, 180, 400), ridge_candidates=(0.0, 0.001, 0.01))
            inversion = select_fista_configuration(
                a, y, val.occupancy, query.matrix, query_observations["validation"].clean_y, **common_args
            )
            fixed = select_fixed_height_configuration(
                a,
                y,
                val.occupancy,
                query.matrix,
                query_observations["validation"].clean_y,
                grid.shape,
                profiles,
                **common_args,
            )
            matched = matched_filter(a, y).reshape(-1, *grid.shape)
            baseline_configs[key] = {
                "inverse3d": inversion,
                "fixed25d": fixed,
                "matched_filter": select_occupancy_threshold(matched, val.occupancy),
            }
            print(
                json.dumps(
                    {
                        "baselines_selected": key,
                        "inverse": inversion["selected"],
                        "fixed25d": fixed["selected"],
                    }
                ),
                flush=True,
            )
    write_json(args.output / "baseline_validation.json", baseline_configs)
    write_json(args.output / "status.json", {"status": "evaluating_frozen_models"})
    all_records, summaries, predictions = [], [], {}
    stress_rng = np.random.default_rng(args.data_seed + 900)
    susceptibility = np.exp(0.2 * stress_rng.standard_normal(test.occupancy.shape) - 0.02).astype(np.float32)
    stress_query = observe_scenes(test, query, noise_seed=args.data_seed + 901, susceptibility=susceptibility)
    conditions = {
        "nominal": (observations, query_observations["test"]),
        "susceptibility_mismatch": (
            {
                ("test", layout): observe_scenes(
                    test, operator, noise_seed=args.data_seed + 910 + i, susceptibility=susceptibility
                )
                for i, (layout, operator) in enumerate(operators.items())
            },
            stress_query,
        ),
    }
    np.savez_compressed(
        args.output / "stress_data.npz",
        susceptibility=susceptibility,
        query_clean=stress_query.clean_y,
        query_noisy=stress_query.y,
        **{layout: obs.y for (_, layout), obs in conditions["susceptibility_mismatch"][0].items()},
    )

    def record_prediction(prob, method, layout, prefix, seed, condition, threshold, elapsed_s, obs, qobs):
        rows = prefix * rf.n_frequencies
        detail = evaluate_field(
            prob,
            truth=test.occupancy,
            observations=obs.y[:, :rows],
            operator=operators[layout].matrix[:rows],
            query_operator=query.matrix,
            query_clean=qobs.clean_y,
            query_noisy=qobs.y,
            grid=grid,
            threshold=threshold,
            scene_hashes=test.hashes,
            pair_ids=test.pair_ids,
        )
        context = {
            "method": method,
            "layout": layout,
            "prefix_poses": prefix,
            "seed": seed,
            "condition": condition,
            "threshold": float(threshold),
        }
        all_records.extend([{**context, **row} for row in detail])
        summary = {
            **context,
            **macro_metrics(detail),
            "batch_inference_elapsed_s": elapsed_s,
            "test_scenes": len(test.occupancy),
        }
        summaries.append(summary)
        if prefix == 12:
            pred_key = f"{condition}__{layout}__{method}__seed{seed}"
            predictions[pred_key] = prob.astype(np.float32)
        print(json.dumps({"test_result": summary}), flush=True)

    for condition, (condition_obs, qobs) in conditions.items():
        for layout, operator in operators.items():
            obs = condition_obs["test", layout]
            for prefix in PREFIXES:
                rows = prefix * rf.n_frequencies
                a, y = operator.matrix[:rows], obs.y[:, :rows]
                config = baseline_configs[f"{layout}_prefix{prefix}"]
                for method in ("matched_filter", "fixed25d", "inverse3d"):
                    tick = time.perf_counter()
                    if method == "matched_filter":
                        field = matched_filter(a, y).reshape(-1, *grid.shape)
                        threshold = config[method]["threshold"]
                    else:
                        selected = config[method]["selected"]
                        if method == "inverse3d":
                            field = fista_box_inverse(
                                a, y, iterations=selected["iterations"], ridge=selected["ridge"]
                            ).reshape(-1, *grid.shape)
                        else:
                            field = fixed_height_25d(
                                a,
                                y,
                                grid.shape,
                                profiles[selected["profile_name"]],
                                iterations=selected["iterations"],
                                ridge=selected["ridge"],
                            )
                        threshold = selected["threshold"]
                    elapsed = time.perf_counter() - tick
                    record_prediction(
                        field, method, layout, prefix, None, condition, threshold, elapsed, obs, qobs
                    )
                feature = prefix_features(
                    obs.y, operator.matrix, grid.coordinates_m, grid.shape, rf.n_frequencies, prefix
                )
                for name, (model, model_layout, seed, variant) in learned.items():
                    if model_layout != layout:
                        continue
                    tick = time.perf_counter()
                    field = predict(model, feature, device, args.batch_size)
                    elapsed = time.perf_counter() - tick
                    record_prediction(
                        field,
                        variant,
                        layout,
                        prefix,
                        seed,
                        condition,
                        val_thresholds[name][prefix]["threshold"],
                        elapsed,
                        obs,
                        qobs,
                    )
    write_json(args.output / "per_scene_metrics.json", all_records)
    write_json(args.output / "summary.json", summaries)
    np.savez_compressed(args.output / "test_predictions.npz", **predictions)
    write_json(
        args.output / "status.json",
        {
            "status": "completed",
            "training_stages": len(training_records),
            "learned_test_cells": len(learned),
            "evaluation_cells": len(summaries),
            "elapsed_s": time.perf_counter() - started,
            "source_sha256": source_hashes(),
            "source_unchanged_during_run": source_hashes() == protocol["source_sha256"],
            "note": "Batched network timing excludes feature generation; not end-to-end latency",
        },
    )
    print(json.dumps({"completed": str(args.output), "elapsed_s": time.perf_counter() - started}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=512)
    parser.add_argument("--val-scenes", type=int, default=128)
    parser.add_argument("--test-scenes", type=int, default=192)
    parser.add_argument("--data-seed", type=int, default=20261124)
    parser.add_argument("--exclude-scene-hashes", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rf-weight", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--determinism", choices=("strict", "warn"), default="strict")
    args = parser.parse_args()
    if min(args.train_scenes, args.val_scenes, args.test_scenes) < 2:
        parser.error("Each split needs at least two scenes")
    if any(n % 2 for n in (args.train_scenes, args.val_scenes, args.test_scenes)):
        parser.error("Mirror-paired split counts must be even")
    if min(args.warmup_epochs, args.finetune_epochs, args.batch_size, args.threads) < 1:
        parser.error("Training counts must be positive")
    if not np.isfinite(args.rf_weight) or args.rf_weight <= 0:
        parser.error("Physical branch weight must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Initialization seeds must be distinct")
    output_preexists = args.output.exists() and any(args.output.iterdir())
    try:
        run(args)
    except Exception as exc:
        if args.output.is_dir() and not output_preexists:
            write_json(
                args.output / "status.json",
                {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()},
            )
        raise


if __name__ == "__main__":
    main()
