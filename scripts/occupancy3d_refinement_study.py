"""Fresh holdout study of RF correction initialized by a frozen learned map.

The original registered study motivates this new hypothesis and is development
evidence for it. Models are never retrained. Original validation selects bounded
correction budgets; fresh scene families are evaluated only after that selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from occupancy3d_study import evaluate_field, macro_metrics, predict, write_json

from isac_wm.occupancy3d_data import GridSpec, RFConfig, generate_scenes, make_operator, observe_scenes
from isac_wm.occupancy3d_evaluation import (
    complex_nmse,
    fista_box_inverse,
    paired_family_bootstrap,
    select_occupancy_threshold,
)
from isac_wm.occupancy3d_model import OccupancyReadout, prefix_features, weight_fingerprint

PREFIXES = (3, 6, 12)
STEPS = (8, 24, 60)
METRICS = {"iou_3d": "higher", "brier": "lower", "height_column_rmse_m": "lower",
           "query_rf_nmse_soft": "lower", "query_rf_nmse_hard": "lower"}


def read_json(path: Path):
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_sources(reference_protocol: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    sources = {name: file_hash(root / name) for name in reference_protocol["source_sha256"]}
    sources[str(Path(__file__).resolve().relative_to(root))] = file_hash(Path(__file__))
    return sources


def paired_statistics(records: list[dict], scene_hashes: tuple[str, ...], family_ids: tuple[str, ...],
                      seeds: list[int]) -> dict:
    cells = {}
    for row in records:
        key = (row["condition"], row["prefix_poses"], row["method"], row["seed"])
        cell = cells.setdefault(key, {})
        if row["scene_hash"] in cell:
            raise AssertionError("Duplicate fresh-test metric")
        cell[row["scene_hash"]] = row

    def values(condition, prefix, method, seed, metric):
        cell = cells[(condition, prefix, method, seed)]
        if set(cell) != set(scene_hashes):
            raise AssertionError("Missing fresh-test scene metric")
        return np.asarray([cell[h][metric] for h in scene_hashes], dtype=np.float64)

    comparisons = []
    for condition in ("nominal", "susceptibility_mismatch"):
        for prefix in PREFIXES:
            for reference in ("geometry_rf_frozen", "inverse3d_matched_steps", "inverse3d_converged"):
                for metric, direction in METRICS.items():
                    a = [values(condition, prefix, "geometry_rf_refined", seed, metric) for seed in seeds]
                    b = [values(condition, prefix, reference, None if reference == "inverse3d_converged" else seed, metric)
                         for seed in seeds]
                    mean_a, mean_b = np.mean(a, axis=0), np.mean(b, axis=0)
                    interval = paired_family_bootstrap(mean_a, mean_b, family_ids, seed=20261129, repetitions=2000)
                    sign = 1 if direction == "higher" else -1
                    comparisons.append({
                        "condition": condition, "prefix_poses": prefix, "a": "geometry_rf_refined", "b": reference,
                        "metric": metric, "better_direction": direction, "mean_a": float(np.mean(mean_a)),
                        "mean_b": float(np.mean(mean_b)), **interval,
                        "improvement_a_over_b": sign * interval["mean_difference_a_minus_b"],
                        "improvement_ci_lower": interval["lower"] if sign > 0 else -interval["upper"],
                        "improvement_ci_upper": interval["upper"] if sign > 0 else -interval["lower"],
                        "per_initialization": [
                            {"seed": seed, "mean_a": float(np.mean(aa)), "mean_b": float(np.mean(bb)),
                             "difference_a_minus_b": float(np.mean(aa - bb))}
                            for seed, aa, bb in zip(seeds, a, b, strict=True)
                        ],
                    })
    return {"comparisons": comparisons, "comparison_count": len(comparisons),
            "test_scenes": len(scene_hashes), "independent_scene_families": len(set(family_ids)),
            "interval_scope": "Pointwise paired family percentile intervals conditional on frozen seeds; no multiplicity correction",
            "no_result_filtering": True}


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    reference = args.reference_run
    original = read_json(reference / "protocol.json")
    original_status = read_json(reference / "status.json")
    if original_status["status"] != "completed" or not original_status["source_unchanged_during_run"]:
        raise RuntimeError("Reference must be a completed study with frozen source")
    for name, expected in original["source_sha256"].items():
        if file_hash(root / name) != expected:
            raise RuntimeError(f"Reference source changed: {name}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    seeds = list(original["training"]["seeds"])
    grid = GridSpec(**{key: tuple(original["grid"][key]) for key in ("shape", "lower_m", "upper_m")})
    rf = RFConfig(**original["rf"])
    operator = make_operator(grid, rf, "height_diverse")
    query = make_operator(grid, rf, "height_diverse", heldout=True)
    provenance = read_json(reference / "data_provenance.json")
    training = {row["name"]: row for row in read_json(reference / "training_manifest.json")}
    original_thresholds = read_json(reference / "validation_thresholds.json")
    original_baselines = read_json(reference / "baseline_validation.json")
    models, checkpoints = {}, {}
    for seed in seeds:
        name = f"height_diverse_seed{seed}_geometry_rf"
        path = reference / training[name]["checkpoint"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = OccupancyReadout(checkpoint["channels"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        if weight_fingerprint(model) != training[name]["selected_weight_sha256"]:
            raise AssertionError("Frozen model fingerprint mismatch")
        models[seed] = model
        checkpoints[str(seed)] = {"path": str(path), "sha256": file_hash(path),
                                  "selected_weight_sha256": weight_fingerprint(model),
                                  "selected_epoch": checkpoint["selected_epoch"]}
    protocol = {
        "status": "registered_before_fresh_test_evaluation",
        "hypothesis": "Bounded RF correction of a learned initial map retains useful sparse-view priors while enforcing observed RF",
        "development_evidence": "The original registered run motivated this new refinement hypothesis; fresh test families are used here",
        "reference_run": str(reference.resolve()), "reference_protocol_sha256": file_hash(reference / "protocol.json"),
        "source_sha256": current_sources(original), "frozen_checkpoints": checkpoints,
        "model_updates": "No retraining; checkpoint weights remain frozen",
        "grid": asdict(grid), "rf": asdict(rf), "layout": "height_diverse", "prefixes": PREFIXES,
        "fresh_test_seed": args.data_seed, "fresh_test_scenes": args.test_scenes,
        "exclude": "Every scene hash in original train, validation, and test; vertical mirror families stay together",
        "validation": "Original validation scenes and observations only",
        "step_candidates": STEPS, "ridge": "Original validation-selected converged FISTA ridge at each prefix",
        "selection": "Within .005 of best candidate validation macro 3D IoU, then lowest common-query soft RF NMSE",
        "matched_step_control": "Zero initialization; identical steps and ridge to each refined model; own validation threshold",
        "converged_reference": "Original validation-selected FISTA configuration and threshold, unchanged",
        "susceptibility_stress": {"log_std": 0.2, "mean": 1.0, "seed": args.data_seed + 900},
        "noise_pairing": "Identical acquisition AWGN RNG draws across nominal and stress; identical query AWGN draws across them, before complex64 rounding",
        "acquisition_noise_seed": args.data_seed + 1000, "query_noise_seed": args.data_seed + 2000,
        "timing": "Standalone batch pipeline components include RF feature generation, frozen CNN inference, and inverse updates where applicable",
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                    "device": "cpu", "threads": args.threads, "batch_size": args.batch_size},
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "status.json", {"status": "selecting_on_original_validation"})
    with np.load(reference / "dataset.npz", allow_pickle=False) as dataset:
        val_truth = dataset["validation_occupancy"].copy()
        val_y = dataset["validation_height_diverse_y"].copy()
        val_query = dataset["validation_query_clean"].copy()
    configs = {}
    for prefix in PREFIXES:
        rows = prefix * rf.n_frequencies
        a, y = operator.matrix[:rows], val_y[:, :rows]
        feature = prefix_features(val_y, operator.matrix, grid.coordinates_m, grid.shape, rf.n_frequencies, prefix)
        converged = original_baselines[f"height_diverse_prefix{prefix}"]["inverse3d"]["selected"]
        ridge = converged["ridge"]
        for seed, model in models.items():
            probability = predict(model, feature, device, args.batch_size)
            candidates = []
            for steps in STEPS:
                refined = fista_box_inverse(a, y, iterations=steps, ridge=ridge,
                                           initial=probability.reshape(len(probability), -1))
                selected = select_occupancy_threshold(refined, val_truth.reshape(len(val_truth), -1))
                candidates.append({"iterations": steps, "ridge": ridge, **selected,
                                   "validation_query_nmse": float(np.mean(complex_nmse(refined @ query.matrix.T, val_query)))})
            best_iou = max(row["validation_macro_iou_3d"] for row in candidates)
            selected = min((row for row in candidates if row["validation_macro_iou_3d"] >= best_iou - 0.005),
                           key=lambda row: (row["validation_query_nmse"], row["iterations"]))
            zero = fista_box_inverse(a, y, iterations=selected["iterations"], ridge=ridge)
            zero_threshold = select_occupancy_threshold(zero, val_truth.reshape(len(val_truth), -1))
            name = f"height_diverse_seed{seed}_geometry_rf"
            config = {"seed": seed, "prefix_poses": prefix, "selected": selected, "candidates": candidates,
                      "frozen_model_threshold": original_thresholds[name][str(prefix)]["threshold"],
                      "matched_step_threshold": zero_threshold, "converged_reference": converged}
            configs[f"prefix{prefix}_seed{seed}"] = config
            print(json.dumps({"validation_selected": config}), flush=True)
    write_json(args.output / "validation_selection.json", configs)
    selection_hash = file_hash(args.output / "validation_selection.json")
    excluded = {value for split in provenance["splits"].values() for value in split["scene_hashes"]}
    test = generate_scenes(args.test_scenes, args.data_seed, grid, excluded_hashes=excluded)
    if excluded.intersection(test.hashes):
        raise AssertionError("Fresh-test scene overlaps original study")
    susceptibility = np.exp(0.2 * np.random.default_rng(args.data_seed + 900).standard_normal(test.occupancy.shape) - 0.02).astype(np.float32)
    observations = {}
    for condition, material in (("nominal", None), ("susceptibility_mismatch", susceptibility)):
        observations[condition] = (
            observe_scenes(test, operator, args.data_seed + 1000, susceptibility=material),
            observe_scenes(test, query, args.data_seed + 2000, susceptibility=material),
        )
    write_json(args.output / "data_provenance.json", {
        "fresh_test": test.metadata(), "original_scene_hashes_excluded": len(excluded),
        "overlap_count": len(excluded.intersection(test.hashes)),
        "operator": operator.metadata(), "query_operator": query.metadata(),
        "observations": {condition: {"acquisition": obs.metadata(), "query": qobs.metadata()}
                         for condition, (obs, qobs) in observations.items()},
    })
    np.savez_compressed(args.output / "fresh_dataset.npz", occupancy=test.occupancy, susceptibility=susceptibility,
                        **{f"{condition}_{name}": value for condition, (obs, qobs) in observations.items()
                           for name, value in (("y", obs.y), ("clean_y", obs.clean_y),
                                               ("query_clean", qobs.clean_y), ("query_noisy", qobs.y))})
    write_json(args.output / "status.json", {"status": "evaluating_fresh_holdout", "validation_selection_sha256": selection_hash})
    per_scene, summaries, saved_predictions = [], [], {}

    def record(field, condition, prefix, method, seed, threshold, timings, obs, qobs):
        rows = prefix * rf.n_frequencies
        detail = evaluate_field(field, truth=test.occupancy, observations=obs.y[:, :rows],
                                operator=operator.matrix[:rows], query_operator=query.matrix,
                                query_clean=qobs.clean_y, query_noisy=qobs.y, grid=grid, threshold=threshold,
                                scene_hashes=test.hashes, pair_ids=test.pair_ids)
        context = {"condition": condition, "layout": "height_diverse", "prefix_poses": prefix,
                   "method": method, "seed": seed, "threshold": float(threshold)}
        per_scene.extend([{**context, **row} for row in detail])
        summary = {**context, **macro_metrics(detail), "timing_components_s": timings,
                   "standalone_batch_compute_s": sum(timings.values()), "test_scenes": len(test.occupancy)}
        summaries.append(summary)
        saved_predictions[f"{condition}__prefix{prefix}__{method}__seed{seed}"] = field.astype(np.float32)
        print(json.dumps({"fresh_test_result": summary}), flush=True)

    for condition, (obs, qobs) in observations.items():
        for prefix in PREFIXES:
            rows = prefix * rf.n_frequencies
            a, y = operator.matrix[:rows], obs.y[:, :rows]
            converged = configs[f"prefix{prefix}_seed{seeds[0]}"]["converged_reference"]
            tick = time.perf_counter()
            inverse = fista_box_inverse(a, y, iterations=converged["iterations"], ridge=converged["ridge"])
            inverse_time = time.perf_counter() - tick
            record(inverse.reshape(-1, *grid.shape), condition, prefix, "inverse3d_converged", None,
                   converged["threshold"], {"inverse": inverse_time}, obs, qobs)
            tick = time.perf_counter()
            feature = prefix_features(obs.y, operator.matrix, grid.coordinates_m, grid.shape, rf.n_frequencies, prefix)
            feature_time = time.perf_counter() - tick
            for seed, model in models.items():
                config = configs[f"prefix{prefix}_seed{seed}"]
                selected = config["selected"]
                tick = time.perf_counter()
                probability = predict(model, feature, device, args.batch_size)
                network_time = time.perf_counter() - tick
                record(probability, condition, prefix, "geometry_rf_frozen", seed,
                       config["frozen_model_threshold"], {"rf_features": feature_time, "network": network_time}, obs, qobs)
                tick = time.perf_counter()
                refined = fista_box_inverse(a, y, iterations=selected["iterations"], ridge=selected["ridge"],
                                           initial=probability.reshape(len(probability), -1))
                inverse_time = time.perf_counter() - tick
                record(refined.reshape(-1, *grid.shape), condition, prefix, "geometry_rf_refined", seed,
                       selected["threshold"], {"rf_features": feature_time, "network": network_time,
                                              "inverse": inverse_time}, obs, qobs)
                tick = time.perf_counter()
                zero = fista_box_inverse(a, y, iterations=selected["iterations"], ridge=selected["ridge"])
                inverse_time = time.perf_counter() - tick
                record(zero.reshape(-1, *grid.shape), condition, prefix, "inverse3d_matched_steps", seed,
                       config["matched_step_threshold"]["threshold"], {"inverse": inverse_time}, obs, qobs)
    write_json(args.output / "per_scene_metrics.json", per_scene)
    write_json(args.output / "summary.json", summaries)
    np.savez_compressed(args.output / "test_predictions.npz", **saved_predictions)
    statistics = paired_statistics(per_scene, test.hashes, test.pair_ids, seeds)
    write_json(args.output / "paired_comparisons.json", statistics)
    unchanged_sources = current_sources(original) == protocol["source_sha256"]
    unchanged_weights = all(file_hash(Path(item["path"])) == item["sha256"] for item in checkpoints.values())
    unchanged_selection = file_hash(args.output / "validation_selection.json") == selection_hash
    if not (unchanged_sources and unchanged_weights and unchanged_selection):
        raise AssertionError("Sources, weights, or selected settings changed during fresh-test evaluation")
    write_json(args.output / "status.json", {
        "status": "completed", "elapsed_s": time.perf_counter() - started,
        "fresh_test_scenes": len(test.occupancy), "independent_scene_families": len(set(test.pair_ids)),
        "evaluation_cells": len(summaries), "comparisons": statistics["comparison_count"],
        "source_unchanged_during_run": unchanged_sources, "checkpoint_weights_unchanged": unchanged_weights,
        "validation_selection_unchanged": unchanged_selection,
        "source_sha256": current_sources(original), "validation_selection_sha256": selection_hash,
        "fresh_dataset_sha256": file_hash(args.output / "fresh_dataset.npz"),
        "note": "No retraining. Matched inverse iteration counts do not imply equal total compute; neural refinement includes CNN cost.",
    })
    print(json.dumps({"completed": str(args.output), "elapsed_s": time.perf_counter() - started}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-scenes", type=int, default=192)
    parser.add_argument("--data-seed", type=int, default=20261129)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.test_scenes < 2 or args.test_scenes % 2 or min(args.threads, args.batch_size) < 1:
        parser.error("Use a positive even test count and positive threads/batch size")
    existed = args.output.exists() and any(args.output.iterdir())
    try:
        run(args)
    except Exception as exc:
        if args.output.is_dir() and not existed:
            write_json(args.output / "status.json", {"status": "failed", "error": repr(exc),
                                                     "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
