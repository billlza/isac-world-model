"""Replay completed 3D checkpoints and compute paired scene-family intervals.

Run only after occupancy3d_study.py reports completion. This script does not
change model, data, training, or existing result files. Its outputs are audit.json
and paired_comparisons.json in the completed run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from isac_wm.occupancy3d_data import GridSpec, RFConfig, make_operator, scene_hash
from isac_wm.occupancy3d_evaluation import paired_family_bootstrap
from isac_wm.occupancy3d_model import OccupancyReadout, prefix_features, weight_fingerprint

METRICS = {
    "iou_3d": "higher",
    "brier": "lower",
    "height_column_rmse_m": "lower",
    "query_rf_nmse_soft": "lower",
    "query_rf_nmse_hard": "lower",
}
LEARNED = ("geometry", "geometry_rf")
BASELINES = ("matched_filter", "fixed25d", "inverse3d")


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(run_dir: Path, audit: dict, tolerance: float) -> tuple[dict, dict]:
    protocol = read_json(run_dir / "protocol.json")
    provenance = read_json(run_dir / "data_provenance.json")
    status = read_json(run_dir / "status.json")
    repository = Path(__file__).resolve().parents[1]

    def check(name: str, condition: bool, **details) -> None:
        audit["checks"].append({"check": name, "passed": bool(condition), **details})
        if not condition:
            raise AssertionError(f"Audit failed: {name}")

    check("run_completed", status.get("status") == "completed")
    check("source_unchanged_during_run", status.get("source_unchanged_during_run") is True)
    actual_source = {name: file_hash(repository / name) for name in protocol["source_sha256"]}
    check(
        "current_source_matches_registered_source",
        actual_source == protocol["source_sha256"],
        actual_sha256=actual_source,
    )
    check("completed_status_source_matches_protocol", status["source_sha256"] == protocol["source_sha256"])
    grid_values = protocol["grid"]
    grid = GridSpec(**{key: tuple(grid_values[key]) for key in ("shape", "lower_m", "upper_m")})
    rf = RFConfig(**protocol["rf"])
    all_scene_hashes, all_family_sets = [], []
    with np.load(run_dir / "dataset.npz", allow_pickle=False) as dataset:
        for split, metadata in provenance["splits"].items():
            occupancy = dataset[f"{split}_occupancy"]
            hashes = [scene_hash(volume, grid) for volume in occupancy]
            check(f"{split}_scene_hashes_match_bytes", hashes == metadata["scene_hashes"])
            joined = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
            check(f"{split}_dataset_hash_matches", joined == metadata["dataset_sha256"])
            expected_count = protocol["training"][
                "val_scenes" if split == "validation" else f"{split}_scenes"
            ]
            check(f"{split}_count", len(hashes) == expected_count, scenes=len(hashes))
            families = metadata["mirror_pair_ids"]
            counts = Counter(families)
            check(
                f"{split}_mirror_families_are_pairs",
                len(families) == len(hashes) and all(c == 2 for c in counts.values()),
            )
            family_indices: dict[str, list[int]] = {}
            for index, family in enumerate(families):
                family_indices.setdefault(family, []).append(index)
            check(
                f"{split}_pairs_are_geometric_mirrors",
                all(
                    np.array_equal(occupancy[indices[0]], occupancy[indices[1]][:, :, ::-1])
                    for indices in family_indices.values()
                ),
            )
            all_scene_hashes.extend(hashes)
            all_family_sets.append(set(families))
        duplicate_count = len(all_scene_hashes) - len(set(all_scene_hashes))
        check("zero_duplicate_scenes_across_splits", duplicate_count == 0, duplicate_count=duplicate_count)
        check(
            "zero_shared_families_across_splits",
            all(
                not left.intersection(right)
                for i, left in enumerate(all_family_sets)
                for right in all_family_sets[i + 1 :]
            ),
        )
        check("recorded_duplicate_count_matches", provenance["duplicate_scene_hash_count"] == duplicate_count)
        test_y = {layout: dataset[f"test_{layout}_y"].copy() for layout in protocol["layouts"]}

    records = read_json(run_dir / "training_manifest.json")
    by_name = {row["name"]: row for row in records}
    check("unique_training_stage_names", len(by_name) == len(records))
    seeds = protocol["training"]["seeds"]
    expected_stage_count = len(protocol["layouts"]) * len(seeds) * 3
    check("all_training_stages_recorded", len(records) == expected_stage_count)
    for layout in protocol["layouts"]:
        for seed in seeds:
            warm = by_name[f"{layout}_seed{seed}_warmup"]
            branches = [by_name[f"{layout}_seed{seed}_{variant}"] for variant in LEARNED]
            check(f"{layout}_seed{seed}_both_stages_completed", all(row["completed"] for row in branches))
            for key in ("initial_weight_sha256", "batch_schedule_sha256", "optimizer_updates"):
                check(
                    f"{layout}_seed{seed}_paired_{key}",
                    branches[0][key] == branches[1][key],
                    value=branches[0][key],
                )
            check(
                f"{layout}_seed{seed}_forks_selected_warmup",
                all(row["initial_weight_sha256"] == warm["selected_weight_sha256"] for row in branches),
            )

    torch.set_num_threads(int(protocol["runtime"]["threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    batch_size = int(protocol["training"]["batch_size"])
    replay_records = []
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as stored:
        for layout in protocol["layouts"]:
            operator = make_operator(grid, rf, layout)
            operator_hash = hashlib.sha256(operator.matrix.tobytes()).hexdigest()
            check(
                f"{layout}_operator_reproduces",
                operator_hash == provenance["operators"][layout]["operator_sha256"],
            )
            feature = prefix_features(
                test_y[layout],
                operator.matrix,
                grid.coordinates_m,
                grid.shape,
                rf.n_frequencies,
                max(protocol["prefixes"]),
            )
            for seed in seeds:
                for variant in LEARNED:
                    name = f"{layout}_seed{seed}_{variant}"
                    record = by_name[name]
                    checkpoint_path = (run_dir / record["checkpoint"]).resolve()
                    check(
                        f"{name}_checkpoint_path_is_local", checkpoint_path.is_relative_to(run_dir.resolve())
                    )
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    check(
                        f"{name}_checkpoint_source", checkpoint["source_sha256"] == protocol["source_sha256"]
                    )
                    check(f"{name}_selected_epoch", checkpoint["selected_epoch"] == record["selected_epoch"])
                    model = OccupancyReadout(int(checkpoint["channels"]))
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model.eval()
                    digest = weight_fingerprint(model)
                    check(f"{name}_selected_weight_fingerprint", digest == record["selected_weight_sha256"])
                    prediction = []
                    with torch.no_grad():
                        for start in range(0, len(feature), batch_size):
                            prediction.append(
                                model(torch.from_numpy(feature[start : start + batch_size])).sigmoid().numpy()
                            )
                    probability = np.concatenate(prediction)
                    key = f"nominal__{layout}__{variant}__seed{seed}"
                    original = stored[key]
                    shape_matches = probability.shape == original.shape
                    maximum = float(np.max(np.abs(probability - original))) if shape_matches else None
                    check(
                        f"{name}_nominal_full_prefix_replay",
                        shape_matches
                        and bool(np.all(np.isfinite(probability)))
                        and maximum is not None
                        and maximum <= tolerance,
                        maximum_absolute_difference=maximum,
                        absolute_tolerance=tolerance,
                    )
                    replay_records.append(
                        {
                            "name": name,
                            "checkpoint": record["checkpoint"],
                            "checkpoint_sha256": file_hash(checkpoint_path),
                            "selected_weight_sha256": digest,
                            "prediction_key": key,
                            "maximum_absolute_difference": maximum,
                            "parameter_count": sum(p.numel() for p in model.parameters()),
                        }
                    )
    audit["checkpoint_replay"] = replay_records
    audit["replayed_checkpoint_count"] = len(replay_records)
    audit["checkpoint_replay_batch_size"] = batch_size
    audit["checkpoint_replay_device"] = "cpu"
    audit["inputs_sha256"] = {
        name: file_hash(run_dir / name)
        for name in (
            "protocol.json",
            "data_provenance.json",
            "dataset.npz",
            "training_manifest.json",
            "summary.json",
            "per_scene_metrics.json",
            "test_predictions.npz",
        )
    }
    return protocol, provenance


def paired_statistics(
    run_dir: Path, protocol: dict, provenance: dict, repetitions: int, bootstrap_seed: int
) -> dict:
    records = read_json(run_dir / "per_scene_metrics.json")
    cells: dict[tuple, dict[str, dict]] = {}
    for row in records:
        key = (row["condition"], row["layout"], row["prefix_poses"], row["method"], row["seed"])
        cell = cells.setdefault(key, {})
        if row["scene_hash"] in cell:
            raise AssertionError(f"Duplicated scene metric in {key}")
        cell[row["scene_hash"]] = row
    test = provenance["splits"]["test"]
    scene_order, family_ids = test["scene_hashes"], test["mirror_pair_ids"]
    for key, cell in cells.items():
        if set(cell) != set(scene_order):
            raise AssertionError(f"Missing or extra test scenes in {key}")
        if [cell[name]["family_id"] for name in scene_order] != family_ids:
            raise AssertionError(f"Scene-family identity mismatch in {key}")
    seeds = protocol["training"]["seeds"]
    conditions = ("nominal", "susceptibility_mismatch")
    expected_count = (
        len(conditions)
        * len(protocol["layouts"])
        * len(protocol["prefixes"])
        * (len(BASELINES) + len(LEARNED) * len(seeds))
    )
    if len(cells) != expected_count:
        raise AssertionError("Incomplete registered evaluation matrix")

    def values(condition, layout, prefix, method, seed, metric):
        cell = cells[(condition, layout, prefix, method, seed)]
        return np.asarray([cell[name][metric] for name in scene_order], dtype=np.float64)

    def comparison(kind, condition, prefix, metric, a_arrays, b_arrays, context):
        a, b = np.mean(a_arrays, axis=0), np.mean(b_arrays, axis=0)
        interval = paired_family_bootstrap(a, b, family_ids, repetitions=repetitions, seed=bootstrap_seed)
        sign = 1 if METRICS[metric] == "higher" else -1
        low, high = interval["lower"], interval["upper"]
        return {
            "comparison": kind,
            "condition": condition,
            "prefix_poses": prefix,
            "metric": metric,
            "better_direction": METRICS[metric],
            **context,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            **interval,
            "improvement_a_over_b": sign * interval["mean_difference_a_minus_b"],
            "improvement_ci_lower": low if sign > 0 else -high,
            "improvement_ci_upper": high if sign > 0 else -low,
            "per_initialization": [
                {
                    "seed": seed,
                    "mean_a": float(np.mean(aa)),
                    "mean_b": float(np.mean(bb)),
                    "difference_a_minus_b": float(np.mean(aa - bb)),
                }
                for seed, aa, bb in zip(context["averaged_initializations"], a_arrays, b_arrays, strict=True)
            ],
        }

    comparisons = []
    for condition in conditions:
        for prefix in protocol["prefixes"]:
            for metric in METRICS:
                for layout in protocol["layouts"]:
                    a = [values(condition, layout, prefix, "geometry_rf", seed, metric) for seed in seeds]
                    b = [values(condition, layout, prefix, "geometry", seed, metric) for seed in seeds]
                    comparisons.append(
                        comparison(
                            "rf_consistency_vs_geometry",
                            condition,
                            prefix,
                            metric,
                            a,
                            b,
                            {
                                "layout": layout,
                                "a": "geometry_rf",
                                "b": "geometry",
                                "averaged_initializations": seeds,
                            },
                        )
                    )
                for method in (*BASELINES, *LEARNED):
                    method_seeds = seeds if method in LEARNED else [None]
                    a = [
                        values(condition, "height_diverse", prefix, method, seed, metric)
                        for seed in method_seeds
                    ]
                    b = [
                        values(condition, "same_height", prefix, method, seed, metric)
                        for seed in method_seeds
                    ]
                    comparisons.append(
                        comparison(
                            "height_diverse_vs_same_height",
                            condition,
                            prefix,
                            metric,
                            a,
                            b,
                            {
                                "method": method,
                                "a": "height_diverse",
                                "b": "same_height",
                                "averaged_initializations": method_seeds,
                            },
                        )
                    )
    return {
        "status": "completed",
        "comparison_count": len(comparisons),
        "metrics": METRICS,
        "test_scenes": len(scene_order),
        "independent_scene_families": len(set(family_ids)),
        "method": "Average each scene metric over registered initializations, then paired family percentile bootstrap",
        "interval_scope": "Scene-family sampling conditional on these trained initializations; not seed-population uncertainty",
        "no_result_filtering": "All requested comparisons retained, including negative or zero improvement",
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=5e-6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261124)
    args = parser.parse_args()
    if not math.isfinite(args.absolute_tolerance) or args.absolute_tolerance <= 0:
        parser.error("absolute tolerance must be positive and finite")
    if args.bootstrap_repetitions < 1:
        parser.error("bootstrap repetitions must be positive")
    # A running experiment is never touched by this audit, including its status.
    status = read_json(args.run_dir / "status.json")
    if status.get("status") != "completed":
        raise RuntimeError("Audit accepts only completed runs; no files were written")
    audit = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir.resolve()),
        "audit_source_sha256": file_hash(Path(__file__)),
        "checks": [],
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
    }
    try:
        protocol, provenance = verify_run(args.run_dir, audit, args.absolute_tolerance)
        statistics = paired_statistics(
            args.run_dir, protocol, provenance, args.bootstrap_repetitions, args.bootstrap_seed
        )
        write_json(args.run_dir / "paired_comparisons.json", statistics)
        audit["paired_comparisons_sha256"] = file_hash(args.run_dir / "paired_comparisons.json")
        audit["paired_comparison_count"] = statistics["comparison_count"]
        audit["status"] = "passed"
        write_json(args.run_dir / "audit.json", audit)
        print(
            json.dumps(
                {
                    "audit": "passed",
                    "checkpoint_replays": audit["replayed_checkpoint_count"],
                    "paired_comparisons": statistics["comparison_count"],
                    "independent_scene_families": statistics["independent_scene_families"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = repr(exc)
        audit["traceback"] = traceback.format_exc()
        write_json(args.run_dir / "audit.json", audit)
        raise


if __name__ == "__main__":
    main()
