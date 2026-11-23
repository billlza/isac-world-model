"""Render audited result tables and figures for a completed occupancy3d study.

Usage:
  python scripts/occupancy3d_report.py --run-dir research_outputs/occupancy3d/RUN \
      --output-dir research_outputs/occupancy3d/RUN/report

The figures describe synthetic discrete point-scatterer support. They do not
claim continuous room-surface recovery, real measurements or dynamic prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METHODS = ("matched_filter", "fixed25d", "inverse3d", "geometry", "geometry_rf")
METHOD_LABELS = {
    "matched_filter": "Matched filter",
    "fixed25d": "Fixed-height 2.5D",
    "inverse3d": "FISTA 3D",
    "geometry": "Geometry network",
    "geometry_rf": "Geometry + RF network",
}
SHORT_LABELS = (
    "Matched\nfilter",
    "Fixed-height\n2.5D",
    "FISTA\n3D",
    "Geometry\nnetwork",
    "Geometry + RF\nnetwork",
)
COLORS = {
    "matched_filter": "#94A3B8",
    "fixed25d": "#A879C5",
    "inverse3d": "#D68126",
    "geometry": "#2474B5",
    "geometry_rf": "#008B7A",
}
LAYOUTS = ("same_height", "height_diverse")
CONDITIONS = ("nominal", "susceptibility_mismatch")
METRICS = (
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
    "batch_inference_elapsed_s",
)
PLOT_METRICS = ("iou_3d", "height_column_rmse_m", "query_rf_nmse_soft")
METRIC_LABELS = (
    "3D IoU (higher is better)",
    "Column-height RMSE, m (lower)",
    "Unseen-pose RF NMSE (lower; log scale)",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell_key(row: dict) -> tuple:
    return (row["condition"], row["layout"], row["prefix_poses"], row["method"], row["seed"])


def load_completed_run(run_dir: Path) -> dict:
    status = load_json(run_dir / "status.json")
    if status.get("status") != "completed":
        raise RuntimeError("Study is not completed; refusing to report partial results")
    if status.get("source_unchanged_during_run") is not True:
        raise RuntimeError("Study source provenance changed or was not verified")
    protocol = load_json(run_dir / "protocol.json")
    provenance = load_json(run_dir / "data_provenance.json")
    summary = load_json(run_dir / "summary.json")
    details = load_json(run_dir / "per_scene_metrics.json")
    seeds = tuple(protocol["training"]["seeds"])
    prefixes = tuple(protocol["prefixes"])
    expected = {
        (condition, layout, prefix, method, seed)
        for condition in CONDITIONS
        for layout in LAYOUTS
        for prefix in prefixes
        for method in METHODS
        for seed in (seeds if method in ("geometry", "geometry_rf") else (None,))
    }
    cells = {cell_key(row): row for row in summary}
    if len(cells) != len(summary) or set(cells) != expected:
        raise ValueError("Summary has duplicate, missing or unexpected experimental cells")
    scene_hashes = provenance["splits"]["test"]["scene_hashes"]
    if len(scene_hashes) != len(set(scene_hashes)):
        raise ValueError("Test scene hashes are not unique")
    expected_scenes = set(scene_hashes)
    seen = {key: set() for key in cells}
    detail_map = {}
    for row in details:
        key, scene = cell_key(row), row["scene_hash"]
        if key not in seen or scene not in expected_scenes or scene in seen[key]:
            raise ValueError("Per-scene records are duplicated or do not match registered cells")
        seen[key].add(scene)
        detail_map[key + (scene,)] = row
    if any(value != expected_scenes for value in seen.values()):
        raise ValueError("At least one method cell is missing test scenes")
    for row in summary:
        if row["test_scenes"] != len(scene_hashes):
            raise ValueError("Summary test count disagrees with dataset provenance")
    return {
        "status": status,
        "protocol": protocol,
        "provenance": provenance,
        "summary": summary,
        "details": details,
        "cells": cells,
        "detail_map": detail_map,
        "seeds": seeds,
        "prefixes": prefixes,
        "scene_hashes": scene_hashes,
    }


def aggregate_cells(run: dict) -> list[dict]:
    result = []
    for condition in CONDITIONS:
        for layout in LAYOUTS:
            for prefix in run["prefixes"]:
                for method in METHODS:
                    seeds = run["seeds"] if method in ("geometry", "geometry_rf") else (None,)
                    rows = [run["cells"][condition, layout, prefix, method, seed] for seed in seeds]
                    output = {
                        "condition": condition,
                        "layout": layout,
                        "prefix_poses": prefix,
                        "method": method,
                        "model_count": len(rows),
                        "seeds": list(seeds),
                        "test_scenes_per_model": rows[0]["test_scenes"],
                    }
                    for metric in METRICS:
                        values = [row[metric] for row in rows if row.get(metric) is not None]
                        if values and not np.all(np.isfinite(values)):
                            raise ValueError(f"Non-finite metric {metric}")
                        output[metric] = float(np.mean(values)) if values else None
                        output[metric + "_seed_sd"] = float(np.std(values)) if values else None
                        output[metric + "_defined_models"] = len(values)
                    result.append(output)
    return result


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A7B2C0",
            "text.color": "#1D3044",
            "axes.labelcolor": "#1D3044",
            "xtick.color": "#46556A",
            "ytick.color": "#46556A",
            "savefig.facecolor": "white",
        }
    )


def plot_comparison(run: dict, aggregated: list[dict], destination: Path) -> None:
    lookup = {(r["condition"], r["layout"], r["prefix_poses"], r["method"]): r for r in aggregated}
    final_prefix = max(run["prefixes"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), gridspec_kw={"hspace": 0.42, "wspace": 0.27})
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.06, right=0.98)
    fig.suptitle(
        "3D RF support reconstruction: measurement design and learned models",
        y=0.97,
        fontsize=19,
        fontweight="bold",
    )
    n_scenes = len(run["scene_hashes"])
    n_families = len(set(run["provenance"]["splits"]["test"]["mirror_pair_ids"]))
    fig.text(
        0.5,
        0.925,
        f"Synthetic discrete point scatterers | nominal unit strengths | {n_scenes} held-out scenes "
        f"({n_families} mirror families)",
        ha="center",
        fontsize=11,
        color="#526579",
    )
    fig.text(
        0.5,
        0.89,
        f"Networks: mean of all {len(run['seeds'])} registered initialization seeds "
        f"{list(run['seeds'])}; error bars show descriptive seed SD.",
        ha="center",
        fontsize=10,
        color="#526579",
    )
    fig.legend(
        handles=[
            Patch(color="#90A3B7", label="Same-height acquisition"),
            Patch(color="#008B7A", label="Height-diverse acquisition"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    positions = np.arange(len(METHODS))
    for column, metric in enumerate(PLOT_METRICS):
        ax = axes[0, column]
        for index, (layout, color) in enumerate(zip(LAYOUTS, ("#90A3B7", "#008B7A"), strict=True)):
            rows = [lookup["nominal", layout, final_prefix, method] for method in METHODS]
            values = np.array([r[metric] if r[metric] is not None else np.nan for r in rows])
            errors = np.array([r[metric + "_seed_sd"] or 0.0 for r in rows])
            bars = ax.bar(
                positions + (index - 0.5) * 0.36,
                values,
                width=0.34,
                color=color,
                yerr=errors,
                capsize=2,
                error_kw={"lw": 0.8, "ecolor": "#27384B"},
            )
            for bar, value in zip(bars, values, strict=True):
                if np.isfinite(value) and (value > 0 or metric != "query_rf_nmse_soft"):
                    ax.annotate(
                        f"{value:.2g}",
                        (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4 + 8 * index),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color="#35475B",
                    )
        ax.set_title(f"{final_prefix} acquisition poses", loc="left", fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[column])
        ax.set_xticks(positions, SHORT_LABELS, fontsize=8)
        ax.grid(axis="y", color="#DCE3EA", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        if metric == "query_rf_nmse_soft":
            ax.set_yscale("log")
            ax.margins(y=0.25)
        elif metric == "iou_3d":
            ax.set_ylim(0, 1.12)
        else:
            ax.set_ylim(bottom=0)
            ax.margins(y=0.2)

        lower = axes[1, column]
        for method in METHODS:
            rows = [lookup["nominal", "height_diverse", prefix, method] for prefix in run["prefixes"]]
            values = np.array([r[metric] if r[metric] is not None else np.nan for r in rows])
            sd = np.array([r[metric + "_seed_sd"] or 0.0 for r in rows])
            lower.plot(
                run["prefixes"],
                values,
                marker="o",
                markersize=4,
                lw=2.0 if method == "geometry_rf" else 1.5,
                color=COLORS[method],
            )
            if method in ("geometry", "geometry_rf"):
                minimum = np.maximum(values - sd, 1e-12) if metric == "query_rf_nmse_soft" else values - sd
                lower.fill_between(run["prefixes"], minimum, values + sd, color=COLORS[method], alpha=0.12)
        lower.set_title("Acquisition prefix: height-diverse layout", loc="left", fontsize=11)
        lower.set_xlabel("Available acquisition poses")
        lower.set_ylabel(METRIC_LABELS[column])
        lower.set_xticks(run["prefixes"])
        lower.grid(color="#DCE3EA", linewidth=0.6, alpha=0.8)
        if metric == "query_rf_nmse_soft":
            lower.set_yscale("log")
        elif metric == "iou_3d":
            lower.set_ylim(0, 1.03)
        else:
            lower.set_ylim(bottom=0)
    fig.legend(
        handles=[Line2D([0], [0], color=COLORS[m], marker="o", label=METHOD_LABELS[m]) for m in METHODS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.06,
        0.032,
        "RF query: the same 8 unseen height-diverse poses for every method. "
        "Thresholds and baseline settings use validation only.",
        fontsize=9,
        color="#526579",
    )
    fig.text(
        0.06,
        0.012,
        "Static support experiment; no continuum furniture, multipath, real-data or future-motion claim. "
        "Mismatch results are reported separately in the tables.",
        fontsize=8.5,
        color="#526579",
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def select_examples(run: dict) -> dict:
    seed = 7 if 7 in run["seeds"] else min(run["seeds"])
    key = ("nominal", "height_diverse", max(run["prefixes"]), "geometry_rf", seed)
    rows = [r for r in run["details"] if cell_key(r) == key]
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family_id"], []).append(row)
    families = [
        (family, float(np.mean([r["iou_3d"] for r in members]))) for family, members in by_family.items()
    ]
    median_value = float(np.median([value for _, value in families]))
    median = min(families, key=lambda x: (abs(x[1] - median_value), x[0]))
    worst = min(families, key=lambda x: (x[1], x[0]))
    selected = []
    for label, (family, value) in (("Median family", median), ("Worst family", worst)):
        member = min(by_family[family], key=lambda r: r["scene_hash"])
        selected.append(
            {
                "label": label,
                "family_id": family,
                "family_mean_iou": value,
                "scene_hash": member["scene_hash"],
                "scene_index": run["scene_hashes"].index(member["scene_hash"]),
            }
        )
    return {
        "condition": "nominal",
        "layout": "height_diverse",
        "prefix_poses": max(run["prefixes"]),
        "seed": seed,
        "ranking_method": "geometry_rf",
        "rule": "Rank mirror families by mean 3D IoU of geometry_rf at the registered seed; "
        "choose the family nearest the median and the worst family, ties by family hash; "
        "display the member with the lexicographically smallest scene hash.",
        "examples": selected,
    }


def plot_examples(run_dir: Path, run: dict, selection: dict, destination: Path) -> None:
    with np.load(run_dir / "dataset.npz", allow_pickle=False) as data:
        truth = data["test_occupancy"]
    grid = run["protocol"]["grid"]
    shape, lower, upper = tuple(grid["shape"]), np.array(grid["lower_m"]), np.array(grid["upper_m"])
    spacing = (upper - lower) / np.array(shape)
    coordinates = np.stack(
        np.meshgrid(*[lower[i] + (np.arange(shape[i]) + 0.5) * spacing[i] for i in range(3)], indexing="ij"),
        axis=-1,
    )
    if truth.shape != (len(run["scene_hashes"]), *shape):
        raise ValueError("Saved test labels disagree with registered grid and scene count")
    seed = selection["seed"]
    methods = (None, "inverse3d", "geometry", "geometry_rf")
    fig = plt.figure(figsize=(16.4, 9.6))
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.04, right=0.92, wspace=0.02, hspace=0.13)
    fig.suptitle("Held-out 3D discrete-scatterer supports", y=0.97, fontsize=20, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        f"Nominal data | 12 height-diverse acquisition poses | neural initialization seed {seed} "
        "| validation-selected thresholds",
        ha="center",
        fontsize=11,
        color="#526579",
    )
    fig.text(
        0.5,
        0.89,
        "Examples are selected by family IoU of Geometry + RF: median family and worst family. "
        "The displayed family member is fixed by its hash.",
        ha="center",
        fontsize=9.5,
        color="#526579",
    )
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as predictions:
        for row_index, example in enumerate(selection["examples"]):
            index = example["scene_index"]
            for column, method in enumerate(methods):
                ax = fig.add_subplot(2, 4, row_index * 4 + column + 1, projection="3d")
                if method is None:
                    occupied = truth[index].astype(bool)
                    title = "Ground truth"
                    metric_text = f"{int(occupied.sum())} occupied scatterer sites"
                else:
                    method_seed = None if method == "inverse3d" else seed
                    key = ("nominal", "height_diverse", 12, method, method_seed)
                    summary = run["cells"][key]
                    array_key = f"nominal__height_diverse__{method}__seed{method_seed}"
                    probability = predictions[array_key][index]
                    occupied = probability >= summary["threshold"]
                    detail = run["detail_map"][key + (example["scene_hash"],)]
                    title = METHOD_LABELS[method]
                    metric_text = f"IoU {detail['iou_3d']:.3f} | query RF {detail['query_rf_nmse_soft']:.3g}"
                points = coordinates[occupied]
                if len(points):
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        c=points[:, 2],
                        cmap="viridis",
                        vmin=lower[2],
                        vmax=upper[2],
                        s=17,
                        marker="s",
                        edgecolors="none",
                        alpha=0.95,
                        depthshade=False,
                    )
                else:
                    ax.text2D(0.5, 0.45, "No occupied sites", transform=ax.transAxes, ha="center")
                ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
                ax.set_box_aspect(upper - lower)
                ax.view_init(elev=24, azim=-58)
                ax.set_xticks([lower[0], (lower[0] + upper[0]) / 2, upper[0]])
                ax.set_yticks([lower[1], (lower[1] + upper[1]) / 2, upper[1]])
                ax.set_zticks([lower[2], (lower[2] + upper[2]) / 2, upper[2]])
                ax.set_xlabel("x (m)", labelpad=-1, fontsize=8)
                ax.set_ylabel("y (m)", labelpad=-1, fontsize=8)
                ax.set_zlabel("z (m)", labelpad=0, fontsize=8)
                ax.tick_params(labelsize=7, pad=0)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.pane.fill = False
                    axis._axinfo["grid"].update(color="#DFE5EC", linewidth=0.5)
                ax.set_title(f"{title}\n{metric_text}", fontsize=10, pad=4)
                if column == 0:
                    ax.text2D(
                        -0.08,
                        0.5,
                        f"{example['label']}\nfamily IoU {example['family_mean_iou']:.3f}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=10,
                        fontweight="bold",
                    )
    cax = fig.add_axes([0.943, 0.25, 0.012, 0.43])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(lower[2], upper[2]), cmap="viridis"), cax=cax
    )
    colorbar.set_label("Scatterer height z (m)", fontsize=10)
    fig.text(
        0.055,
        0.075,
        "Each marker denotes one occupied voxel-center scatterer. These are support estimates, "
        "not continuous surfaces or photorealistic reconstructions.",
        fontsize=9.5,
        color="#526579",
    )
    fig.text(
        0.055,
        0.044,
        "All methods see the same acquisition data for each scene. Query RF is evaluated at "
        "8 unseen positions and is never used as estimator input.",
        fontsize=9.5,
        color="#526579",
    )
    fig.text(
        0.055,
        0.017,
        "Exact scene/family hashes, selection rule and every method result are saved in the report JSON/CSV.",
        fontsize=9,
        color="#526579",
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def display(value, sd=None, count: int = 1) -> str:
    if value is None:
        return "undefined"
    if count > 1 and sd is not None:
        return f"{value:.4g} ± {sd:.2g}"
    return f"{value:.4g}"


def write_tables(run: dict, aggregated: list[dict], selection: dict, run_dir: Path, output: Path) -> None:
    keys = list(aggregated[0])
    with (output / "occupancy3d_all_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(aggregated)
    result = {
        "run_directory": str(run_dir.resolve()),
        "source_base_commit": run["protocol"]["source_base_commit"],
        "status": run["status"],
        "task": run["protocol"]["task"],
        "aggregation": "Equal-weight average of per-seed macro-scene metrics; no seed selection. "
        "Seed SD is descriptive population SD, not a confidence interval.",
        "source_files_sha256": {
            name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
            for name in (
                "protocol.json",
                "status.json",
                "summary.json",
                "per_scene_metrics.json",
                "data_provenance.json",
            )
        },
        "aggregate_cells": aggregated,
        "example_selection": selection,
    }
    (output / "occupancy3d_report_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    splits = run["provenance"]["splits"]
    lines = [
        "# Static 3D RF support reconstruction — evaluated synthetic results",
        "",
        f"Run: {run_dir.name}. Status: **completed**; source files unchanged during the run.",
        "",
        "Scope: synthetic, coherent, calibrated monostatic RF from one weak point scatterer per "
        "occupied voxel center. Cuboids describe discrete support. This does not model continuum "
        "furniture surfaces, occlusion, multipath or motion.",
        "",
        f"Data: {splits['train']['scene_count']} training / {splits['validation']['scene_count']} validation / "
        f"{splits['test']['scene_count']} test configurations. Mirror families remain in one split; "
        f"duplicate geometry count: {run['provenance']['duplicate_scene_hash_count']}.",
        "",
        f"Random initialization seeds: {list(run['seeds'])}. All registered seeds contribute to the "
        "network averages below. Values after ± are descriptive seed SD, not confidence intervals. "
        "Baselines have one deterministic result per cell.",
        "",
        "Both acquisition layouts are evaluated on the same 8 unseen height-diverse query poses. "
        "All checkpoint, threshold and baseline selection uses validation data.",
        "",
        "## Nominal comparison figure",
        "",
        "![Comparison](occupancy3d_comparison.png)",
        "",
        "## Example selection",
        "",
        selection["rule"],
        "",
        "![Examples](occupancy3d_examples.png)",
        "",
        "| Example | Family mean IoU | Scene hash |",
        "|---|---:|---|",
    ]
    for example in selection["examples"]:
        lines.append(f"| {example['label']} | {example['family_mean_iou']:.5f} | {example['scene_hash']} |")
    for condition in CONDITIONS:
        lines.extend(["", f"## {condition.replace('_', ' ').title()}", ""])
        if condition == "susceptibility_mismatch":
            lines.extend(
                [
                    "Relative scatterer strengths are lognormal with the registered perturbation. "
                    "Models and thresholds remain frozen from nominal training/validation. "
                    "This is a synthetic forward-model mismatch test.",
                    "",
                ]
            )
        for layout in LAYOUTS:
            lines.extend(
                [
                    f"### {layout.replace('_', ' ').title()}",
                    "",
                    "| Poses | Method | 3D IoU ↑ | Height RMSE m ↓ | Query RF NMSE ↓ | Precision | Recall | Brier ↓ |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in aggregated:
                if (row["condition"], row["layout"]) != (condition, layout):
                    continue
                cols = [str(row["prefix_poses"]), METHOD_LABELS[row["method"]]]
                for metric in (
                    "iou_3d",
                    "height_column_rmse_m",
                    "query_rf_nmse_soft",
                    "precision",
                    "recall",
                    "brier",
                ):
                    cols.append(display(row[metric], row[metric + "_seed_sd"], row["model_count"]))
                lines.append("| " + " | ".join(cols) + " |")
    lines.extend(
        [
            "",
            "## Provenance and timing",
            "",
            f"Source base: {run['protocol']['source_base_commit']}.",
            "",
            f"Total recorded study duration: {run['status']['elapsed_s']:.1f} s. "
            f"Runtime: {run['protocol']['runtime']['device']}, "
            f"{run['protocol']['runtime']['threads']} Torch threads.",
            "",
            "Batched network inference timing excludes feature generation. Baseline and network "
            "timings therefore describe their respective computation blocks, not a matched "
            "end-to-end real-time comparison.",
            "",
            "The CSV contains every aggregate metric for every condition/layout/prefix/method. "
            "The JSON retains exact source hashes, seed counts, aggregate cells and example-selection metadata. "
            "The original run retains per-seed and per-scene records, training logs, thresholds, "
            "data and loadable checkpoints.",
            "",
        ]
    )
    (output / "occupancy3d_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = load_completed_run(args.run_dir)
    aggregated = aggregate_cells(run)
    selection = select_examples(run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    plot_comparison(run, aggregated, args.output_dir / "occupancy3d_comparison.png")
    plot_examples(args.run_dir, run, selection, args.output_dir / "occupancy3d_examples.png")
    write_tables(run, aggregated, selection, args.run_dir, args.output_dir)
    print(
        json.dumps(
            {
                "status": "completed",
                "output_directory": str(args.output_dir.resolve()),
                "aggregate_cells": len(aggregated),
                "registered_seeds": list(run["seeds"]),
                "example_seed": selection["seed"],
            }
        )
    )


if __name__ == "__main__":
    main()
