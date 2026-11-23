"""Report the completed fresh-holdout RF refinement experiment without retraining."""

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

METHODS = ("geometry_rf_frozen", "geometry_rf_refined", "inverse3d_matched_steps", "inverse3d_converged")
LABELS = {"geometry_rf_frozen": "Frozen neural map", "geometry_rf_refined": "Neural map + RF correction",
          "inverse3d_matched_steps": "Zero start, matched RF steps",
          "inverse3d_converged": "Validation-selected FISTA"}
COLORS = dict(zip(METHODS, ("#2674B7", "#008B7A", "#E29732", "#596A7D"), strict=True))
CONDITIONS = ("nominal", "susceptibility_mismatch")
METRICS = ("iou_3d", "precision", "recall", "brier", "binary_nll", "ece", "height_column_rmse_m",
           "height_column_mae_m", "surface_chamfer_m", "surface_f1", "empty_predicted_surface",
           "missed_columns", "false_positive_columns", "query_rf_nmse_soft", "query_rf_nmse_hard",
           "query_noisy_rf_nmse_soft", "observed_rf_nmse_soft", "standalone_batch_compute_s")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(row):
    return row["condition"], row["prefix_poses"], row["method"], row["seed"]


def load_run(path):
    status, protocol = read(path / "status.json"), read(path / "protocol.json")
    checks = ("source_unchanged_during_run", "checkpoint_weights_unchanged", "validation_selection_unchanged")
    if status.get("status") != "completed" or not all(status.get(name) is True for name in checks):
        raise RuntimeError("Refinement run is incomplete or its frozen provenance changed")
    provenance = read(path / "data_provenance.json")
    if provenance["overlap_count"] != 0:
        raise ValueError("Fresh test overlaps development data")
    seeds = tuple(sorted(int(seed) for seed in protocol["frozen_checkpoints"]))
    rows = read(path / "summary.json")
    cells = {key(row): row for row in rows}
    expected = {(c, p, m, s) for c in CONDITIONS for p in protocol["prefixes"] for m in METHODS
                for s in ((None,) if m == "inverse3d_converged" else seeds)}
    if set(cells) != expected or len(cells) != len(rows):
        raise ValueError("Unexpected, missing or duplicate refinement cells")
    details = read(path / "per_scene_metrics.json")
    hashes = provenance["fresh_test"]["scene_hashes"]
    detail_map = {(key(row), row["scene_hash"]): row for row in details}
    if len(detail_map) != len(details) or len(details) != len(cells) * len(hashes):
        raise ValueError("Duplicate or missing per-scene result")
    if any((cell, scene) not in detail_map for cell in cells for scene in hashes):
        raise ValueError("Incomplete per-scene coverage")
    return dict(status=status, protocol=protocol, provenance=provenance, seeds=seeds, cells=cells,
                details=details, detail_map=detail_map, hashes=hashes,
                comparisons=read(path / "paired_comparisons.json"),
                selection=read(path / "validation_selection.json"))


def aggregate(run):
    output = []
    for condition in CONDITIONS:
        for prefix in run["protocol"]["prefixes"]:
            for method in METHODS:
                seeds = (None,) if method == "inverse3d_converged" else run["seeds"]
                rows = [run["cells"][condition, prefix, method, seed] for seed in seeds]
                row = dict(condition=condition, prefix_poses=prefix, method=method,
                           contributing_configs=len(rows), seeds=list(seeds),
                           test_scenes_per_config=rows[0]["test_scenes"])
                for metric in METRICS:
                    values = [r[metric] for r in rows if r.get(metric) is not None]
                    if values and not np.all(np.isfinite(values)):
                        raise ValueError(f"Non-finite {metric}")
                    row[metric] = float(np.mean(values)) if values else None
                    row[metric + "_sd"] = float(np.std(values)) if values else None
                output.append(row)
    return output


def plot_comparison(run, rows, path):
    lookup = {(r["condition"], r["prefix_poses"], r["method"]): r for r in rows}
    prefixes = run["protocol"]["prefixes"]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.79, bottom=0.19, hspace=0.48, wspace=0.30)
    fig.suptitle("Frozen learned maps + RF correction on fresh 3D scenes", y=0.97,
                 fontsize=19, fontweight="bold")
    fig.text(0.5, 0.925,
             f"{len(run['hashes'])} unseen configurations / {run['status']['independent_scene_families']} "
             f"mirror families | excluded all original splits | frozen seeds {list(run['seeds'])}",
             ha="center", fontsize=11, color="#526579")
    selected_steps = sorted({v["selected"]["iterations"] for v in run["selection"].values()})
    fig.text(0.5, 0.885, f"Correction budgets selected on original validation: {selected_steps} steps. "
             "The zero-start control uses the identical budgets and ridge.",
             ha="center", fontsize=10, color="#526579")
    plot_metrics = ("iou_3d", "query_rf_nmse_soft", "standalone_batch_compute_s")
    labels = ("3D IoU ↑", "Unseen-pose RF NMSE ↓ (log)", "Measured CPU batch seconds (log)")
    for i, condition in enumerate(CONDITIONS):
        for j, metric in enumerate(plot_metrics):
            ax = axes[i, j]
            for method in METHODS:
                values = np.asarray([lookup[condition, prefix, method][metric] for prefix in prefixes])
                sd = np.asarray([lookup[condition, prefix, method][metric + "_sd"] for prefix in prefixes])
                ax.plot(prefixes, values, color=COLORS[method], marker="o", markersize=4,
                        linestyle="--" if method == "inverse3d_converged" else "-",
                        linewidth=2.2 if method == "geometry_rf_refined" else 1.6)
                if method in ("geometry_rf_frozen", "geometry_rf_refined") and j < 2:
                    ax.fill_between(prefixes, np.maximum(values - sd, 1e-10), values + sd,
                                    color=COLORS[method], alpha=0.12)
            ax.set_title("Nominal scatterer strengths" if i == 0 else "Paired scatterer-strength mismatch",
                         loc="left", fontsize=11, fontweight="bold")
            ax.set_ylabel(labels[j])
            ax.set_xticks(prefixes)
            ax.set_xlabel("Available acquisition poses")
            ax.grid(color="#DFE5EC", linewidth=0.6)
            if j:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0.22, 1.04)
    fig.legend(handles=[Line2D([0], [0], color=COLORS[m], marker="o",
                               linestyle="--" if m == "inverse3d_converged" else "-", label=LABELS[m])
                        for m in METHODS], loc="lower center", bbox_to_anchor=(0.5, 0.105),
               ncol=4, frameon=False, fontsize=10)
    fig.text(0.065, 0.072, "Neural means include both seeds; shading is descriptive seed SD. "
             "Zero-start controls are averaged over matched configurations; the FISTA reference is counted once.",
             fontsize=9, color="#526579")
    fig.text(0.065, 0.043, "Timing is one measured standalone CPU batch of 192 scenes, including RF features "
             "and network inference where used; no acquisition or I/O. It is not single-scene latency.",
             fontsize=9, color="#526579")
    fig.text(0.065, 0.014, "Synthetic discrete point-scatterer support. Both stress conditions share AWGN draws. "
             "Refined fields are bounded support estimates, not calibrated posterior probabilities.",
             fontsize=9, color="#526579")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def example_selection(run):
    by_family = {}
    for row in run["details"]:
        if (row["condition"], row["prefix_poses"], row["method"]) == ("nominal", 6, "geometry_rf_refined"):
            by_family.setdefault(row["family_id"], []).append(row)
    values = [(family, float(np.mean([r["iou_3d"] for r in members])))
              for family, members in by_family.items()]
    median = float(np.median([v for _, v in values]))
    selected = (("Median family", min(values, key=lambda x: (abs(x[1] - median), x[0]))),
                ("Worst family", min(values, key=lambda x: (x[1], x[0]))))
    return [{"label": label, "family_id": family, "family_mean_iou": value,
             "scene_hash": min(r["scene_hash"] for r in by_family[family])}
            for label, (family, value) in selected]


def plot_examples(run_dir, run, examples, path):
    with np.load(run_dir / "fresh_dataset.npz", allow_pickle=False) as source:
        truth = source["occupancy"]
    grid = run["protocol"]["grid"]
    shape, lower, upper = np.array(grid["shape"]), np.array(grid["lower_m"]), np.array(grid["upper_m"])
    spacing = (upper - lower) / shape
    xyz = np.stack(np.meshgrid(*[lower[d] + (np.arange(shape[d]) + 0.5) * spacing[d]
                                 for d in range(3)], indexing="ij"), axis=-1)
    seed = 7 if 7 in run["seeds"] else min(run["seeds"])
    methods = (None, "geometry_rf_frozen", "geometry_rf_refined", "inverse3d_converged")
    fig = plt.figure(figsize=(16.2, 9.0))
    fig.subplots_adjust(left=0.04, right=0.92, top=0.82, bottom=0.12, hspace=0.15, wspace=0.02)
    fig.suptitle("Six-pose reconstruction on fresh synthetic 3D supports", y=0.97, fontsize=19, fontweight="bold")
    fig.text(0.5, 0.92, f"Frozen seed {seed} shown | median and worst families ranked by refined IoU "
             "averaged across both seeds | member selected by hash", ha="center", fontsize=10.5, color="#526579")
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as predictions:
        for row, example in enumerate(examples):
            index = run["hashes"].index(example["scene_hash"])
            for column, method in enumerate(methods):
                ax = fig.add_subplot(2, 4, row * 4 + column + 1, projection="3d")
                if method is None:
                    occupied = truth[index].astype(bool)
                    title = f"Ground truth\n{occupied.sum()} occupied sites"
                else:
                    model_seed = None if method == "inverse3d_converged" else seed
                    cell = ("nominal", 6, method, model_seed)
                    threshold = run["cells"][cell]["threshold"]
                    occupied = predictions[f"nominal__prefix6__{method}__seed{model_seed}"][index] >= threshold
                    detail = run["detail_map"][cell, example["scene_hash"]]
                    title = f"{LABELS[method]}\nIoU {detail['iou_3d']:.3f} | query RF {detail['query_rf_nmse_soft']:.3g}"
                points = xyz[occupied]
                if len(points):
                    ax.scatter(*points.T, c=points[:, 2], cmap="viridis", vmin=lower[2], vmax=upper[2],
                               marker="s", s=18, edgecolors="none", depthshade=False)
                else:
                    ax.text2D(0.5, 0.45, "No occupied sites", transform=ax.transAxes, ha="center")
                ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
                ax.set_box_aspect(upper - lower)
                ax.view_init(elev=24, azim=-58)
                for d, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
                    axis.set_ticks([lower[d], (lower[d] + upper[d]) / 2, upper[d]])
                    axis.pane.fill = False
                    axis._axinfo["grid"].update(color="#DFE5EC", linewidth=0.5)
                ax.set_xlabel("x (m)", labelpad=-1, fontsize=8)
                ax.set_ylabel("y (m)", labelpad=-1, fontsize=8)
                ax.set_zlabel("z (m)", labelpad=0, fontsize=8)
                ax.tick_params(labelsize=7, pad=0)
                ax.set_title(title, fontsize=10, pad=3)
                if column == 0:
                    ax.text2D(-0.07, 0.5, f"{example['label']}\nfamily IoU {example['family_mean_iou']:.3f}",
                              rotation=90, transform=ax.transAxes, va="center", ha="center", fontweight="bold")
    cax = fig.add_axes([0.945, 0.25, 0.012, 0.40])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lower[2], upper[2]), cmap="viridis"),
                 cax=cax, label="Scatterer height z (m)")
    fig.text(0.055, 0.075, "All maps use validation-selected thresholds. RF correction uses acquisition measurements "
             "only; the common unseen-position RF query remains evaluation-only.", fontsize=9.5, color="#526579")
    fig.text(0.055, 0.041, "Each marker is an occupied voxel-center scatterer; these are discrete support reconstructions. "
             "Refined continuous values do not constitute calibrated posterior probabilities.", fontsize=9.5, color="#526579")
    fig.text(0.055, 0.014, "Median/worst family selection and exact scene hashes are recorded in the report JSON. "
             "The FISTA reference uses its original validation-selected budget.", fontsize=9, color="#526579")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(run_dir, output, run, rows, examples):
    with (output / "occupancy3d_refinement_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = run["comparisons"]["comparisons"]
    with (output / "occupancy3d_refinement_pairwise.csv").open("w", encoding="utf-8", newline="") as file:
        fields = [k for k in comparisons[0] if k != "per_initialization"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)
    report = {"status": run["status"], "aggregate_cells": rows, "paired_comparisons": run["comparisons"],
              "example_selection": {"prefix": 6, "condition": "nominal", "ranking": "refined family mean over both seeds",
                                    "display_seed": 7 if 7 in run["seeds"] else min(run["seeds"]),
                                    "member_rule": "lexicographically smallest scene hash", "examples": examples},
              "timing_scope": run["protocol"]["timing"],
              "note": "Stored inverse3d_converged denotes the validation-selected fixed-budget reference; "
                      "optimization convergence is not certified. Refined fields are continuous bounded support estimates.",
              "source_sha256": {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                                for name in ("protocol.json", "status.json", "summary.json", "paired_comparisons.json",
                                             "data_provenance.json", "validation_selection.json")}}
    (output / "occupancy3d_refinement_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Fresh-holdout 3D RF refinement results", "",
             f"Completed: {len(run['hashes'])} fresh configurations, {run['status']['independent_scene_families']} "
             f"independent mirror families, {len(run['cells'])} evaluation cells. "
             f"Original-scene overlap: {run['provenance']['overlap_count']}.", "",
             "The original study motivated this supplementary hypothesis. Frozen checkpoints were not retrained. "
             "Correction budgets and thresholds were selected on original validation; fresh families were evaluated afterward. "
             "All source, checkpoint and selection hashes remained unchanged.", "",
             "Synthetic discrete weak point scatterers; height-diverse acquisition. The nominal and strength-mismatch "
             "conditions share acquisition/query AWGN draws. All methods use the same unseen query positions.", "",
             "![Comparison](occupancy3d_refinement_comparison.png)", "",
             "![Six-pose examples](occupancy3d_refinement_examples.png)", "",
             "Neural means average both trained seeds. Matched-step zero-start controls are averaged over the two "
             "matched configurations; identical settings are not independent training seeds. The long-budget FISTA "
             "reference is counted once. Its stored method key does not certify mathematical convergence.", ""]
    for condition in CONDITIONS:
        lines.extend([f"## {condition.replace('_', ' ').title()}", "",
                      "| Poses | Method | IoU ↑ | Query RF NMSE ↓ | Height RMSE m ↓ | Brier ↓ | ECE ↓ | CPU batch s |",
                      "|---:|---|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            if row["condition"] == condition:
                fields = [str(row["prefix_poses"]), LABELS[row["method"]]]
                fields.extend(f"{row[m]:.5g}" if row[m] is not None else "undefined" for m in
                              ("iou_3d", "query_rf_nmse_soft", "height_column_rmse_m", "brier", "ece",
                               "standalone_batch_compute_s"))
                lines.append("| " + " | ".join(fields) + " |")
        lines.append("")
    lines.extend(["## Paired uncertainty", "", run["comparisons"]["interval_scope"], "",
                  "All 90 metric comparisons are retained in the paired CSV and report JSON. "
                  "Intervals condition on the two frozen initializations and resample whole mirror families. "
                  "Continuous RF-refined support values are not calibrated posterior probabilities; "
                  "Brier/ECE are reported diagnostically alongside geometry and RF.", "",
                  "## Timing", "", "Single measured CPU batch, including feature construction and frozen CNN "
                  "inference where used, plus inverse updates. No I/O or acquisition time is included. "
                  "This is not an end-to-end per-scene latency benchmark; matched inverse steps do not imply "
                  "equal total computation.", ""])
    (output / "occupancy3d_refinement_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = load_run(args.run_dir)
    rows, examples = aggregate(run), example_selection(run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#1D3044",
                         "axes.labelcolor": "#1D3044", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#A7B2C0", "savefig.facecolor": "white"})
    plot_comparison(run, rows, args.output_dir / "occupancy3d_refinement_comparison.png")
    plot_examples(args.run_dir, run, examples, args.output_dir / "occupancy3d_refinement_examples.png")
    write_outputs(args.run_dir, args.output_dir, run, rows, examples)
    print(json.dumps({"status": "completed", "aggregate_cells": len(rows),
                      "comparison_count": run["comparisons"]["comparison_count"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
