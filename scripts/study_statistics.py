"""Post-hoc descriptive summaries; NEVER selects checkpoints or hyperparameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from compare_estimators import sources

from isac_wm.artifacts import write_json


def summarize(original: Path, verified: Path | None, out: Path) -> dict:
    report = json.loads(original.read_text(encoding="utf-8"))
    if report["status"] == "passed":
        if report.get("source_sha256") != sources():
            raise ValueError("Study source hash differs from current frozen implementation")
        verification_status = "passed_complete_run_and_source_hash_match"
    elif report["status"] == "failed" and verified is not None:
        receipt = json.loads(verified.read_text(encoding="utf-8"))
        if (
            receipt["status"] != "verified_model_and_metric_stability_under_cli_only_change"
            or receipt.get("original_study_sha256") != hashlib.sha256(original.read_bytes()).hexdigest()
        ):
            raise ValueError("Metric re-evaluation must pass first")
        if len(report["scores"]) != receipt["recomputed_rows"]:
            raise ValueError("Study/verification disagreement")
        verification_status = receipt["status"]
    else:
        raise ValueError("Only completed passed studies or independently verified failed runs accepted")
    if not report["scores"] or len(report["scores"]) != len(report["protocol"]["train_seeds"]) * len(
        report["protocol"]["scenarios"]
    ):
        raise ValueError("Incomplete study")
    if out.exists():
        raise FileExistsError("Never overwrite research evidence")
    random = np.random.default_rng(26)
    rows = []
    for scenario, delay in report["protocol"]["scenarios"]:
        batches = [r for r in report["scores"] if (r["scenario"], r["compute_ms"]) == (scenario, delay)]
        differences = []
        all_errors = {name: [] for name in batches[0]["per_episode"]}
        for batch in batches:
            paired = []
            a = batch["per_episode"]["structured-attention"]
            b = batch["per_episode"]["structured"]
            for aa, bb in zip(a, b, strict=True):
                if aa["seed"] != bb["seed"]:
                    raise ValueError("Paired episode identity changed")
                paired.append(aa["rmse_m"] - bb["rmse_m"])
            differences.append(np.array(paired, dtype=float))
            for name, series in batch["per_episode"].items():
                all_errors[name].extend(x["rmse_m"] for x in series)
        count = len(differences)
        samples = []
        # Hierarchical bootstrap: resample training-seed groups THEN trajectory IDs.
        # With only 3 training seeds this is exploratory, not a coverage guarantee.
        for _ in range(3000):
            groups = random.integers(0, count, size=count)
            samples.append(
                statistics.mean(
                    float(
                        differences[g][
                            random.integers(0, len(differences[g]), size=len(differences[g]))
                        ].mean()
                    )
                    for g in groups
                )
            )
        pooled = np.concatenate(differences)
        rows.append(
            {
                "scenario": scenario,
                "assumed_compute_ms": delay,
                "trajectory_pairs": len(pooled),
                "train_seed_groups": count,
                "rmse_m_per_episode_mean": {name: float(np.mean(vals)) for name, vals in all_errors.items()},
                "attention_minus_structured_mean_m": float(pooled.mean()),
                "hierarchical_bootstrap_95pct": [float(z) for z in np.quantile(samples, [0.025, 0.975])],
                "train_seed_group_differences_m": [float(d.mean()) for d in differences],
            }
        )
    output = {
        "status": "descriptive_analysis_only",
        "original_study_status": report["status"],
        "verification_status": verification_status,
        "training": report["training"],
        "baseline_tuning": report["baseline_validation"],
        "scenarios": rows,
        "limitations": [
            "Source hashes and completed run verified when status=passed; prior failed run remains separate historical evidence",
            "Only 3 training seeds; hierarchical confidence intervals remain exploratory",
            "Fixed synthetic 10m room and radar proxy, not new physical environment families",
            "Attention variants have different parameters, parameter-matching not done",
            "No metric/architecture chosen after seeing test outcomes",
        ],
    }
    write_json(out, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--verification", type=Path, help="Required only for independently rechecked failed studies"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.study, args.verification, args.out)
    print(json.dumps({"status": result["status"], "scenarios": result["scenarios"]}, indent=2))


if __name__ == "__main__":
    main()
