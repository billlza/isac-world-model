"""Re-evaluate a completed-but-failed research study without modifying its evidence.

Only accepts an original manifest whose recorded sole source modification was
CLI-only, whose checkpoint hashes match, and whose full saved evaluations
reproduce numerically on the current frozen estimator/model sources.
CPU-timing metrics are intentionally excluded from numerical equality checks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from compare_estimators import run_estimator, sha, sources

from isac_wm.artifacts import write_json
from isac_wm.scenarios import scenario_config
from isac_wm.simulator import SceneConfig
from isac_wm.training import load_model, runtime_info


def verify(input_path: Path, out_path: Path) -> dict:
    source = input_path / "comparison.json"
    old = json.loads(source.read_text())
    if old["status"] != "failed" or old.get("error") != "ValueError: Source changed during comparison":
        raise ValueError("Original study must retain its source-drift failure")
    current = sources()
    changes = [k for k in old["source_sha256"] if old["source_sha256"][k] != current.get(k)]
    if changes != ["src/isac_wm/cli.py"] or set(old["source_sha256"]) != set(current):
        raise ValueError(f"Unsupported changes relative to original study: {changes}")
    protocol = old["protocol"]
    if len(old["scores"]) != len(protocol["train_seeds"]) * len(protocol["scenarios"]):
        raise ValueError("Original study incomplete")
    torch.set_num_threads(2)
    models = {}
    for entry in old["training"]:
        checkpoint = input_path / f"{entry['architecture']}-{entry['seed']}" / "model.pt"
        if sha(checkpoint) != entry["model_sha256"]:
            raise ValueError(f"Checkpoint has changed: {checkpoint}")
        models[(entry["seed"], entry["architecture"])] = load_model(checkpoint, "cpu")[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError("Verification output already exists")
    report = {
        "status": "running",
        "original_study_sha256": sha(source),
        "original_status": old["status"],
        "source_changes": changes,
        "source_sha256_current": current,
        "execution": runtime_info(torch.device("cpu")),
        "recomputed_rows": 0,
        "verified_episodes": 0,
        "maximum_abs_position_rmse_difference_m": 0.0,
        "interpretation": (
            "Only CLI presentation options changed during the original study. "
            "The ORIGINAL manifest remains failed; this new independent verifier "
            "recomputes all recorded held-out metrics using the frozen model and "
            "evaluator sources. CPU latency is remeasured separately."
        ),
    }
    write_json(out_path, report)
    try:
        tuning = old["baseline_validation"]
        for row in old["scores"]:
            scene = scenario_config(row["scenario"], SceneConfig(steps=48))
            for method, recorded in row["per_episode"].items():
                model = (
                    models[(row["training_seed"], method)]
                    if method in {"structured", "structured-attention"}
                    else None
                )
                for item in recorded:
                    fresh = run_estimator(
                        method,
                        seed=item["seed"],
                        scene=scene,
                        compute_ms=row["compute_ms"],
                        model=model,
                        kf_process_noise=tuning["kalman"]["chosen"],
                        imm_switch_rate=tuning["imm"]["chosen"],
                    )
                    for key in (
                        "rmse_m",
                        "p95_error_m",
                        "maximum_error_m",
                        "sensing_airtime",
                        "communication_bits_proxy",
                        "rejected",
                        "no_output_fraction",
                    ):
                        if not math.isclose(fresh[key], item[key], abs_tol=1e-7, rel_tol=1e-7):
                            raise AssertionError(
                                f"Mismatch {row['scenario']}/{row['compute_ms']}ms, "
                                f"{row['training_seed']}/{method}/{item['seed']} {key}: "
                                f"{fresh[key]} vs {item[key]}"
                            )
                    report["maximum_abs_position_rmse_difference_m"] = max(
                        report["maximum_abs_position_rmse_difference_m"],
                        abs(fresh["rmse_m"] - item["rmse_m"]),
                    )
                    report["verified_episodes"] += 1
            report["recomputed_rows"] += 1
            write_json(out_path, report)
        report["status"] = "verified_model_and_metric_stability_under_cli_only_change"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(out_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.input, args.out)
    print(
        json.dumps({"status": result["status"], "verified_episodes": result["verified_episodes"]}, indent=2)
    )


if __name__ == "__main__":
    main()
