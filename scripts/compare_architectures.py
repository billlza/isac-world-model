"""Paired train-from-scratch architecture study; no hidden downloads or test selection.

All variants use the same training episodes, optimization steps and held-out tests.
Parameter counts and compute differ and are reported, NOT called budget-matched.
Selection uses validation RMSE before any test evaluation; test outcomes never tune it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from isac_wm.artifacts import write_json
from isac_wm.evaluation import evaluate
from isac_wm.model import ModelConfig
from isac_wm.runtime import RuntimeConfig
from isac_wm.scenarios import scenario_config
from isac_wm.simulator import SceneConfig
from isac_wm.structured import StructuredConfig
from isac_wm.training import load_model, runtime_info, train
from isac_wm.validation import integer

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("gru", "structured", "structured-attention")
CASES = (("nominal", 0), ("nominal", 150), ("camera_loss", 150), ("radio_outliers", 150), ("maneuver", 150))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict:
    files = sorted((ROOT / "src/isac_wm").glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def compare(out: Path, *, seeds=(7, 11, 13), episodes=64, epochs=16, test_episodes=8, quick=False) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct training seeds")
    for seed in seeds:
        integer(seed, "seed", 0, 99)
    for value, name in ((episodes, "episodes"), (epochs, "epochs"), (test_episodes, "test_episodes")):
        integer(value, name, 1, 100_000)
    if test_episodes > 10_000:
        raise ValueError("test_episodes exceeds evaluation limit")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    initial_source = sources()
    scene = SceneConfig(steps=12 if quick else 48)
    f, h = (16, 32) if quick else (32, 64)
    if quick:
        episodes, epochs, test_episodes = 4, 1, 1
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "pipeline smoke only" if quick else "small synthetic CPU architecture study",
        "runtime": runtime_info(torch.device("cpu")),
        "source_sha256": initial_source,
        "protocol": {
            "architectures": ARCHITECTURES,
            "training_seeds": list(seeds),
            "episodes": episodes,
            "epochs": epochs,
            "features": f,
            "hidden": h,
            "test_episodes": test_episodes,
            "test_seed_start": 60_000_000,
            "policy": "fixed",
            "selection": "mean best-validation RMSE across seeds, before held-out tests",
            "matching": "same data, epochs, loss, optimizer, batch size; NOT equal parameters or compute",
        },
        "training": [],
        "tests": [],
        "limitations": [
            "synthetic orthographic images and processed radio features, not real radar/SLAM",
            "CV Kalman baseline, not IMM or all modern tracking algorithms",
            "fixed assumed service latency, not hardware real-time certification",
            "no covariance calibration or safety guarantee",
            "same test trajectories are reused across training seeds; these are not independent test sets",
            "structured and attention variants change physical state/visual encoder, not attention alone versus GRU",
        ],
    }
    path = out / "comparison.json"
    write_json(path, report)
    try:
        for architecture in ARCHITECTURES:
            for seed in seeds:
                print(f"train {architecture} seed={seed}", flush=True)
                config = (
                    ModelConfig(f, h)
                    if architecture == "gru"
                    else StructuredConfig(f, h, architecture, 16, 4)
                )
                directory = out / f"{architecture}-{seed}"
                manifest = train(
                    directory,
                    episodes=episodes,
                    epochs=epochs,
                    batch_size=2 if quick else 8,
                    device_name="cpu",
                    seed=seed,
                    scene=scene,
                    model_config=config,
                )
                report["training"].append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "parameters": manifest["parameters"],
                        "training_seconds": manifest["training_seconds"],
                        "best_validation_rmse_m": manifest["best_validation_rmse_m"],
                        "checkpoint_sha256": digest(directory / "model.pt"),
                    }
                )
                write_json(path, report)
        report["validation_ranking"] = sorted(
            [
                {
                    "architecture": architecture,
                    "mean_validation_rmse_m": statistics.mean(
                        x["best_validation_rmse_m"]
                        for x in report["training"]
                        if x["architecture"] == architecture
                    ),
                }
                for architecture in ARCHITECTURES
            ],
            key=lambda item: item["mean_validation_rmse_m"],
        )
        report["selected_by_validation"] = report["validation_ranking"][0]["architecture"]
        write_json(path, report)
        for item in report["training"]:
            directory = out / f"{item['architecture']}-{item['seed']}"
            model, base, device = load_model(directory / "model.pt", "cpu")
            for scenario, compute_ms in CASES:
                print(
                    f"test {item['architecture']} seed={item['seed']} {scenario}/{compute_ms}ms", flush=True
                )
                result = evaluate(
                    directory / f"{scenario}-{compute_ms}ms",
                    model,
                    scenario_config(scenario, base),
                    device,
                    episodes=test_episodes,
                    seed=60_000_000,
                    policies=("fixed",),
                    runtime=RuntimeConfig("fixed", compute_ms),
                )
                report["tests"].append(
                    {
                        "architecture": item["architecture"],
                        "training_seed": item["seed"],
                        "scenario": scenario,
                        "compute_ms": compute_ms,
                        "summary": result["summary"],
                        "per_episode": [
                            {"seed": r["seed"], "estimator": r["estimator"], "metrics": r["metrics"]}
                            for r in result["runs"]
                        ],
                    }
                )
                write_json(path, report)
        if sources() != initial_source:
            raise ValueError("Source changed during architecture comparison")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="7,11,13")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = compare(
        args.out,
        seeds=tuple(int(v) for v in args.seeds.split(",")),
        episodes=args.episodes,
        epochs=args.epochs,
        test_episodes=args.test_episodes,
        quick=args.quick,
    )
    print(
        json.dumps({"status": result["status"], "validation_ranking": result["validation_ranking"]}, indent=2)
    )


if __name__ == "__main__":
    main()
