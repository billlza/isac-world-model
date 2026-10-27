"""Paired, independently seeded IMM/robust-KF vs from-scratch world-model study.

One group per training seed, using disjoint *test* trajectories between groups.
Within a group, all estimators see the same scenario, seed, airtime budget,
and assumed compute service time. Baseline q is selected exclusively on a
separate validation group before any test is evaluated. Attention is assessed
against the exact same structured architecture without attention.

Physical sensing remains a documented synthetic distance/bearing proxy. This
is not a claim of real-world superiority or hardware-level quantum ISAC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from isac_wm.artifacts import write_json
from isac_wm.evaluation import rollout
from isac_wm.imm import IMMConfig
from isac_wm.runtime import RuntimeConfig
from isac_wm.scenarios import scenario_config
from isac_wm.simulator import SceneConfig
from isac_wm.structured import StructuredConfig
from isac_wm.training import load_model, runtime_info, train
from isac_wm.validation import integer

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    ("nominal", 0),
    ("nominal", 150),
    ("maneuver", 150),
    ("radio_outliers", 150),
    ("radio_bias", 150),
    ("sparse_camera", 150),
    ("camera_loss", 150),
    ("blackout", 150),
)
IMM_BANK = (0.05, 0.8, 6.0)
VALIDATION_CASES = (("nominal", 0), ("maneuver", 150), ("radio_outliers", 150))
ESTIMATORS = ("kalman", "kalman-gated", "imm", "imm-gated", "structured", "structured-attention")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    tracked = sorted((ROOT / "src/isac_wm").glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): sha(p) for p in tracked}


def run_estimator(
    name: str,
    *,
    seed: int,
    scene: SceneConfig,
    compute_ms: float,
    model=None,
    kf_process_noise: float = 0.8,
    imm_switch_rate: float = 0.6,
    trace: bool = False,
):
    label = "model" if name in {"structured", "structured-attention"} else name
    result = rollout(
        seed,
        scene,
        label,
        "fixed",
        model,
        torch.device("cpu") if model is not None else None,
        runtime=RuntimeConfig("fixed", compute_ms),
        baseline_process_noise=kf_process_noise,
        imm_config=IMMConfig(IMM_BANK, switch_rate_hz=imm_switch_rate),
    )
    if trace:
        return result
    m = result["metrics"]
    return {
        "seed": seed,
        "rmse_m": m["position_rmse_m"],
        "p95_error_m": m["position_p95_m"],
        "maximum_error_m": m["maximum_error_m"],
        "sensing_airtime": m["sensing_airtime_fraction"],
        "communication_bits_proxy": m["communication_bits_proxy"],
        "measured_cpu_p95_ms": m["compute_p95_ms"],
        "rejected": m["gated_radio_or_vision_measurements"],
        "no_output_fraction": m["no_output_fraction"],
    }


def tune_baselines(base: SceneConfig, count: int = 8, quick: bool = False) -> dict:
    """Validation-only q and mode transition selection, never test-adaptive."""
    results = {}
    candidates = {
        "kalman": (0.1, 0.8, 3.0),
        "imm": (0.15, 0.6, 1.5),
    }
    if quick:
        count = 1
    for family, values in candidates.items():
        rows = []
        for value in values:
            errors = []
            for scenario, latency in VALIDATION_CASES:
                scene = scenario_config(scenario, base)
                for i in range(count):
                    name = "kalman-gated" if family == "kalman" and scenario == "radio_outliers" else (
                        "imm-gated" if family == "imm" and scenario == "radio_outliers" else family
                    )
                    r = run_estimator(
                        name, seed=50_000_000 + i, scene=scene, compute_ms=latency,
                        kf_process_noise=value if family == "kalman" else 0.8,
                        imm_switch_rate=value if family == "imm" else 0.6,
                    )
                    errors.append(r["rmse_m"])
            rows.append({"value": value, "validation_mean_episode_rmse_m": statistics.mean(errors)})
        winner = min(rows, key=lambda r: (r["validation_mean_episode_rmse_m"], r["value"]))
        results[family] = {"candidates": rows, "chosen": winner["value"]}
    return results


def paired_bootstrap(differences: list[float], seed=899, draws=2000) -> dict:
    data = np.asarray(differences, dtype=float)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Missing or invalid paired results")
    generator = np.random.default_rng(seed)
    samples = data[generator.integers(0, len(data), (draws, len(data)))].mean(-1)
    return {
        "mean_attention_minus_structured_m": float(data.mean()),
        "percentile95_interval_m": (
            [float(x) for x in np.quantile(samples, [0.025, 0.975])]
            if len(data) >= 2 else None
        ),
        "pairs": len(differences),
        "note": "Exploratory percentile bootstrap over held-out trajectory seeds; no uniform/physical safety guarantee",
    }


def compare(
    out: Path,
    *,
    seeds=(7, 11, 13),
    episodes=96,
    epochs=16,
    test_episodes=24,
    quick=False,
    validate_episodes=8,
    keep_trace=False,
) -> dict:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be nonempty and unique")
    for seed in seeds:
        integer(seed, "training seed", 0, 99)
    for v, name in ((episodes, "episodes"), (epochs, "epochs"), (test_episodes, "test episodes"),
                    (validate_episodes, "validation episodes")):
        integer(v, name, 1, 10000)
    # 10k gap allows unique held-out trajectories for each training seed.
    if test_episodes > 9999:
        raise ValueError("Test groups must be disjoint")
    if quick:
        episodes, epochs, test_episodes, validate_episodes = 4, 1, 1, 1
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    start = time.monotonic()
    source_hashes = sources()
    base = SceneConfig(steps=12 if quick else 48)
    report = {
        "status": "running",
        "scope": "pipeline smoke; not for ranking" if quick else "paired synthetic research comparison",
        "schema_version": 1,
        "runtime": runtime_info(torch.device("cpu")),
        "source_sha256": source_hashes,
        "protocol": {
            "train_seeds": seeds, "train_episodes": episodes, "train_epochs": epochs,
            "validation_seed_start": 50_000_000, "baseline_validation_episodes": validate_episodes,
            "test_seed_starts": [70_000_000 + i * 10_000 for i in range(len(seeds))],
            "test_episodes_per_training_seed": test_episodes,
            "scenarios": SCENARIOS,
            "policy": "fixed",
            "matching": "paired scenarios, same capture/release, sensing and fixed simulated-service budget; not equal parameters, training FLOPs or observed CPU time",
            "architecture_selection": "no test-driven switching or hyperparameter search",
        },
        "baseline_validation": None,
        "training": [],
        "scores": [],
        "attention_ablation": None,
        "limitations": [
            "synthetic orthographic visual and processed polar radio observations; not waveform-level ISAC or real SLAM",
            "IMM noise-intensity bank is fixed; only switch rate chosen with validation; no bank covering arbitrary physics",
            "camera and radio streams are not conditionally independent under all possible calibrations",
            "model and filters have different input processing and compute budgets (reported, not conflated)",
            "no calibrated conditional-risk or deployment safety guarantee",
            "same scenarios across held-out groups; seeds disjoint across groups; not unseen environment families",
            "attention ablation has additional parameters, not parameter-matched",
            "service time is imposed fixed synthetic latency; measured CPU cost separately reported",
        ],
    }
    path = out / "comparison.json"
    write_json(path, report)
    try:
        tuning = tune_baselines(base, validate_episodes, quick)
        report["baseline_validation"] = tuning
        write_json(path, report)
        models = {}
        for seed in seeds:
            for architecture in ("structured", "structured-attention"):
                name = f"{architecture}-{seed}"
                config = StructuredConfig(16 if quick else 32, 32 if quick else 64, architecture, 16, 4)
                print(f"train {name}", flush=True)
                manifest = train(
                    out / name, episodes=episodes, epochs=epochs, batch_size=2 if quick else 8,
                    device_name="cpu", seed=seed, scene=base, model_config=config
                )
                model, _, _ = load_model(out / name / "model.pt", "cpu")
                models[(seed, architecture)] = model
                report["training"].append({
                    "seed": seed, "architecture": architecture,
                    "model_sha256": sha(out / name / "model.pt"),
                    "parameter_count": manifest["parameters"],
                    "training_seconds": manifest["training_seconds"],
                    "best_validation_rmse_m": manifest["best_validation_rmse_m"],
                })
                write_json(path, report)
        for i, seed in enumerate(seeds):
            offset = 70_000_000 + i * 10_000
            for scenario, latency in SCENARIOS:
                print(f"evaluate train_seed={seed} {scenario}/{latency}ms", flush=True)
                scene = scenario_config(scenario, base)
                # Each model's test seeds are fresh relative to other training-seed groups.
                results = {}
                for name in ESTIMATORS:
                    model = models[(seed, name)] if name in ("structured", "structured-attention") else None
                    records = [
                        run_estimator(
                            name, seed=offset+j, scene=scene, compute_ms=latency,
                            model=model, kf_process_noise=tuning["kalman"]["chosen"],
                            imm_switch_rate=tuning["imm"]["chosen"],
                        )
                        for j in range(test_episodes)
                    ]
                    results[name] = records
                    if keep_trace:
                        trace = run_estimator(
                            name, seed=offset, scene=scene, compute_ms=latency, model=model,
                            kf_process_noise=tuning["kalman"]["chosen"],
                            imm_switch_rate=tuning["imm"]["chosen"], trace=True,
                        )
                        out_trace = out / "traces" / str(seed) / f"{scenario}-{latency}ms"
                        out_trace.mkdir(parents=True, exist_ok=True)
                        write_json(out_trace / f"{name}.json", trace)
                report["scores"].append({
                    "training_seed": seed,
                    "scenario": scenario,
                    "compute_ms": latency,
                    "test_seed_start": offset,
                    "per_episode": results,
                    "means": {
                        name: {
                            "rmse_m": statistics.mean(item["rmse_m"] for item in values),
                            "p95_m": statistics.mean(item["p95_error_m"] for item in values),
                            "sensing_airtime": statistics.mean(item["sensing_airtime"] for item in values),
                            "cpu_p95_ms": statistics.mean(item["measured_cpu_p95_ms"] for item in values),
                        }
                        for name, values in results.items()
                    },
                })
                write_json(path, report)
        ablations = {}
        for scene, latency in SCENARIOS:
            rows = [r for r in report["scores"] if (r["scenario"],r["compute_ms"]) == (scene,latency)]
            differences = [
                a["rmse_m"] - b["rmse_m"]
                for row in rows
                for a,b in zip(row["per_episode"]["structured-attention"], row["per_episode"]["structured"], strict=True)
            ]
            ablations[f"{scene}-{latency}ms"] = paired_bootstrap(differences)
        report["attention_ablation"] = ablations
        if sources() != source_hashes:
            raise ValueError("Source changed during comparison")
        report["duration_seconds"] = time.monotonic() - start
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="7,11,13")
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument("--validate-episodes", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--keep-trace", action="store_true")
    args = parser.parse_args()
    result = compare(
        args.out,
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        episodes=args.episodes,
        epochs=args.epochs,
        test_episodes=args.test_episodes,
        validate_episodes=args.validate_episodes,
        quick=args.quick,
        keep_trace=args.keep_trace,
    )
    print(json.dumps({"status": result["status"], "attention_ablation": result["attention_ablation"]}, indent=2))


if __name__ == "__main__":
    main()
