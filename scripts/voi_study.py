"""Paired real-timing simulation for the Gaussian arrival-time VoI comparator.

This is not hyperparameter selection, independent hardware verification, or a
claim of improved error. All methods receive paired synthetic truth/noise keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from isac_wm.evaluation import rollout
from isac_wm.runtime import RuntimeConfig
from isac_wm.scenarios import scenario_config
from isac_wm.simulator import SceneConfig
from isac_wm.voi import VoIConfig

SOURCE_FILES = (
    "src/isac_wm/voi.py",
    "src/isac_wm/evaluation.py",
    "src/isac_wm/streaming.py",
    "src/isac_wm/simulator.py",
    "scripts/voi_study.py",
)


def run_study(out: Path, *, episodes: int = 12, steps: int = 48, seed: int = 80_000_000,
              voi_tradeoff: float = 0.4) -> dict:
    if not 1 <= episodes <= 500 or not 8 <= steps <= 10_000 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Invalid study dimensions")
    voi_config = VoIConfig(tradeoff_m2_per_mbit=voi_tradeoff)
    if out.exists():
        raise FileExistsError(f"Study output exists: {out}")
    out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    configurations = ("nominal", "maneuver", "radio_outliers", "radio_bias", "camera_loss")
    policies = ("fixed", "guarded", "skip", "voi")
    estimators = ("kalman", "imm-gated")
    runtime = RuntimeConfig("fixed", 150.0)
    experiments = {}
    for scenario in configurations:
        config = scenario_config(scenario, SceneConfig(steps=steps))
        for estimator in estimators:
            per_policy = {}
            for policy in policies:
                records = []
                for i in range(episodes):
                    result = rollout(seed + i, config, estimator, policy, runtime=runtime, voi_config=voi_config)
                    m = result["metrics"]
                    records.append({
                        "seed": seed + i,
                        "position_rmse_m": m["position_rmse_m"],
                        "position_p95_m": m["position_p95_m"],
                        "sensing_airtime_fraction": m["sensing_airtime_fraction"],
                        "communication_bits_proxy": m["communication_bits_proxy"],
                        "action_skip_count": m["action_skip_count"],
                    })
                per_policy[policy] = records
            paired = {}
            for other in ("fixed", "guarded", "skip"):
                diffs = np.array([
                    x["position_rmse_m"] - y["position_rmse_m"]
                    for x, y in zip(per_policy["voi"], per_policy[other], strict=True)
                ])
                rng = np.random.default_rng(np.random.SeedSequence([seed, len(scenario), len(estimator), len(other)]))
                indices = rng.integers(0, episodes, size=(1000, episodes))
                ci = np.quantile(diffs[indices].mean(axis=1), [0.025, 0.975]).tolist()
                paired[f"voi_minus_{other}"] = {
                    "mean_rmse_difference_m": float(diffs.mean()),
                    "bootstrap_95pct_interval_m": ci,
                    "description": "Paired resampling of whole simulated trajectories; descriptive only",
                }
            experiments[f"{scenario}/{estimator}"] = {
                "methods": {
                    key: {
                        "mean_rmse_m": float(np.mean([r["position_rmse_m"] for r in records])),
                        "mean_sensing_airtime_fraction": float(np.mean([r["sensing_airtime_fraction"] for r in records])),
                        "mean_communication_bits_proxy": float(np.mean([r["communication_bits_proxy"] for r in records])),
                        "trajectories": records,
                    }
                    for key, records in per_policy.items()
                },
                "paired": paired,
            }
    report = {
        "schema_version": 1,
        "scope": "toy 2D RF-feature task; not waveform-level ISAC nor hardware validation",
        "seed_start": seed,
        "episodes_per_variant": episodes,
        "steps": steps,
        "runtime": runtime.to_dict(),
        "voi_configuration": asdict(voi_config),
        "source_sha256": source_hashes,
        "limitations": [
            "VoI score is Gaussian hypothetical covariance reduction, not measured counterfactual risk",
            "model mismatch and persistent bias invalidate expected covariance gain",
            "no future observations included in candidate projections",
            "communication measured using toy AWGN time-sharing proxy",
            "no learned VoI network, real radio hardware or physical latency measurements",
        ],
        "experiments": experiments,
    }
    target = out / "report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=80_000_000)
    parser.add_argument("--voi-tradeoff", type=float, default=0.4)
    args = parser.parse_args()
    result = run_study(args.out, episodes=args.episodes, steps=args.steps, seed=args.seed,
                       voi_tradeoff=args.voi_tradeoff)
    for key, experiment in result["experiments"].items():
        values = ", ".join(f"{p}={v['mean_rmse_m']:.3f}m @ {v['mean_sensing_airtime_fraction']:.3f}"
                           for p, v in experiment["methods"].items())
        print(f"{key}: {values}")


if __name__ == "__main__":
    main()
