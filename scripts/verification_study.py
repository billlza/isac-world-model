"""Paired geometry-only experiment isolating the verification action rule.

Both planners use the corrected evidence protocol. This is not a byte-for-byte
rerun of the October baseline, nor a test of recurrent learning superiority.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from isac_wm.active_observation import SCENARIOS, ExperimentConfig, SyntheticWorld, run_episode
from isac_wm.artifacts import sha256_file, write_json
from isac_wm.observation_planning import PLANNERS

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "src/isac_wm/active_observation.py",
    "src/isac_wm/observation_planning.py",
    "src/isac_wm/evidence_reasoning.py",
    "src/isac_wm/phy_ofdm.py",
    "src/isac_wm/artifacts.py",
    "scripts/verification_study.py",
    "pyproject.toml",
    "uv.lock",
)
METRICS = (
    "wrong_confirmation",
    "ever_wrong_confirmation",
    "abstained",
    "candidate_set_rejected",
    "capture_count",
    "travel_m",
    "late_packets",
    "missing_detections",
    "camera_transport_bits_reserved",
    "remaining_raw_payload_bits_proxy",
)


def summarize(records: list[dict], seeds: list[int]) -> tuple[list[dict], list[dict]]:
    summary, paired = [], []
    for scenario in SCENARIOS:
        for planner in PLANNERS:
            rows = [r for r in records if r["scenario"] == scenario and r["planner"] == planner]
            if sorted(r["seed"] for r in rows) != sorted(seeds):
                raise ValueError("Missing or duplicate paired episodes")
            summary.append(
                {
                    "scenario": scenario,
                    "planner": planner,
                    "episodes": len(rows),
                    **{key: float(np.mean([r["metrics"][key] for r in rows])) for key in METRICS},
                }
            )
        for seed in seeds:
            pair = {
                r["planner"]: r["metrics"]
                for r in records
                if r["scenario"] == scenario and r["seed"] == seed
            }
            paired.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "delta_verification_minus_legacy": {
                        key: float(pair["verification-aware"][key]) - float(pair["legacy-voi"][key])
                        for key in METRICS
                    },
                }
            )
    return summary, paired


def run_study(out: Path, *, episodes: int = 8, seed: int = 20261103):
    if type(episodes) is not int or not 1 <= episodes <= 512:
        raise ValueError("episodes must be an integer in [1,512]")
    if type(seed) is not int or not 0 <= seed < 2**32 - episodes:
        raise ValueError("Invalid seed")
    out = Path(out)
    if out.exists():
        raise FileExistsError("Evidence output already exists")
    out.mkdir(parents=True)
    cfg = ExperimentConfig()
    seeds = list(range(seed, seed + episodes))
    provenance = {
        "schema_version": 1,
        "status": "running",
        "config": asdict(cfg),
        "episode_seeds": seeds,
        "planners": list(PLANNERS),
        "method": "geometry_voi",
        "comparison": "same corrected evidence protocol; only the action rule differs",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
    }
    started = time.perf_counter()
    write_json(out / "manifest.json", provenance)
    try:
        provenance["base_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        provenance["source_sha256"] = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
        records = []
        for scenario in SCENARIOS:
            for episode_seed in seeds:
                world = SyntheticWorld.create(episode_seed, scenario)
                anchor, phy = world.initial_rf()
                for planner in PLANNERS:
                    record = run_episode(
                        world, "geometry_voi", cfg, anchor=anchor, phy_report=phy, planner=planner
                    )
                    record["initial_phy"] = phy
                    records.append(record)
            print(f"Completed paired actions: {scenario}", flush=True)
        summary, paired = summarize(records, seeds)
        provenance.update(status="completed", elapsed_s=time.perf_counter() - started)
        report = {
            "schema_version": 1,
            "kind": "verification-planning-study",
            "provenance": provenance,
            "summary": summary,
            "paired_episodes": paired,
            "episodes": records,
            "limitations": [
                "Small synthetic diagnostic study; no claim of statistical superiority or GPU validation",
                "Verification can spend more observations, transport bits and travel; inspect all costs",
                "Two compatible detections and view separation do not establish independent evidence",
                "No neural training, candidate generation, real data or online parameter relearning",
                "Predictive NLL in episode traces uses policy-selected observations; not a common evaluation set",
            ],
        }
        write_json(out / "study.json", report)
        write_json(out / "summary.json", {"provenance": provenance, "summary": summary, "paired": paired})
        write_json(out / "manifest.json", provenance)
        return report
    except Exception as error:
        provenance.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        write_json(out / "manifest.json", provenance)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episodes", default=8, type=int)
    parser.add_argument("--seed", default=20261103, type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_study(**vars(args))
    print(json.dumps({"status": "completed", "episodes": len(report["episodes"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
