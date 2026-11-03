"""A96 initialization × training-block factorial with common validation and separate test."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (7, 11)
CELLS = tuple((i, b) for i in LEVELS for b in LEVELS)
ORDER_SEED = 37
CONDITIONS = ("control", "maneuver")


def design(profile):
    if profile == "formal":
        return {
            "train_episodes": 96,
            "epochs": 16,
            "batch": 8,
            "train_offset": 0,
            "validation_base": 73_511_000,
            "challenge_episodes": 24,
            "splits": {"replication_diagnostic": 93_621_000, "fresh_test": 94_761_000},
        }
    if profile == "smoke":
        return {
            "train_episodes": 8,
            "epochs": 2,
            "batch": 4,
            "train_offset": 60_000_000,
            "validation_base": 74_511_000,
            "challenge_episodes": 2,
            "splits": {"development_A": 83_430_000, "development_B": 83_431_000},
        }
    raise ValueError("Unknown study profile")


def cell_name(i, b):
    return f"init-{i}-block-{b}"


def factorial_auc(cells):
    """Pair by trajectory. Uncertainty is conditional on the two chosen factor levels."""
    if set(cells) != set(CELLS):
        raise ValueError("Missing or unexpected factorial cell")
    first = cells[CELLS[0]]
    seeds = sorted(first)
    if len(first) < 2 or any(set(v) != set(first) for v in cells.values()):
        raise ValueError("Missing or mismatched paired trajectories")
    a, b, c, d = (np.array([cells[key][e] for e in seeds], dtype=float) for key in CELLS)
    if not np.isfinite([a, b, c, d]).all():
        raise ValueError("Non-finite factorial AUC")
    deltas = {
        "init_11_minus_7_at_block_7": c - a,
        "init_11_minus_7_at_block_11": d - b,
        "data_11_minus_7_at_init_7": b - a,
        "data_11_minus_7_at_init_11": d - c,
        "init_11_minus_7_average": ((c - a) + (d - b)) / 2,
        "data_11_minus_7_average": ((b - a) + (d - c)) / 2,
        "interaction": (d - c) - (b - a),
    }
    idx = np.random.default_rng(20261103).integers(len(seeds), size=(4000, len(seeds)))
    return {
        k: {
            "mean_m_s": float(x.mean()),
            "trajectory_bootstrap_95_m_s": np.quantile(x[idx].mean(axis=1), [0.025, 0.975]).tolist(),
            "negative_pairs": int((x < 0).sum()),
        }
        for k, x in deltas.items()
    }


def run_baseline(est, cfg, ready, truth, anchor):
    from recovery_challenge_core import score

    xy = []
    start = time.perf_counter()
    for k, packets in enumerate(ready):
        now = k * cfg.dt
        est.ingest(packets, now)
        xy.append(est.estimate(now)[0][:2].copy())
    metrics = score(np.asarray(xy), truth[:, :2], anchor)
    metrics["inference_wall_s"] = time.perf_counter() - start
    metrics["gated_measurements"] = est.rejected_measurements
    return metrics


def score_split(models, cfg, device, start_seed, count):
    from recovery_challenge_core import observed_trace, score
    from score_scale_recovery import apply_network

    from isac_wm.baselines import ReplayKalman
    from isac_wm.imm import IMMConfig, ReplayIMM

    seeds = range(start_seed, start_seed + count)
    if any(m.provenance_seeds.intersection(seeds) for m in models.values()):
        raise ValueError("Scored episodes overlap training or validation")
    rows = []
    summary = {}
    for cond in CONDITIONS:
        panels = {k: {} for k in CELLS}
        for s in seeds:
            snaps, ready, truth, meta = observed_trace(s, cfg, cond == "maneuver")
            anchor = meta["first_post_blackout_camera_delivery_tick"]
            scores = {}
            for (i, b), model in models.items():
                then = time.perf_counter()
                est, future = apply_network(model, cfg, device, snaps)
                met = score(est, truth[:, :2], anchor)
                met["inference_wall_s"] = time.perf_counter() - then
                err = future[5:-3] - truth[8:, :2]
                met["three_step_forecast_rmse_m"] = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
                panels[i, b][s] = met["position_error_auc_m_s"]
                scores[cell_name(i, b)] = met
            scores["kalman"] = run_baseline(ReplayKalman(cfg.room_m), cfg, ready, truth, anchor)
            scores["gated-imm"] = run_baseline(
                ReplayIMM(cfg.room_m, config=IMMConfig(gate_threshold=9.21)), cfg, ready, truth, anchor
            )
            rows.append({"seed": s, "condition": cond, "observations": meta, "scores": scores})
        condition_rows = [r["scores"] for r in rows if r["condition"] == cond]
        summary[cond] = {
            "factorial_contrasts": factorial_auc(panels),
            "methods": {
                n: {
                    "mean_auc_m_s": float(np.mean([r[n]["position_error_auc_m_s"] for r in condition_rows])),
                    "unrecovered_by_1s": sum(int(r[n]["not_recovered_by_1s"]) for r in condition_rows),
                }
                for n in condition_rows[0]
            },
        }
    return {"rows": rows, "summary": summary}


def run(out, device_name, profile):
    import torch

    from isac_wm.artifacts import sha256_file, write_json
    from isac_wm.simulator import SceneConfig
    from isac_wm.structured import StructuredConfig
    from isac_wm.training import load_model, resolve_device, runtime_info, train

    spec = design(profile)
    expected_device = "cpu" if profile == "smoke" else "cuda"
    if device_name != expected_device:
        raise ValueError(f"{profile} requires --device {expected_device}")
    device = resolve_device(device_name)
    torch.set_num_threads(2)
    cfg = SceneConfig(steps=64)
    val = list(range(spec["validation_base"], spec["validation_base"] + max(8, spec["train_episodes"] // 8)))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "schema_version": 1,
        "profile": profile,
        "git_commit": revision,
        "runtime": runtime_info(device),
        "initialization_seeds": list(LEVELS),
        "training_blocks": list(LEVELS),
        "minibatch_order_seed": ORDER_SEED,
        "validation_episode_seeds": val,
        "train_config": spec,
        "interpretation": (
            "Conditional on these two initializations and two training trajectory blocks. "
            "Intervals resample challenge trajectories, never training seeds or ticks."
        ),
        "cells": {},
        "results": {},
    }
    dest = out / "crossed.json"
    write_json(dest, report)
    try:
        trained, provenance = {}, {}
        for i, b in CELLS:
            name = cell_name(i, b)
            folder = out / name
            base = spec["train_offset"] + b * 100_000
            stage = {"status": "running", "init": i, "train_base": base}
            report["cells"][name] = stage
            write_json(dest, report)
            start = time.perf_counter()
            m = train(
                folder,
                epochs=spec["epochs"],
                episodes=spec["train_episodes"],
                batch_size=spec["batch"],
                device_name=device_name,
                seed=i,
                scene=cfg,
                model_config=StructuredConfig(architecture="structured-attention"),
                train_seed_base=base,
                validation_seed_base=spec["validation_base"],
                batch_order_seed=ORDER_SEED,
            )
            train_set = list(range(base, base + spec["train_episodes"]))
            if (
                m["parameters"] != 172246
                or m["completed_epochs"] != spec["epochs"]
                or m["runtime"]["device"] != device_name
                or m["train_episode_seeds"] != train_set
                or m["validation_episode_seeds"] != val
                or m["resume_source"] is not None
                or m["seed_plan"] != dict(initialization=i, train_seed_base=base,
                                          validation_seed_base=spec["validation_base"], batch_order_seed=ORDER_SEED)
            ):
                raise RuntimeError(f"Invalid cell training evidence: {name}")
            ck = folder / "model.pt"
            digest = sha256_file(ck)
            if Path(str(ck) + ".sha256").read_text().strip() != digest:
                raise ValueError("Model digest does not match its sidecar")
            stage.update(
                status="completed",
                wall_seconds=time.perf_counter() - start,
                train_episodes=train_set,
                initial_parameters_sha256=m["initial_parameters_sha256"],
                checkpoint_sha256=digest,
                best_validation_rmse_m=m["best_validation_rmse_m"],
            )
            trained[i, b] = ck
            provenance[i, b] = set(train_set + val)
            write_json(dest, report)
        # All four models and their digests are committed to the local report before any test score.
        starts = {}
        for i, b in CELLS:
            fp = report["cells"][cell_name(i, b)]["initial_parameters_sha256"]
            if starts.setdefault(i, fp) != fp:
                raise RuntimeError("Initial weights differed across training blocks for the same init")
        if len(set(starts.values())) != 2:
            raise RuntimeError("The two initialization settings produced identical models")
        models = {}
        for key, ck in trained.items():
            if sha256_file(ck) != report["cells"][cell_name(*key)]["checkpoint_sha256"]:
                raise ValueError("Best checkpoint changed after training")
            model, scene, dev = load_model(ck, device_name)
            if scene != cfg or dev != device or model.provenance_seeds != provenance[key]:
                raise ValueError("Checkpoint training or geometry provenance mismatch")
            models[key] = model
        for alias, start_seed in spec["splits"].items():
            result = score_split(models, cfg, device, start_seed, spec["challenge_episodes"])
            write_json(out / f"{alias}.json", result)
            report["results"][alias] = result["summary"]
            write_json(dest, report)
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(dest, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--profile", choices=["formal", "smoke"], default="formal")
    args = parser.parse_args()
    report = run(args.out, args.device, args.profile)
    print(json.dumps({"status": report["status"], "results": report["results"]}, allow_nan=False))


if __name__ == "__main__":
    main()
