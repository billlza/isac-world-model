"""Read a matched pair of trained checkpoints and score a blind recovery challenge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from recovery_challenge_core import observed_trace, score

from isac_wm.baselines import ReplayKalman
from isac_wm.imm import IMMConfig, ReplayIMM


def apply_network(model, cfg, device, snapshots):
    import torch

    from isac_wm.training import INPUT_KEYS

    outputs, forecasts = [], []
    with torch.inference_mode():
        memory = model.initial(1, device)
        for snap in snapshots:
            inputs = {key: torch.from_numpy(snap[key])[None].to(device) for key in INPUT_KEYS}
            current, memory = model.step(inputs, memory)
            future = model.forecast(current["hidden"], 3 * cfg.dt)
            outputs.append(current["mean"][0, :2])
            forecasts.append(future["mean"][0, :2])
    return tuple(torch.stack(x).cpu().numpy() * cfg.room_m for x in (outputs, forecasts))


def evaluate(structured, attention, *, device_name, seed_base, episodes, steps):
    from isac_wm.artifacts import sha256_file
    from isac_wm.training import load_model

    if not 1 <= episodes <= 256 or not 0 <= seed_base <= 2**32 - episodes or steps != 64:
        raise ValueError("Protocol requires 1–256 valid seeds and 64 ticks")
    configs, models, hashes, widths, provenance = [], {}, {}, [], []
    for arch, path in (("structured", structured), ("structured-attention", attention)):
        model, cfg, device = load_model(path, device_name)
        if model.config.architecture != arch or cfg.steps != steps or not model.provenance_seeds:
            raise ValueError("Checkpoint architecture, sequence length, or provenance mismatch")
        if model.provenance_seeds.intersection(range(seed_base, seed_base + episodes)):
            raise ValueError("Challenge overlaps checkpoint training/validation episodes")
        models[arch] = model
        configs.append(cfg)
        hashes[arch] = sha256_file(path)
        widths.append(model.config.hidden_dim)
        provenance.append(model.provenance_seeds)
    if configs[0] != configs[1] or widths[0] != widths[1] or provenance[0] != provenance[1]:
        raise ValueError("Not a matched pair of trained models")
    cfg = configs[0]
    rows = []
    for seed in range(seed_base, seed_base + episodes):
        for cond in ("control", "maneuver"):
            snaps, ready, truth, meta = observed_trace(seed, cfg, cond == "maneuver")
            anchor = meta["first_post_blackout_camera_delivery_tick"]
            scores = {}
            for name, model in models.items():
                then = time.perf_counter()
                est, forecast = apply_network(model, cfg, device, snaps)
                scores[name] = score(est, truth[:, :2], anchor)
                e = forecast[5:-3] - truth[8:, :2]
                scores[name]["three_step_forecast_rmse_m"] = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
                scores[name]["inference_wall_s"] = time.perf_counter() - then
            for name, estimator in (
                ("kalman", ReplayKalman(cfg.room_m)),
                ("gated-imm", ReplayIMM(cfg.room_m, config=IMMConfig(gate_threshold=9.21))),
            ):
                then = time.perf_counter()
                xy = []
                for k, packets in enumerate(ready):
                    now = k * cfg.dt
                    estimator.ingest(packets, now)
                    xy.append(estimator.estimate(now)[0][:2].copy())
                scores[name] = score(xy, truth[:, :2], anchor)
                scores[name]["inference_wall_s"] = time.perf_counter() - then
                scores[name]["gated_measurements"] = estimator.rejected_measurements
            rows.append({"seed": seed, "condition": cond, "scores": scores, "observations": meta})
    return {
        "status": "completed",
        "width": widths[0],
        "checkpoints_sha256": hashes,
        "device": str(device),
        "episodes": rows,
        "protocol": "blackout ticks 16:32; optional 90-degree turn at tick 24; recovery starts at actual delivery; trapezoidal XY-error AUC over 1s; confirmation requires 3 ticks <=0.25m",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=93_621_000)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = evaluate(
        args.structured,
        args.attention,
        device_name=args.device,
        seed_base=args.seed_base,
        episodes=args.episodes,
        steps=args.steps,
    )
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
