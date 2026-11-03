"""Forecast one synthetic target from a released sensor prefix using a structured checkpoint.

Run from the repository root: PYTHONPATH=src python scripts/forecast_from_prefix.py
--checkpoint path/to/model.pt --out new.json [--radio-queries]. No future observations
or truth are generated. The output is a state forecast, not a sensor-frame generator,
chat model, or policy that takes actions. Model covariance is not calibrated.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_wm.events import EventQueue, SnapshotBuffer
from isac_wm.simulator import SceneConfig, ToyScene
from isac_wm.validation import finite, integer

HORIZONS_S = (0.1, 0.3, 1.0)


def synthetic_prefix(scene_config: SceneConfig, seed: int, ticks: int):
    """Generate only the prefix. At the last tick, release evidence but acquire nothing."""
    integer(ticks, "prefix_steps", 8, 256)
    cfg = replace(scene_config, steps=ticks)
    scene, queue, buffer = ToyScene(seed, cfg), EventQueue(), SnapshotBuffer(cfg.image_size)
    for k in range(ticks):
        now = k * cfg.dt
        yield now, buffer.update(queue.release(now), now, cfg.dt)
        if k + 1 < ticks:
            for obs in (scene.camera(k), scene.radio(k, 1)):  # fixed quick history
                if obs is not None:
                    queue.push(obs)


def state_in_units(prediction, scale):
    mean = prediction["mean"][0].detach().cpu().numpy().astype(float) * scale
    covariance = prediction["covariance"][0].detach().cpu().numpy().astype(float) * np.outer(scale, scale)
    variance = covariance.diagonal()
    if not (np.isfinite(mean).all() and np.isfinite(covariance).all() and (variance > 0).all()):
        raise ValueError("Invalid state mean or covariance")
    return {"mean": mean.tolist(), "variance": variance.tolist(), "covariance": covariance.tolist()}


def horizon_scope(horizon, trained):
    finite(horizon, "horizon", positive=True)
    if math.isclose(horizon, trained):
        return "direct training horizon"
    if horizon > trained:
        return "longer than training horizon: extrapolation"
    return "shorter than training horizon"


def query_belief(model, belief, cfg, radio_queries=False):
    """Accept only a single estimated belief and geometry, never a scene or sensor stream."""
    import torch

    scale = cfg.state_scale.astype(float)
    forecasts = [
        {
            "horizon_s": h,
            "horizon_scope": horizon_scope(h, 3 * cfg.dt),
            **state_in_units(model.forecast(belief, h), scale),
        }
        for h in HORIZONS_S
    ]
    result = {"cutoff_state": state_in_units(model.decode_state(belief), scale), "forecasts": forecasts}
    if radio_queries:
        # These are model feature means for capture NOW; no packet, action, or future state update.
        radio = {
            "features": ["range", "cos_bearing", "sin_bearing", "radial_velocity"],
            "units": ["m", "unitless", "unitless", "m/s"],
            "interpretation": "Independent model feature means; no unit-circle constraint or packet delivery.",
        }
        rf_scale = np.array([cfg.room_m * math.sqrt(2), 1.0, 1.0, 2.0])
        for action_id, name in ((1, "quick"), (2, "precise")):
            action = torch.tensor([action_id], dtype=torch.long, device=belief.device)
            raw = model.predict_observation(belief, action)["mean"][0].detach().cpu().numpy()
            values = raw * rf_scale
            if not np.isfinite(values).all():
                raise ValueError("Non-finite RF feature mean")
            radio[name] = values.tolist()
        result["hypothetical_rf_at_cutoff"] = radio
    return result


def run(checkpoint: Path, *, device_name="cpu", seed=83_000_007, prefix_steps=12, radio_queries=False):
    import torch

    from isac_wm.artifacts import model_fingerprint, sha256_file
    from isac_wm.structured import StructuredEventModel
    from isac_wm.training import INPUT_KEYS, load_model, runtime_info

    integer(prefix_steps, "prefix_steps", 8, 256)
    integer(seed, "seed", 0, 2**32 - 1)
    model, cfg, device = load_model(checkpoint, device_name)
    if not isinstance(model, StructuredEventModel):
        raise ValueError("Expected a structured or structured-attention checkpoint")
    if seed in model.provenance_seeds:
        raise ValueError("Choose a demo seed outside the checkpoint's recorded train and validation sets")
    fresh = np.zeros(2, dtype=int)
    with torch.inference_mode():
        memory = model.initial(1, device)
        for now, snap in synthetic_prefix(cfg, seed, prefix_steps):
            fresh += snap["new"].astype(int)
            batch = {key: torch.from_numpy(snap[key])[None].to(device) for key in INPUT_KEYS}
            pred, memory = model.step(batch, memory)
        result = query_belief(model, pred["hidden"], cfg, radio_queries)
    return {
        "schema_version": 1,
        "scope": "Synthetic ToyScene prefix, model prior initialization; no future evidence or truth. "
        "Learned covariance is uncalibrated. This is neither a hardware measurement, a chat model, "
        "an acting policy nor a frame generator.",
        "architecture": model.config.architecture,
        "checkpoint_sha256": sha256_file(checkpoint),
        "loaded_parameters_sha256": model_fingerprint(model),
        "runtime": runtime_info(device),
        "seed": seed,
        "synthetic_scene": replace(cfg, steps=prefix_steps).to_dict(),
        "prefix_policy": "fixed quick RF + scheduled camera; release before capture; stop at cutoff",
        "prefix_steps": prefix_steps,
        "cutoff_s": now,
        "dt_s": cfg.dt,
        "fresh_update_ticks": {"vision": int(fresh[0]), "radio": int(fresh[1])},
        "training_forecast_horizon_s": 3 * cfg.dt,
        "state_order": ["x", "y", "vx", "vy"],
        "mean_units": ["m", "m", "m/s", "m/s"],
        "variance_units": ["m^2", "m^2", "(m/s)^2", "(m/s)^2"],
        "covariance_units": "entry [i,j] carries mean_units[i] * mean_units[j]",
        **result,
    }


def write_new(path, result):
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new JSON path in an existing directory")
    parser.add_argument("--seed", type=int, default=83_000_007)
    parser.add_argument("--prefix-steps", type=int, default=12, help="8 to 256; also sets ToyScene duration")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--radio-queries", action="store_true", help="include quick/precise RF feature means")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must name a new file")
    write_new(
        args.out,
        run(
            args.checkpoint,
            device_name=args.device,
            seed=args.seed,
            prefix_steps=args.prefix_steps,
            radio_queries=args.radio_queries,
        ),
    )
    print(args.out)


if __name__ == "__main__":
    main()
