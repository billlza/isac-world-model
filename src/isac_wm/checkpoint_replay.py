"""Read-only, bounded replay of registered validation checkpoints.

The two execution paths receive identical causal inputs and the same recorded
episodes. Truth is used only after predictions, for descriptive error summaries.
Output tolerances diagnose numerical differences; they are not safety limits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    MAX_CHECKPOINT_BYTES,
    MAX_RESUME_CHECKPOINT_BYTES,
    model_fingerprint,
    read_checkpoint,
    read_resume_checkpoint,
    sha256_file,
    write_json,
)
from .simulator import SceneConfig, recorded_episode
from .structured import StructuredConfig, build_model, model_config_from_dict
from .training import INPUT_KEYS, batchify, resolve_device, runtime_info
from .visual_training_diagnostics import passive_observer

FORMAL_VALIDATION_SEEDS = tuple(range(74611000, 74611192))
FORMAL_TRAIN_SEEDS = tuple(range(700000, 701536))
SCORED_TICK_START = 5
ERROR_THRESHOLDS_M = (0.1, 0.25, 0.5, 1.0)
OUTPUT_ATOL = 2e-5
OUTPUT_RTOL = 2e-4
SCORE_ATOL_M = 1e-5
SCORE_RTOL = 1e-4
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
STRATA_GROUPS = (
    ("fresh_visible_vision", "fresh_occluded_vision", "no_fresh_vision"),
    ("radio_fresh", "radio_not_fresh"),
    (
        "camera_unavailable",
        "camera_age_lt_0.2s",
        "camera_age_0.2_to_0.5s",
        "camera_age_0.5_to_1s",
        "camera_age_ge_1s",
    ),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, label):
    array = np.asarray(value, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError(f"Non-finite {label}")
    return array


def _digest(value, label, length=64):
    _require(isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value), f"Invalid {label}")
    return value


def error_summary(values) -> dict:
    """Pool scalar Euclidean errors; quantiles use NumPy's linear convention."""
    values = _finite(values, "error samples").reshape(-1)
    _require(bool((values >= 0).all()), "Error samples must be non-negative")
    count = int(values.size)
    scale = float(values.max()) if count else 0.0
    normalized = values / scale if scale else values
    return {
        "count": count,
        "rmse_m": float(scale * np.sqrt(np.mean(np.square(normalized)))) if count else None,
        "mean_m": float(scale * normalized.mean()) if count else None,
        "p95_m": float(np.quantile(values, 0.95, method="linear")) if count else None,
        "p99_m": float(np.quantile(values, 0.99, method="linear")) if count else None,
        "max_m": float(values.max()) if count else None,
        "exceedance": {
            f"{threshold:g}": {
                "threshold_m": threshold,
                "count": int((values > threshold).sum()),
                "fraction": float((values > threshold).mean()) if count else None,
            }
            for threshold in ERROR_THRESHOLDS_M
        },
    }


def state_dict_fingerprint(state_dict) -> str:
    """Same tensor/name/dtype/shape fingerprint as artifacts.model_fingerprint."""
    _require(isinstance(state_dict, dict) and bool(state_dict), "Missing checkpoint state_dict")
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        _require(isinstance(name, str) and isinstance(value, torch.Tensor), "Invalid checkpoint tensor")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite checkpoint tensor: {name}")
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _preserve_rng():
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _read_manifest(path, expected):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "Manifest must be a regular non-symlink file")
    _require(0 < path.stat().st_size <= MAX_MANIFEST_BYTES, "Manifest exceeds replay budget")
    with path.open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    _require(len(raw) <= MAX_MANIFEST_BYTES, "Manifest exceeds replay budget")
    _require(hashlib.sha256(raw).hexdigest() == expected, "Manifest SHA-256 mismatch")

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    result = json.loads(raw, parse_constant=invalid_constant)
    _require(isinstance(result, dict), "Manifest must contain a mapping")
    return result


def _numerical_settings():
    """Observe the evaluator's actual settings without changing its policy."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def _validate_protocol(manifest, profile, device_name):
    _require(profile in {"formal", "development"}, "Unknown replay profile")
    _require(device_name == ("cuda" if profile == "formal" else "cpu"), "Replay profile/device differ")
    _require(manifest.get("runtime", {}).get("device") == device_name, "Checkpoint training device differs")
    seeds = manifest.get("validation_episode_seeds")
    train = manifest.get("train_episode_seeds")
    for name, values in (("validation", seeds), ("training", train)):
        _require(
            isinstance(values, list)
            and values
            and len(values) <= 1536
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in values)
            and len(set(values)) == len(values),
            f"Invalid {name} replay seeds",
        )
    _require(not set(seeds).intersection(train), "Replay training/validation split overlaps")
    _require(manifest.get("validation_episodes") == len(seeds), "Replay validation count differs")
    _require(manifest.get("episodes") == len(train), "Replay training count differs")
    _require(manifest.get("batch_size") == 8, "Registered replay requires batch size 8")
    scene = SceneConfig(**manifest["scene"])
    _require(SCORED_TICK_START < scene.steps <= 64, "Replay trajectory length exceeds registered bound")
    config = model_config_from_dict(manifest["model"])
    _require(isinstance(config, StructuredConfig), "Replay requires the structured visual model")
    _require(config.room_m == scene.room_m, "Replay model/scene geometry differs")
    if profile == "formal":
        _require(
            tuple(seeds) == FORMAL_VALIDATION_SEEDS,
            "Formal replay must use the existing 192 validation episodes",
        )
        _require(tuple(train) == FORMAL_TRAIN_SEEDS, "Formal replay training split differs")
        _require(
            scene.to_dict() == SceneConfig(steps=64, dt=0.1, image_size=24).to_dict(),
            "Formal replay scene differs",
        )
    else:
        _require(len(seeds) <= 8 and len(train) <= 8, "Development replay exceeds episode bound")
        _require(
            not (set(seeds) | set(train)).intersection(FORMAL_VALIDATION_SEEDS + FORMAL_TRAIN_SEEDS),
            "Development replay must not consume formal episodes",
        )
    _require(manifest.get("model_selection_status") == "selected", "Replay requires a selected model")
    _require(
        type(manifest.get("optimizer_updates")) is int
        and 0 < manifest["best_update"] <= manifest["optimizer_updates"],
        "Invalid checkpoint update counts",
    )
    return config, scene


def _validate_payload(payload, manifest, kind):
    _require(kind in {"best", "final"}, "Checkpoint kind must be best or final")
    _require(payload.get("schema_version") == 2, "Replay requires checkpoint schema version 2")
    expected = {
        "model_config": manifest["model"],
        "scene_config": manifest["scene"],
        "seed": manifest["seed"],
        "seed_plan": manifest["seed_plan"],
        "validation_episodes": manifest["validation_episodes"],
        "validation_every_updates": manifest["validation_every_updates"],
        "vision_loss_weight": manifest["vision_loss_weight"],
        "vision_supervision_protocol": manifest["vision_supervision_protocol"],
        "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
        "diagnostic_protocol": manifest["diagnostic_protocol"],
    }
    for field in ("optimizer_policy", "width_lr_scale"):
        if field in manifest:
            expected[field] = manifest[field]
    if kind == "best":
        expected.update(
            train_episode_seeds=manifest["train_episode_seeds"],
            validation_episode_seeds=manifest["validation_episode_seeds"],
            best_update=manifest["best_update"],
        )
        update = manifest["best_update"]
        score = manifest["best_validation_rmse_m"]
    else:
        expected.update(
            episodes=manifest["episodes"],
            batch_size=manifest["batch_size"],
            learning_rate=manifest["learning_rate"],
            optimizer_updates=manifest["optimizer_updates"],
            completed_epochs=manifest["completed_epochs"],
        )
        update = manifest["optimizer_updates"]
        history = payload.get("history")
        _require(
            isinstance(history, list) and history and history[-1].get("optimizer_updates") == update,
            "Final checkpoint lacks validation at its own update",
        )
        score = history[-1]["validation_position_rmse_m"]
        _require(
            isinstance(payload.get("best_checkpoint"), dict)
            and payload["best_checkpoint"].get("best_update") == manifest["best_update"],
            "Final checkpoint selected-model metadata differs",
        )
    for key, value in expected.items():
        _require(payload.get(key) == value, f"Checkpoint and manifest differ: {key}")
    _require(float(_finite(score, "recorded score")) >= 0, "Recorded score must be non-negative")
    return update, float(score)


def _model_inputs(batch):
    # The recorded action is used by the auxiliary observation predictor only.
    # State and capture labels are intentionally excluded from the model call.
    return {key: batch[key] for key in (*INPUT_KEYS, "action_target")}


def _cpu_output(output):
    result = {}
    for key in ("mean", "logvar", "visual_xy"):
        value = output[key].detach().cpu().numpy()
        _finite(value, f"model output {key}")
        result[key] = value
    return result


def _streaming_output(model, batch):
    """Capture only small per-event tensors, never a full latent-state history."""
    size, steps = batch["image"].shape[:2]
    memory = model.initial(size, batch["image"].device)
    output = {key: [] for key in ("mean", "logvar", "visual_xy")}
    observed = {modality: {} for modality in ("vision", "radio")}
    for tick in range(steps):
        inputs = {key: batch[key][:, tick] for key in INPUT_KEYS}
        encoded = model.encode_new(inputs)
        diagnostics = {}
        state, memory = model.step(inputs, memory, encoded, diagnostics=diagnostics)
        output["mean"].append(state["mean"])
        output["logvar"].append(state["logvar"])
        output["visual_xy"].append(encoded[0][:, :2])
        for modality in observed:
            row = diagnostics[modality]
            fields = {
                "active": row["active"].squeeze(-1),
                "correction_m": row["applied_state_correction"][:, :2] * model.config.room_m,
                "innovation_m": row["innovation_xy"] * model.config.room_m,
                "gain": row["gain"],
                "variance_m2": row["measurement_variance_xy"] * model.config.room_m**2,
            }
            if modality == "vision":
                fields.update(
                    learned_variance_m2=row["learned_measurement_variance_xy"] * model.config.room_m**2,
                    heatmap_entropy=row["heatmap_entropy"].squeeze(-1),
                )
            for key, value in fields.items():
                observed[modality].setdefault(key, []).append(value)
    arrays = _cpu_output({key: torch.stack(value, 1) for key, value in output.items()})
    corrections = {
        modality: {key: torch.stack(values, 1).cpu().numpy() for key, values in fields.items()}
        for modality, fields in observed.items()
    }
    for modality, fields in corrections.items():
        for key, values in fields.items():
            _finite(values, f"{modality} {key}")
    return arrays, corrections


def _strata(episode):
    fresh = (episode["new"] * episode["valid"]).astype(bool)
    visible = episode["vision_target_mask"].astype(bool)
    valid_camera = episode["valid"][:, 0].astype(bool)
    age = episode["age"][:, 0]
    result = {
        "fresh_visible_vision": fresh[:, 0] & visible,
        "fresh_occluded_vision": fresh[:, 0] & ~visible,
        "no_fresh_vision": ~fresh[:, 0],
        "radio_fresh": fresh[:, 1],
        "radio_not_fresh": ~fresh[:, 1],
        "camera_unavailable": ~valid_camera,
        "camera_age_lt_0.2s": valid_camera & (age < 0.2),
        "camera_age_0.2_to_0.5s": valid_camera & (age >= 0.2) & (age < 0.5),
        "camera_age_0.5_to_1s": valid_camera & (age >= 0.5) & (age < 1.0),
        "camera_age_ge_1s": valid_camera & (age >= 1.0),
    }
    return {key: values[SCORED_TICK_START:] for key, values in result.items()}


def _path_summary(predictions, episodes, seeds, room_m, reference_score):
    errors, visual_errors, trajectory_rmse, strata = [], [], [], {}
    trajectories = []
    for index, (seed, episode) in enumerate(zip(seeds, episodes, strict=True)):
        difference = predictions["mean"][index, SCORED_TICK_START:, :2].astype(np.float64) - episode[
            "state_target"
        ][SCORED_TICK_START:, :2].astype(np.float64)
        current = np.linalg.norm(difference, axis=-1) * room_m
        row = {
            "seed": seed,
            "position_error_m": current.tolist(),
            "position_error": error_summary(current),
            "visible_captures": [],
        }
        errors.extend(current.tolist())
        trajectory_rmse.append(row["position_error"]["rmse_m"])
        masks = _strata(episode)
        row["strata_masks"] = {key: mask.tolist() for key, mask in masks.items()}
        for name, mask in masks.items():
            strata.setdefault(name, []).extend(current[mask].tolist())
        for tick in np.flatnonzero(episode["vision_target_mask"]):
            visual = float(
                np.linalg.norm(
                    predictions["visual_xy"][index, tick].astype(np.float64)
                    - episode["vision_capture_target_xy"][tick].astype(np.float64)
                )
                * room_m
            )
            _finite(visual, "visual capture error")
            row["visible_captures"].append(
                {
                    "tick": int(tick),
                    "capture_s": float(episode["vision_capture_s"][tick]),
                    "event_uid": int(episode["vision_event_uid"][tick]),
                    "error_m": visual,
                }
            )
            visual_errors.append(visual)
        trajectories.append(row)
    return {
        "reference_position_rmse_m": reference_score,
        "position_error": error_summary(errors),
        "trajectory_rmse": error_summary(trajectory_rmse),
        "visual_capture_error": error_summary(visual_errors),
        "strata": {key: error_summary(values) for key, values in strata.items()},
        "trajectories": trajectories,
    }


def _correction_summary(batches):
    result = {}
    for modality in ("vision", "radio"):
        collected = {}
        for batch in batches:
            mask = batch[modality]["active"].astype(bool)
            for key, values in batch[modality].items():
                if key != "active":
                    collected.setdefault(key, []).append(values[mask])
        arrays = {key: np.concatenate(values) for key, values in collected.items()}
        count = len(arrays["correction_m"])
        result[modality] = {
            "applied_event_count": count,
            "correction_length": error_summary(np.linalg.norm(arrays["correction_m"], axis=-1)),
            "innovation_length": error_summary(np.linalg.norm(arrays["innovation_m"], axis=-1)),
            "gain_rms": float(np.sqrt(np.mean(arrays["gain"].astype(np.float64) ** 2))) if count else None,
            "position_gain_frobenius_rms": (
                float(np.sqrt(np.mean(np.sum(arrays["gain"][:, :2].astype(np.float64) ** 2, axis=(-2, -1)))))
                if count
                else None
            ),
            "measurement_variance_mean_m2": float(arrays["variance_m2"].mean()) if count else None,
            "measurement_variance_min_m2": float(arrays["variance_m2"].min()) if count else None,
            "measurement_variance_max_m2": float(arrays["variance_m2"].max()) if count else None,
            "scope": "all applied fresh valid corrections, including occluded captures; all replay ticks",
        }
        if modality == "vision":
            result[modality].update(
                learned_variance_mean_m2=float(arrays["learned_variance_m2"].mean()) if count else None,
                heatmap_entropy_mean=float(arrays["heatmap_entropy"].mean()) if count else None,
            )
    return result


def compare_outputs(reference, candidate, seeds, room_m):
    """Describe differences at every tick without treating tolerance as a gate."""
    _require(set(reference) == set(candidate), "Replay output fields differ")
    for key in reference:
        _require(reference[key].shape == candidate[key].shape, "Replay output shapes differ")
        _finite(reference[key], f"reference {key}")
        _finite(candidate[key], f"candidate {key}")
    difference = candidate["mean"].astype(np.float64) - reference["mean"].astype(np.float64)
    per_coordinate = np.abs(difference)
    ratios = (per_coordinate / (OUTPUT_ATOL + OUTPUT_RTOL * np.abs(reference["mean"]))).max(-1)
    bad = ratios > 1
    position = np.linalg.norm(difference[..., :2], axis=-1) * room_m
    rows = []
    for index, seed in enumerate(seeds):
        locations = np.flatnonzero(bad[index])
        rows.append(
            {
                "seed": seed,
                "position_delta_m": position[index].tolist(),
                "mean_max_abs_delta_normalized": per_coordinate[index].max(-1).tolist(),
                "mean_max_tolerance_ratio": ratios[index].tolist(),
                "logvar_max_abs_delta": np.abs(candidate["logvar"][index] - reference["logvar"][index])
                .max(-1)
                .tolist(),
                "visual_max_abs_delta_normalized": np.abs(
                    candidate["visual_xy"][index] - reference["visual_xy"][index]
                )
                .max(-1)
                .tolist(),
                "mean_exceeds_tolerance": bad[index].tolist(),
                "first_divergence_tick": int(locations[0]) if len(locations) else None,
            }
        )
    locations = np.argwhere(bad)
    first = min(((int(tick), int(index)) for index, tick in locations), default=None)
    return {
        "mean_atol_normalized": OUTPUT_ATOL,
        "mean_rtol": OUTPUT_RTOL,
        "tolerance_purpose": "descriptive numerical-path comparison, not a safety limit",
        "within_registered_tolerance": not bool(bad.any()),
        "exceeding_tick_count": int(bad.sum()),
        "first_divergence": None if first is None else {"seed": seeds[first[1]], "tick": first[0]},
        "max_mean_abs_delta_normalized": float(per_coordinate.max()),
        "position_delta": error_summary(position.reshape(-1)),
        "trajectories": rows,
    }


def _replay_model(model, manifest, device, profile):
    """Evaluate an already loaded model, preserving its weights, modes and RNG."""
    _, scene = _validate_protocol(manifest, profile, str(device))
    seeds = manifest["validation_episode_seeds"]
    before = model_fingerprint(model)
    with passive_observer(model):
        episodes = [recorded_episode(seed, scene) for seed in seeds]
        all_outputs = {
            path: {key: [] for key in ("mean", "logvar", "visual_xy")} for path in ("batched", "streaming")
        }
        sums = {path: 0.0 for path in all_outputs}
        count = 0
        observed = []
        for begin in range(0, len(seeds), 8):
            batch = batchify(episodes[begin : begin + 8], device)
            reference = model(
                _model_inputs(batch), forecast_s=3 * scene.dt, batched_encoding=True, return_visual=True
            )
            current = _cpu_output(reference)
            streamed, corrections = _streaming_output(model, batch)
            observed.append(corrections)
            # Reproduce the original float32, per-batch full-validation reduction.
            mse = (
                (
                    reference["mean"][:, SCORED_TICK_START:, :2]
                    - batch["state_target"][:, SCORED_TICK_START:, :2]
                )
                .square()
                .sum(-1)
            )
            sums["batched"] += float(mse.sum()) * scene.room_m**2
            count += mse.numel()
            stream_tensor = torch.from_numpy(streamed["mean"]).to(device)
            stream_mse = (
                (stream_tensor[:, SCORED_TICK_START:, :2] - batch["state_target"][:, SCORED_TICK_START:, :2])
                .square()
                .sum(-1)
            )
            sums["streaming"] += float(stream_mse.sum()) * scene.room_m**2
            for path, values in (("batched", current), ("streaming", streamed)):
                for key, value in values.items():
                    all_outputs[path][key].append(value)
        all_outputs = {
            path: {key: np.concatenate(values) for key, values in fields.items()}
            for path, fields in all_outputs.items()
        }
        prefix = min(4, len(seeds))
        prefix_batch = batchify(episodes[:prefix], device)
        prefix_batched = _cpu_output(
            model(
                _model_inputs(prefix_batch),
                forecast_s=3 * scene.dt,
                batched_encoding=True,
                return_visual=True,
            )
        )
        prefix_streaming, _ = _streaming_output(model, prefix_batch)
        result = {
            "paths": {
                path: _path_summary(output, episodes, seeds, scene.room_m, math.sqrt(sums[path] / count))
                for path, output in all_outputs.items()
            },
            "comparisons": {
                "full_streaming_vs_batched": compare_outputs(
                    all_outputs["batched"], all_outputs["streaming"], seeds, scene.room_m
                ),
                "first4_streaming_vs_batched": compare_outputs(
                    prefix_batched, prefix_streaming, seeds[:prefix], scene.room_m
                ),
                "first4_batched4_vs_batched8": compare_outputs(
                    {key: values[:prefix] for key, values in all_outputs["batched"].items()},
                    prefix_batched,
                    seeds[:prefix],
                    scene.room_m,
                ),
            },
        }
        result["paths"]["streaming"]["correction_diagnostics"] = _correction_summary(observed)
        result["prefix_position_error"] = {
            "batched4": _path_summary(prefix_batched, episodes[:prefix], seeds[:prefix], scene.room_m, None)[
                "position_error"
            ],
            "streaming4": _path_summary(
                prefix_streaming, episodes[:prefix], seeds[:prefix], scene.room_m, None
            )["position_error"],
        }
    _require(model_fingerprint(model) == before, "Replay changed model tensors")
    return result


def _all_finite_json(value):
    if isinstance(value, dict):
        for child in value.values():
            _all_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _all_finite_json(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("Non-finite replay report")


def _validate_comparison(value, seeds, steps):
    _require(
        value.get("mean_atol_normalized") == OUTPUT_ATOL and value.get("mean_rtol") == OUTPUT_RTOL,
        "Replay comparison tolerance differs",
    )
    rows = value.get("trajectories")
    _require(
        isinstance(rows, list) and [row.get("seed") for row in rows] == seeds, "Comparison episodes differ"
    )
    positions, maxima, exceedances, firsts = [], [], 0, []
    for index, row in enumerate(rows):
        for field in (
            "position_delta_m",
            "mean_max_abs_delta_normalized",
            "mean_max_tolerance_ratio",
            "logvar_max_abs_delta",
            "visual_max_abs_delta_normalized",
            "mean_exceeds_tolerance",
        ):
            _require(
                isinstance(row.get(field), list) and len(row[field]) == steps, "Comparison tick count differs"
            )
        for field in (
            "position_delta_m",
            "mean_max_abs_delta_normalized",
            "mean_max_tolerance_ratio",
            "logvar_max_abs_delta",
            "visual_max_abs_delta_normalized",
        ):
            _require(bool((_finite(row[field], field) >= 0).all()), "Negative comparison magnitude")
        expected_flags = [ratio > 1 for ratio in row["mean_max_tolerance_ratio"]]
        _require(row["mean_exceeds_tolerance"] == expected_flags, "Comparison exceedances differ")
        locations = [tick for tick, flag in enumerate(expected_flags) if flag]
        _require(
            row["first_divergence_tick"] == (locations[0] if locations else None), "First divergence differs"
        )
        if locations:
            firsts.append((locations[0], index))
        positions.extend(row["position_delta_m"])
        maxima.extend(row["mean_max_abs_delta_normalized"])
        exceedances += len(locations)
    first = min(firsts, default=None)
    _require(value.get("position_delta") == error_summary(positions), "Comparison delta summary differs")
    _require(value.get("max_mean_abs_delta_normalized") == max(maxima), "Comparison maximum differs")
    _require(value.get("exceeding_tick_count") == exceedances, "Comparison exceedance count differs")
    _require(
        value.get("within_registered_tolerance") is (exceedances == 0), "Comparison tolerance outcome differs"
    )
    _require(
        value.get("first_divergence")
        == (None if first is None else {"seed": seeds[first[1]], "tick": first[0]}),
        "Comparison first divergence differs",
    )


def validate_replay_report(
    report,
    manifest,
    *,
    kind,
    device_name,
    profile,
    source_commit,
    expected_checkpoint_sha256,
    expected_manifest_sha256,
):
    """Pure integrity/schema checks; measured numerical disagreement is retained."""
    _, scene = _validate_protocol(manifest, profile, device_name)
    seeds = manifest["validation_episode_seeds"]
    expected = {
        "schema_version": 1,
        "kind": "checkpoint-validation-replay",
        "status": "completed",
        "profile": profile,
        "device": device_name,
        "checkpoint_kind": kind,
        "checkpoint_source_commit": source_commit,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "selected_best_update": manifest["best_update"],
        "weights_update": manifest["best_update"] if kind == "best" else manifest["optimizer_updates"],
        "model_config": manifest["model"],
        "scene_config": manifest["scene"],
        "validation_episode_seeds": seeds,
    }
    for field, value in expected.items():
        _require(report.get(field) == value, f"Replay report differs: {field}")
    _require(kind in {"best", "final"}, "Unknown replay checkpoint kind")
    for field, expected_bool in (
        ("fresh_test_consumed", False),
        ("deployment_certification", False),
        ("unchanged_model", True),
        ("rng_preserved", True),
    ):
        _require(report.get(field) is expected_bool, f"Replay flag differs: {field}")
    _digest(source_commit, "checkpoint source commit", 40)
    for key in ("checkpoint_sha256", "manifest_sha256", "state_dict_fingerprint"):
        _digest(report.get(key), key)
    _all_finite_json(report)
    _require(
        len(json.dumps(report, indent=2, allow_nan=False).encode()) + 1 <= MAX_REPORT_BYTES,
        "Replay report exceeds budget",
    )
    protocol = report.get("protocol", {})
    _require(
        protocol.get("scored_tick_start") == SCORED_TICK_START and protocol.get("steps") == scene.steps,
        "Replay score support differs",
    )
    _require(
        protocol.get("reference_batch_size") == 8 and protocol.get("prefix_episodes") == min(4, len(seeds)),
        "Replay batching protocol differs",
    )
    _require(protocol.get("error_thresholds_m") == list(ERROR_THRESHOLDS_M), "Replay thresholds differ")
    _require(report.get("runtime", {}).get("device") == device_name, "Replay runtime device differs")
    _require(set(report.get("paths", {})) == {"batched", "streaming"}, "Replay execution paths differ")
    for name, path in report["paths"].items():
        rows = path.get("trajectories")
        _require(
            isinstance(rows, list) and [row.get("seed") for row in rows] == seeds,
            "Replay trajectory order differs",
        )
        errors, visual, trajectory_scores, strata = [], [], [], {}
        for row in rows:
            values = row.get("position_error_m")
            _require(
                isinstance(values, list) and len(values) == scene.steps - SCORED_TICK_START,
                "Replay position tick count differs",
            )
            _require(row.get("position_error") == error_summary(values), "Trajectory error summary differs")
            errors.extend(values)
            trajectory_scores.append(row["position_error"]["rmse_m"])
            captures = row.get("visible_captures")
            _require(
                isinstance(captures, list) and len(captures) <= scene.steps,
                "Visual capture count exceeds bound",
            )
            ticks = [capture.get("tick") for capture in captures]
            _require(
                all(type(tick) is int and 0 <= tick < scene.steps for tick in ticks)
                and ticks == sorted(set(ticks)),
                "Visual capture ticks differ",
            )
            for capture in captures:
                _require(
                    type(capture.get("event_uid")) is int and capture["event_uid"] >= 0,
                    "Invalid visual event identity",
                )
                _require(
                    0 <= capture["capture_s"] <= capture["tick"] * scene.dt + 1e-8,
                    "Invalid visual capture time",
                )
                visual.append(capture["error_m"])
            masks = row.get("strata_masks")
            _require(
                isinstance(masks, dict) and set(masks) == {name for group in STRATA_GROUPS for name in group},
                "Replay stratum names differ",
            )
            for stratum, mask in masks.items():
                _require(
                    isinstance(mask, list)
                    and len(mask) == len(values)
                    and all(type(v) is bool for v in mask),
                    "Invalid replay stratum mask",
                )
                strata.setdefault(stratum, []).extend(
                    value for value, included in zip(values, mask, strict=True) if included
                )
            for group in STRATA_GROUPS:
                _require(
                    all(sum(masks[name][tick] for name in group) == 1 for tick in range(len(values))),
                    "Replay stratum group is not a partition",
                )
        _require(path.get("position_error") == error_summary(errors), "Pooled position error summary differs")
        _require(path.get("visual_capture_error") == error_summary(visual), "Visual error summary differs")
        _require(
            path.get("trajectory_rmse") == error_summary(trajectory_scores), "Trajectory RMSE summary differs"
        )
        _require(
            path.get("strata") == {key: error_summary(values) for key, values in strata.items()},
            "Stratified summaries differ",
        )
        score = float(_finite(path.get("reference_position_rmse_m"), f"{name} reference score"))
        _require(
            score >= 0 and math.isclose(score, path["position_error"]["rmse_m"], rel_tol=1e-5, abs_tol=1e-7),
            "Reference and pooled errors disagree",
        )
    expected_comparisons = {
        "full_streaming_vs_batched",
        "first4_streaming_vs_batched",
        "first4_batched4_vs_batched8",
    }
    _require(set(report.get("comparisons", {})) == expected_comparisons, "Replay comparison paths differ")
    for name, comparison in report["comparisons"].items():
        _validate_comparison(comparison, seeds if name.startswith("full_") else seeds[:4], scene.steps)
    scored = report.get("recorded_score", {})
    expected_score = scored.get("expected_rmse_m")
    _require(
        float(_finite(expected_score, "recorded validation score")) >= 0, "Invalid recorded validation score"
    )
    if kind == "best":
        _require(expected_score == manifest["best_validation_rmse_m"], "Best recorded score differs")
    replayed = report["paths"]["batched"]["reference_position_rmse_m"]
    _require(
        scored.get("replayed_rmse_m") == replayed and scored.get("delta_m") == replayed - expected_score,
        "Reference score difference mismatch",
    )
    _require(
        scored.get("atol_m") == SCORE_ATOL_M and scored.get("rtol") == SCORE_RTOL, "Score tolerance differs"
    )
    _require(
        scored.get("within_registered_tolerance")
        is (abs(replayed - expected_score) <= SCORE_ATOL_M + SCORE_RTOL * abs(expected_score)),
        "Reference score tolerance outcome differs",
    )
    diagnostics = report["paths"]["streaming"].get("correction_diagnostics", {})
    _require(set(diagnostics) == {"vision", "radio"}, "Missing full-corpus correction diagnostics")
    for modality, row in diagnostics.items():
        count = row.get("applied_event_count")
        _require(
            type(count) is int and 0 <= count <= len(seeds) * scene.steps, "Correction support exceeds bound"
        )
        _require(
            row.get("correction_length", {}).get("count") == count
            and row.get("innovation_length", {}).get("count") == count,
            "Correction support differs",
        )
    return True


def evaluate_checkpoint(
    checkpoint: Path,
    manifest: Path,
    *,
    kind: str,
    device_name: str = "cuda",
    profile: str = "formal",
    source_commit: str,
    expected_checkpoint_sha256: str,
    expected_manifest_sha256: str,
    output: Path | None = None,
) -> dict:
    """Replay selected best or current final weights, with explicit provenance.

    ``source_commit`` names the checkpoint's training source, not this evaluator.
    The caller verifies that source pin against its study record. Final weights
    come from the continuation's top-level state_dict, never its embedded best.
    """
    _digest(source_commit, "checkpoint source commit", 40)
    _digest(expected_checkpoint_sha256, "expected checkpoint SHA-256")
    _digest(expected_manifest_sha256, "expected manifest SHA-256")
    _require(kind in {"best", "final"}, "Checkpoint kind must be best or final")
    checkpoint, manifest = Path(checkpoint), Path(manifest)
    output = None if output is None else Path(output)
    _require(output is None or not output.exists(), "Replay output already exists")
    started = time.perf_counter()
    try:
        metadata = _read_manifest(manifest, expected_manifest_sha256)
        config, scene = _validate_protocol(metadata, profile, device_name)
        _require(
            checkpoint.is_file() and not checkpoint.is_symlink(),
            "Checkpoint must be a regular non-symlink file",
        )
        budget = MAX_CHECKPOINT_BYTES if kind == "best" else MAX_RESUME_CHECKPOINT_BYTES
        _require(0 < checkpoint.stat().st_size <= budget, "Checkpoint exceeds replay byte budget")
        _require(
            sha256_file(checkpoint) == expected_checkpoint_sha256,
            "Checkpoint SHA-256 differs from study evidence",
        )
        sidecar = Path(str(checkpoint) + ".sha256")
        _require(
            sidecar.is_file() and not sidecar.is_symlink() and sidecar.stat().st_size <= 128,
            "Replay requires a bounded checkpoint checksum sidecar",
        )
        _require(
            sidecar.read_text(encoding="ascii").strip() == expected_checkpoint_sha256,
            "Checkpoint checksum sidecar differs from evidence",
        )
        device = resolve_device(device_name)
        if device.type == "cuda":
            torch.cuda.init()
        settings = _numerical_settings()
        with _preserve_rng():
            payload = read_checkpoint(checkpoint) if kind == "best" else read_resume_checkpoint(checkpoint)
            weights_update, expected_score = _validate_payload(payload, metadata, kind)
            state = payload["state_dict"]
            fingerprint = state_dict_fingerprint(state)
            del payload
            model = build_model(config)
            model.load_state_dict(state, strict=True)
            del state
            model = model.to(device)
            _require(model_fingerprint(model) == fingerprint, "Loaded checkpoint tensors differ")
            _require(
                sum(parameter.numel() for parameter in model.parameters()) == metadata["parameters"],
                "Checkpoint parameter count differs",
            )
            replay = _replay_model(model, metadata, device, profile)
            _require(model_fingerprint(model) == fingerprint, "Read-only replay changed model tensors")
            actual_runtime = runtime_info(device)
        _require(_numerical_settings() == settings, "Replay changed numerical settings")
        actual_score = replay["paths"]["batched"]["reference_position_rmse_m"]
        report = {
            "schema_version": 1,
            "kind": "checkpoint-validation-replay",
            "status": "completed",
            "profile": profile,
            "device": device_name,
            "runtime": actual_runtime,
            "numerical_settings": settings,
            "checkpoint_kind": kind,
            "checkpoint_source_commit": source_commit,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "state_dict_fingerprint": fingerprint,
            "weights_update": weights_update,
            "selected_best_update": metadata["best_update"],
            "model_config": metadata["model"],
            "scene_config": metadata["scene"],
            "validation_episode_seeds": metadata["validation_episode_seeds"],
            "fresh_test_consumed": False,
            "deployment_certification": False,
            "unchanged_model": True,
            "rng_preserved": True,
            "protocol": {
                "scored_tick_start": SCORED_TICK_START,
                "steps": scene.steps,
                "reference_batch_size": 8,
                "prefix_episodes": min(4, metadata["validation_episodes"]),
                "error_thresholds_m": list(ERROR_THRESHOLDS_M),
                "exceedance_operator": ">",
                "quantiles": "linear interpolation of pooled tick-level Euclidean errors",
                "reference_reduction": "original float32 per-batch squared-error sum; then divide and square root",
                "tail_reduction": "float64 Euclidean errors pooled over all scored ticks; separate distribution of per-trajectory RMSE",
                "visual_scope": "fresh valid visible captures at capture time, all ticks; not current state error",
                "strata_scope": "state errors after tick4, grouped by capture visibility, camera age and radio freshness; dimensions overlap",
                "correction_scope": "full validation corpus, streaming path, all applied fresh valid corrections including occluded captures",
                "comparison_scope": "same checkpoint and same episodes; all ticks; no training or model selection",
                "truth_scope": "evaluation labels only; excluded from model inputs",
                "threshold_scope": "descriptive error levels, not safety or deployment requirements",
            },
            "recorded_score": {
                "expected_rmse_m": expected_score,
                "replayed_rmse_m": actual_score,
                "delta_m": actual_score - expected_score,
                "atol_m": SCORE_ATOL_M,
                "rtol": SCORE_RTOL,
                "within_registered_tolerance": abs(actual_score - expected_score)
                <= SCORE_ATOL_M + SCORE_RTOL * abs(expected_score),
            },
            "evaluator_source_sha256": {
                filename: sha256_file(Path(__file__).with_name(filename))
                for filename in (
                    "checkpoint_replay.py",
                    "model.py",
                    "structured.py",
                    "simulator.py",
                    "training.py",
                )
            },
            **replay,
        }
        _require(
            sha256_file(checkpoint) == expected_checkpoint_sha256, "Checkpoint bytes changed during replay"
        )
        _require(sha256_file(manifest) == expected_manifest_sha256, "Manifest bytes changed during replay")
        report["elapsed_seconds"] = time.perf_counter() - started
        validate_replay_report(
            report,
            metadata,
            kind=kind,
            device_name=device_name,
            profile=profile,
            source_commit=source_commit,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if output is not None:
            write_json(output, report)
        return report
    except Exception as error:
        if output is not None and not output.exists():
            write_json(
                output,
                {
                    "schema_version": 1,
                    "kind": "checkpoint-validation-replay",
                    "status": "failed",
                    "checkpoint_kind": kind,
                    "checkpoint_source_commit": source_commit,
                    "checkpoint_sha256": expected_checkpoint_sha256,
                    "manifest_sha256": expected_manifest_sha256,
                    "fresh_test_consumed": False,
                    "deployment_certification": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
        raise
