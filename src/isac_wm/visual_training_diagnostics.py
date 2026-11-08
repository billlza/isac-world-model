"""Bounded, passive observers for the visual-supervision experiment.

These observations never select a checkpoint, change inputs, call backward, or
retain an autograd graph. Parameter deltas describe the actual optimizer step.
"""

from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from .model import training_loss, visual_supervision_loss
from .validation import integer

DEFAULT_DIAGNOSTIC_UPDATES = (0, 1, 2, 4, 8, 16, 32, 64, 96, 192, 384)
MAX_DIAGNOSTIC_EPISODES = 4
MAX_DIAGNOSTIC_STEPS = 128
MAX_DIAGNOSTIC_SNAPSHOTS = 64
VISION_SUPERVISION_PROTOCOL = {
    "schema_version": 1,
    "target": "normalized xy at the accepted image event's capture time",
    "mask": "new and valid and visible_at_capture",
    "normalization": "mean squared error over coordinates, then visible captures",
    "empty_support": "zero loss and zero count",
    "online_input_keys_unchanged": True,
}


def validate_schedule(updates: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if updates is None:
        return None
    if not isinstance(updates, tuple) or not 0 < len(updates) <= MAX_DIAGNOSTIC_SNAPSHOTS:
        raise ValueError("diagnostic_updates must be a non-empty bounded tuple")
    checked = tuple(integer(value, "diagnostic update", 0) for value in updates)
    if tuple(sorted(set(checked))) != checked:
        raise ValueError("diagnostic_updates must be strictly increasing without duplicates")
    return checked


def diagnostic_protocol(updates, validation_seeds, steps) -> dict | None:
    if updates is None:
        return None
    return {
        "schema_version": 1,
        "updates": list(updates),
        "validation_episode_seeds": list(validation_seeds[:MAX_DIAGNOSTIC_EPISODES]),
        "max_validation_episodes": MAX_DIAGNOSTIC_EPISODES,
        "validation_steps": min(steps, MAX_DIAGNOSTIC_STEPS),
        "scope": "diagnostic-only; not checkpoint selection",
        "saturation_threshold": 0.99,
        "gradient_scope": "current training minibatch, before and after norm clipping",
        "parameter_update_scope": "actual optimizer step; includes AdamW weight decay",
    }


def event_counts(batch: dict[str, torch.Tensor]) -> dict[str, int]:
    """Actual delivered-evidence counts, independent of latent width or GPU time."""
    active = (batch["new"] * batch["valid"]).bool()
    updates = active.any(-1)
    return {
        "snapshots": updates.numel(),
        "vision_new_valid": int(active[..., 0].sum().item()),
        "radio_new_valid": int(active[..., 1].sum().item()),
        "evidence_update_ticks": int(updates.sum().item()),
        "prediction_only_ticks": int((~updates).sum().item()),
        "vision_visible_captures": int(batch["vision_target_mask"].sum().item()),
    }


@contextmanager
def passive_observer(model):
    """Restore every module mode and Python/NumPy/Torch RNG even on failure."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps_rng = None
    if next(model.parameters()).device.type == "mps":
        mps_rng = torch.mps.get_rng_state()
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, was_training in modes:
            module.training = was_training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


def _rms(value):
    return float(value.double().square().mean().sqrt().item()) if value.numel() else None


def _summary(value):
    if not value.numel():
        return {"count": 0, "mean": None, "rms": None, "max_abs": None}
    value = value.double()
    return {
        "count": value.numel(),
        "mean": float(value.mean().item()),
        "rms": _rms(value),
        "max_abs": float(value.abs().max().item()),
    }


def validation_snapshot(model, batch, *, room_m, dt, vision_loss_weight) -> dict:
    """Replay a fixed validation prefix, retaining only scalar aggregates.

    Visual RMSE uses visible capture-time labels. Measurement-variance summaries
    include only genuinely applied fresh-vision corrections, including occlusion.
    Network summaries cover all snapshot ticks; residual ratios also report the
    subset where an evidence update was actually used.
    """
    if not model.supports_visual_supervision:
        raise ValueError("Visual diagnostics require a structured spatial encoder")
    if batch["image"].shape[0] > MAX_DIAGNOSTIC_EPISODES:
        raise ValueError("Visual diagnostic batch exceeds the episode bound")
    if batch["image"].shape[1] > MAX_DIAGNOSTIC_STEPS:
        raise ValueError("Visual diagnostic batch exceeds the prefix bound")
    started = time.perf_counter()
    with passive_observer(model):
        n, steps = batch["image"].shape[:2]
        memory = model.initial(n, batch["image"].device)
        output = {
            key: []
            for key in (
                "mean",
                "logvar",
                "future_mean",
                "future_logvar",
                "obs_mean",
                "obs_logvar",
                "visual_xy",
                "visual_entropy",
            )
        }
        network, variances, learned_variances, entropy = {}, [], [], []
        corrections = {
            modality: {key: [] for key in ("innovation", "gain", "position_delta")}
            for modality in ("vision", "radio")
        }
        for tick in range(steps):
            inputs = {key: batch[key][:, tick] for key in ("image", "radio", "age", "valid", "new", "dt")}
            encoded = model.encode_new(inputs)
            observed = {}
            state, memory = model.step(inputs, memory, encoded, diagnostics=observed)
            future = model.forecast(state["hidden"], 3 * dt)
            radio = model.predict_observation(state["hidden"], batch["action_target"][:, tick])
            for key in ("mean", "logvar"):
                output[key].append(state[key])
                output["future_" + key].append(future[key])
                output["obs_" + key].append(radio[key])
            output["visual_xy"].append(encoded[0][:, :2])
            output["visual_entropy"].append(encoded[0][:, 2])
            for key, value in observed["network"].items():
                network.setdefault(key, []).append(value)
            visual = observed["vision"]
            active = visual["active"][:, 0]
            variances.append(visual["measurement_variance_xy"][active])
            learned_variances.append(visual["learned_measurement_variance_xy"][active])
            entropy.append(visual["heatmap_entropy"][active])
            for modality in ("vision", "radio"):
                correction = observed[modality]
                applied = correction["active"][:, 0]
                corrections[modality]["innovation"].append(correction["innovation_xy"][applied])
                corrections[modality]["gain"].append(correction["gain"][applied])
                corrections[modality]["position_delta"].append(
                    correction["applied_state_correction"][applied, :2]
                )
        predicted = {key: torch.stack(values, 1) for key, values in output.items()}
        _, losses = training_loss(predicted, batch, vision_loss_weight=vision_loss_weight)
        visual_mse, count = visual_supervision_loss(predicted, batch)
        # Always expose the auxiliary score in diagnostics, including the control
        # arm where its coefficient is zero and it does not enter optimization.
        losses.update(
            vision_mse_normalized=float(visual_mse.item()),
            vision_visible_capture_count=count,
            weighted_vision_loss=float((vision_loss_weight * visual_mse).item()),
        )
        net = {key: torch.cat(values, 0) for key, values in network.items()}
        hidden_post_norm_rms = _rms(net["hidden"])
        evidence = net["evidence_update"][:, 0].bool()
        residual, fusion = net["attention_residual"][evidence], net["fusion_hidden"][evidence]
        residual_rms, fusion_rms = _rms(residual), _rms(fusion)
        noise, acceleration = net["noise_logits"], net["acceleration_logits"]
        variance, learned_variance = torch.cat(variances, 0), torch.cat(learned_variances, 0)
        measurement_mean = float(variance.mean().item()) if variance.numel() else None
        correction_summary = {}
        for modality, fields in corrections.items():
            values = {key: torch.cat(items, 0) for key, items in fields.items()}
            count_applied = values["position_delta"].shape[0]
            correction_summary[modality] = {
                "applied_event_count": count_applied,
                "position_correction_rms_m": (
                    float(values["position_delta"].square().sum(-1).mean().sqrt().item()) * room_m
                    if count_applied
                    else None
                ),
                "innovation_rms_m": (
                    float(values["innovation"].square().sum(-1).mean().sqrt().item()) * room_m
                    if count_applied
                    else None
                ),
                "gain_rms": _rms(values["gain"]),
                "position_gain_frobenius_rms": (
                    float(values["gain"][:, :2].square().sum((-2, -1)).mean().sqrt().item())
                    if count_applied
                    else None
                ),
                "scope": "applied fresh valid corrections; gain uses normalized state and xy",
            }
        pos_diff = predicted["mean"][:, 5:, :2] - batch["state_target"][:, 5:, :2]
        result = {
            "visual_position_rmse_m": (float((2 * visual_mse).sqrt().item()) * room_m if count else None),
            "visual_visible_capture_count": count,
            "state_position_rmse_m": float(pos_diff.square().sum(-1).mean().sqrt().item()) * room_m,
            "event_counts": event_counts(batch),
            "losses": losses,
            "corrections": correction_summary,
            "network": {
                "hidden_rms": hidden_post_norm_rms,
                "hidden_pre_norm_rms": _rms(net["hidden_pre_norm"]),
                "hidden_post_norm_rms": hidden_post_norm_rms,
                "hidden_scope": "post-normalization latent used by readouts and stored in recurrent memory",
                "prior_hidden_rms": _rms(net["prior_hidden"]),
                "fusion_hidden_rms_on_evidence": fusion_rms,
                "attention_residual_rms": residual_rms,
                "attention_residual_to_hidden_rms": (
                    residual_rms / max(fusion_rms, 1e-12) if fusion_rms is not None else None
                ),
                "attention_residual_scope": "attention plus feedforward residual, applied evidence ticks",
            },
            "measurement": {
                "vision_variance_mean_normalized": measurement_mean,
                "vision_variance_mean_m2": None if measurement_mean is None else measurement_mean * room_m**2,
                "learned_vision_variance": _summary(learned_variance),
                "vision_heatmap_entropy": _summary(torch.cat(entropy, 0)),
                "scope": "fresh valid vision corrections, including occluded captures",
            },
            "logits": {
                "noise": _summary(noise),
                "measurement_noise": _summary(noise[:, 2:]),
                "noise_sigmoid_upper_saturation_fraction": float(
                    (noise[:, 2:].sigmoid() >= 0.99).float().mean().item()
                ),
                "noise_sigmoid_lower_saturation_fraction": float(
                    (noise[:, 2:].sigmoid() <= 0.01).float().mean().item()
                ),
                "vision_noise_sigmoid_upper_saturation_fraction": float(
                    (noise[:, 2:4].sigmoid() >= 0.99).float().mean().item()
                ),
                "vision_noise_sigmoid_lower_saturation_fraction": float(
                    (noise[:, 2:4].sigmoid() <= 0.01).float().mean().item()
                ),
                "radio_noise_sigmoid_upper_saturation_fraction": float(
                    (noise[:, 4:].sigmoid() >= 0.99).float().mean().item()
                ),
                "radio_noise_sigmoid_lower_saturation_fraction": float(
                    (noise[:, 4:].sigmoid() <= 0.01).float().mean().item()
                ),
                "acceleration": _summary(acceleration),
                "acceleration_tanh_saturation_fraction": float(
                    (acceleration.tanh().abs() >= 0.99).float().mean().item()
                ),
                "acceleration_abs_mean_mps2": float((0.5 * acceleration.tanh()).abs().mean().item()),
                "scope": "output heads evaluated at all diagnostic snapshot ticks",
            },
        }
    result["observer_seconds"] = time.perf_counter() - started
    return result


def _branch(name):
    prefix = name.split(".", 1)[0]
    if prefix in {
        "attention",
        "feedforward",
        "event_projection",
        "time_projection",
        "query_norm",
        "memory_norm",
        "null",
    }:
        return "attention"
    if prefix in {"observation_head", "action_embedding"}:
        return "observation"
    return prefix


@torch.no_grad()
def begin_parameter_update(model) -> dict:
    """Capture one selected pre-clip gradient and pre-step parameter snapshot."""
    tensors, sums = {}, {}
    for name, parameter in model.named_parameters():
        branch = _branch(name)
        row = sums.setdefault(branch, {"parameters": 0, "parameter_squared": 0.0, "gradient_squared": 0.0})
        row["parameters"] += parameter.numel()
        row["parameter_squared"] += float(parameter.detach().double().square().sum().item())
        if parameter.grad is not None:
            row["gradient_squared"] += float(parameter.grad.detach().double().square().sum().item())
        tensors[name] = parameter.detach().clone()
    return {"before": tensors, "sums": sums}


@torch.no_grad()
def finish_parameter_update(model, before: dict) -> dict:
    """Summarize actual deltas; discard the bounded pre-step tensor copy."""
    sums = before["sums"]
    for name, parameter in model.named_parameters():
        row = sums[_branch(name)]
        delta = parameter.detach() - before["before"][name]
        row["update_squared"] = row.get("update_squared", 0.0) + float(delta.double().square().sum().item())
        if parameter.grad is not None:
            row["clipped_gradient_squared"] = row.get("clipped_gradient_squared", 0.0) + float(
                parameter.grad.detach().double().square().sum().item()
            )
    result = {}
    for name, row in sums.items():
        parameter_norm, update_norm = row["parameter_squared"] ** 0.5, row["update_squared"] ** 0.5
        result[name] = {
            "parameters": row["parameters"],
            "parameter_norm": parameter_norm,
            "gradient_norm": row["gradient_squared"] ** 0.5,
            "clipped_gradient_norm": row.get("clipped_gradient_squared", 0.0) ** 0.5,
            "update_norm": update_norm,
            "relative_update": update_norm / max(parameter_norm, 1e-12),
        }
    before.clear()
    return result
