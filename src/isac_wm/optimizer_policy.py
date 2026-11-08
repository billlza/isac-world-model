"""Explicit optimizer groups for a controlled width-learning-rate intervention.

The unscaled path constructs the historical single-group AdamW unchanged.
Grouping metadata describes names/shapes, never parameter values or RNG state.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import torch

from .structured import StructuredEventModel
from .validation import finite

WEIGHT_DECAY = 1e-4
FIXED_WIDTH_PREFIXES = frozenset({"visual", "radio_encoder", "action_embedding"})
WIDTH_DEPENDENT_PREFIXES = frozenset(
    {
        "fusion",
        "dynamics",
        "acceleration",
        "noise",
        "observation_head",
        "event_projection",
        "time_projection",
        "query_norm",
        "memory_norm",
        "attention",
        "null",
        "feedforward",
    }
)
ATTENTION_PREFIXES = frozenset(
    {"event_projection", "time_projection", "query_norm", "memory_norm", "attention", "null", "feedforward"}
)


def validate_width_lr_scale(value: float) -> float:
    value = finite(value, "width_lr_scale", positive=True)
    if value > 1:
        raise ValueError("width_lr_scale must be in (0, 1]")
    return value


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _inventory(model):
    named = list(model.named_parameters(remove_duplicate=False))
    if not named or len({name for name, _ in named}) != len(named):
        raise ValueError("Optimizer requires distinct named model parameters")
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("Optimizer policy does not permit aliased/shared parameters")
    return named


def _partition(model, width_lr_scale):
    named = _inventory(model)
    if width_lr_scale == 1:
        return [("all_parameters", named)]
    if not isinstance(model, StructuredEventModel):
        raise ValueError("Width-scaled optimization requires a StructuredEventModel")
    present = {name.split(".", 1)[0] for name, _ in named}
    expected = FIXED_WIDTH_PREFIXES | WIDTH_DEPENDENT_PREFIXES
    if not model.attentive:
        expected -= ATTENTION_PREFIXES
    if present != expected:
        raise ValueError(
            f"Optimizer parameter prefixes differ: missing={sorted(expected - present)}, "
            f"unexpected={sorted(present - expected)}"
        )
    fixed, scaled = [], []
    for name, parameter in named:
        prefix = name.split(".", 1)[0]
        (fixed if prefix in FIXED_WIDTH_PREFIXES else scaled).append((name, parameter))
    if not fixed or not scaled or len(fixed) + len(scaled) != len(named):
        raise ValueError("Optimizer parameter partition is incomplete")
    return [("fixed_width", fixed), ("width_dependent", scaled)]


def describe_optimizer_policy(model, learning_rate: float, width_lr_scale: float = 1.0) -> dict:
    """Return the exhaustive named policy; also works on a meta-device model."""
    learning_rate = finite(learning_rate, "learning_rate", positive=True)
    width_lr_scale = validate_width_lr_scale(width_lr_scale)
    groups = []
    for name, named in _partition(model, width_lr_scale):
        names = [key for key, _ in named]
        structure = [
            {
                "name": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "requires_grad": value.requires_grad,
            }
            for key, value in named
        ]
        groups.append(
            {
                "name": name,
                "learning_rate": learning_rate * (width_lr_scale if name == "width_dependent" else 1),
                "parameters": sum(value.numel() for _, value in named),
                "parameter_tensors": len(named),
                "prefixes": sorted({key.split(".", 1)[0] for key in names}),
                "parameter_names": names,
                "parameter_names_sha256": _digest(names),
                "parameter_structure_sha256": _digest(structure),
            }
        )
    policy = {
        "schema_version": 1,
        "optimizer": "AdamW",
        "mode": "legacy_single_group" if width_lr_scale == 1 else "width_scaled_groups",
        "base_learning_rate": learning_rate,
        "width_lr_scale": width_lr_scale,
        "weight_decay": WEIGHT_DECAY,
        "parameters": sum(group["parameters"] for group in groups),
        "parameter_tensors": sum(group["parameter_tensors"] for group in groups),
        "groups": groups,
    }
    return policy | {"sha256": _digest(policy)}


def make_optimizer(model, learning_rate: float, width_lr_scale: float = 1.0):
    policy = describe_optimizer_policy(model, learning_rate, width_lr_scale)
    if policy["width_lr_scale"] == 1:
        # Keep this construction, parameter order and grouping identical to the
        # pre-policy control. Metadata is stored outside the optimizer group.
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    else:
        parameters = dict(_inventory(model))
        groups = [
            {
                "params": [parameters[name] for name in group["parameter_names"]],
                "lr": group["learning_rate"],
                "name": group["name"],
                "param_names": group["parameter_names"].copy(),
            }
            for group in policy["groups"]
        ]
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    validate_live_optimizer(optimizer, model, policy)
    return optimizer, policy


def validate_checkpoint_optimizer_policy(checkpoint: dict, policy: dict, *, allow_legacy=False) -> bool:
    """Return whether a genuinely metadata-free legacy checkpoint was accepted."""
    if "optimizer_policy" not in checkpoint:
        if allow_legacy and policy["width_lr_scale"] == 1 and "width_lr_scale" not in checkpoint:
            return True
        raise ValueError("Resume optimizer policy is missing")
    saved = checkpoint["optimizer_policy"]
    try:
        valid = (
            isinstance(saved, dict)
            and saved.get("sha256")
            == _digest({key: value for key, value in saved.items() if key != "sha256"})
            and _json(saved) == _json(policy)
            and validate_width_lr_scale(checkpoint["width_lr_scale"]) == policy["width_lr_scale"]
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Resume optimizer policy differs from the requested run")
    return False


def _validate_group_options(group, expected, defaults, *, legacy=False):
    lr = group.get("lr")
    if (
        isinstance(lr, bool)
        or not isinstance(lr, Real)
        or not math.isfinite(lr)
        or lr != expected["learning_rate"]
    ):
        raise ValueError("Resume optimizer learning rate differs from the registered group")
    for key, value in defaults.items():
        if key == "lr":
            continue
        # Older torch state may omit flags filled by AdamW.__setstate__. Core
        # optimization hyperparameters must always be present and equal.
        if key not in group and legacy and key not in {"betas", "eps", "weight_decay", "amsgrad"}:
            continue
        actual = group.get(key)
        if key == "betas" and isinstance(actual, (tuple, list)):
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"Resume optimizer option differs: {key}")
    if expected["name"] == "all_parameters":
        if "name" in group or "param_names" in group:
            raise ValueError("Legacy optimizer must remain a single unnamed parameter group")
    elif group.get("name") != expected["name"] or group.get("param_names") != expected["parameter_names"]:
        raise ValueError("Resume optimizer group names or parameter order differ")


def _validate_moments(state, parameter):
    if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
        raise ValueError("Resume optimizer moment state is incomplete")
    for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if field not in state:
            continue
        value = state[field]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
        ):
            raise ValueError("Resume optimizer moment shape or dtype differs from the named parameter")
    step = state["step"]
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise ValueError("Resume optimizer step must be a scalar")
        step = step.item()
    if isinstance(step, bool) or not isinstance(step, Real) or not math.isfinite(step) or step < 0:
        raise ValueError("Resume optimizer step is invalid")


def validate_live_optimizer(optimizer, model, policy) -> None:
    named = dict(_inventory(model))
    if len(optimizer.param_groups) != len(policy["groups"]):
        raise ValueError("Resume optimizer group count differs")
    registered = set()
    for actual, expected in zip(optimizer.param_groups, policy["groups"]):
        _validate_group_options(actual, expected, optimizer.defaults)
        parameters = [named[name] for name in expected["parameter_names"]]
        if [id(value) for value in actual["params"]] != [id(value) for value in parameters]:
            raise ValueError("Resume optimizer live parameter count or order differs")
        if sum(value.numel() for value in parameters) != expected["parameters"]:
            raise ValueError("Resume optimizer live parameter budget differs")
        registered.update(parameters)
    if not set(optimizer.state).issubset(registered):
        raise ValueError("Resume optimizer contains unregistered parameter state")
    for parameter, state in optimizer.state.items():
        _validate_moments(state, parameter)


def restore_optimizer(optimizer, model, checkpoint: dict, policy: dict) -> bool:
    """Check saved grouping before load and the actual live optimizer before use."""
    legacy = validate_checkpoint_optimizer_policy(checkpoint, policy, allow_legacy=True)
    saved = checkpoint.get("optimizer")
    if (
        not isinstance(saved, dict)
        or not isinstance(saved.get("param_groups"), list)
        or not isinstance(saved.get("state"), dict)
        or len(saved["param_groups"]) != len(policy["groups"])
    ):
        raise ValueError("Resume optimizer group count or schema differs")
    named, by_index, offset = dict(_inventory(model)), {}, 0
    for actual, expected in zip(saved["param_groups"], policy["groups"]):
        if not isinstance(actual, dict):
            raise ValueError("Resume optimizer group is invalid")
        _validate_group_options(actual, expected, optimizer.defaults, legacy=legacy)
        count = expected["parameter_tensors"]
        indices = actual.get("params")
        if (
            not isinstance(indices, list)
            or any(type(index) is not int for index in indices)
            or indices != list(range(offset, offset + count))
        ):
            raise ValueError("Resume optimizer parameter indices, count or order differ")
        by_index.update({offset + i: named[name] for i, name in enumerate(expected["parameter_names"])})
        offset += count
    if any(type(index) is not int for index in saved["state"]) or not set(saved["state"]).issubset(by_index):
        raise ValueError("Resume optimizer contains unexpected state indices")
    for index, state in saved["state"].items():
        _validate_moments(state, by_index[index])
    optimizer.load_state_dict(saved)
    validate_live_optimizer(optimizer, model, policy)
    return legacy
