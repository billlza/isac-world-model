"""Reference-aware supervised pretraining, including differentiable displacement.

This objective is not RLHF, a learned process reward model, or a task-completion
certificate. It keeps current-state, future-state, sensor-prediction and visual
losses explicit. The displacement term supplies an auxiliary motion-consistency
gradient; it never rewards the number of reasoning steps or detached diagnostics.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .model import gaussian_nll, visual_supervision_loss
from .validation import finite, integer


def _require_tensor(mapping: dict[str, Tensor], name: str) -> Tensor:
    value = mapping.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"Missing tensor {name}")
    return value


def _binary_mask(mask: Tensor, shape: torch.Size | tuple[int, ...], name: str) -> Tensor:
    if mask.shape != shape or mask.requires_grad:
        raise ValueError(f"{name} must be an external binary mask with shape {tuple(shape)}")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} must be binary and finite")
    return mask.bool()


def _prediction(output: dict[str, Tensor], name: str, target: Tensor) -> Tensor:
    value = _require_tensor(output, name)
    if value.shape != target.shape or value.device != target.device or not value.is_floating_point():
        raise ValueError(f"{name} must match the floating reference shape and device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _reference_fields(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    target = _require_tensor(batch, "state_target")
    variance = _require_tensor(batch, "state_reference_variance")
    validity = _require_tensor(batch, "state_reference_validity")
    if target.ndim != 3 or target.shape[-1] != 4 or not target.is_floating_point():
        raise ValueError("state_target must have floating shape [batch,time,4]")
    if not target.shape[0] or not target.shape[1]:
        raise ValueError("state_target cannot have empty batch or time dimensions")
    if target.requires_grad or variance.requires_grad:
        raise ValueError("State references and reference variance must be external, not learned confidence")
    if variance.shape != target.shape or not variance.is_floating_point():
        raise ValueError("state_reference_variance must have floating shape [batch,time,4]")
    if variance.device != target.device or validity.device != target.device:
        raise ValueError("State reference fields must share a device")
    active = _binary_mask(validity, target.shape, "state_reference_validity")
    # Unknown references may be NaN only when explicitly invalid. Select before
    # doing arithmetic: multiplying an already-NaN residual by zero is unsafe.
    if not bool(torch.isfinite(target[active]).all()):
        raise ValueError("Valid state references must be finite")
    if not bool((torch.isfinite(variance[active]) & (variance[active] >= 0)).all()):
        raise ValueError("Valid reference variances must be finite and non-negative")
    return target, variance, active


def _reference_nll(
    mean: Tensor, logvar: Tensor, target: Tensor, reference_variance: Tensor, active: Tensor
) -> tuple[Tensor, int]:
    """Coordinate-mean NLL after adding known independent reference variance."""
    count = int(active.sum().item())
    if not count:
        return (mean.sum() + logvar.sum()) * 0.0, 0
    # logaddexp avoids overflow when summing predicted/reference variances.
    # A zero external variance has log -inf and recovers ordinary Gaussian NLL.
    total_logvar = torch.logaddexp(logvar[active], reference_variance[active].log())
    residual = target[active] - mean[active]
    loss = 0.5 * (total_logvar + residual.square() * torch.exp(-total_logvar))
    return loss.mean(), count


def _observation_loss(
    output: dict[str, Tensor], batch: dict[str, Tensor], state: Tensor
) -> tuple[Tensor, int]:
    target = _require_tensor(batch, "radio_target")
    validity = _require_tensor(batch, "radio_target_valid")
    if target.shape != state.shape or target.device != state.device or not target.is_floating_point():
        raise ValueError("radio_target must have floating shape [batch,time,4] on the state device")
    if target.requires_grad or validity.device != target.device:
        raise ValueError("Radio supervision must be external and on one device")
    active = _binary_mask(validity, target.shape[:-1], "radio_target_valid")
    if not bool(torch.isfinite(target[active]).all()):
        raise ValueError("Valid radio targets must be finite")
    mean, logvar = (_prediction(output, key, target) for key in ("obs_mean", "obs_logvar"))
    count = int(active.sum().item())
    if not count:
        return (mean.sum() + logvar.sum()) * 0.0, 0
    return gaussian_nll(mean[active], logvar[active], target[active]).mean(), count


def _displacement_loss(
    mean: Tensor,
    future_mean: Tensor,
    target: Tensor,
    variance: Tensor,
    active: Tensor,
    horizon: int,
) -> tuple[Tensor, int]:
    """Compare forecast displacement to reference displacement, once per pair.

    A shared position/state offset cancels instead of being penalized a second
    time as a process error. Real turns are supervised through their actual
    displacement; there is no constant-velocity or smoothness constraint.

    The externally fixed weight is 1 / (1 + var_now + var_future), with unit
    squared-normalized-state scale. This is a bounded reference-aware auxiliary
    MSE, not an extra independent likelihood and not a calibrated probability.
    No predicted confidence participates in its weight or support selection.
    """
    pair_active = active[:, :-horizon] & active[:, horizon:]
    count = int(pair_active.sum().item())
    if not count:
        return (mean.sum() + future_mean.sum()) * 0.0, 0
    predicted_delta = future_mean[:, :-horizon][pair_active] - mean[:, :-horizon][pair_active]
    reference_delta = target[:, horizon:][pair_active] - target[:, :-horizon][pair_active]
    reference_variance = variance[:, horizon:][pair_active] + variance[:, :-horizon][pair_active]
    return ((predicted_delta - reference_delta).square() / (1.0 + reference_variance)).mean(), count


def pretraining_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    horizon_steps: int = 3,
    *,
    vision_weight: float = 1.0,
    observation_weight: float = 0.15,
    process_weight: float = 0.1,
) -> tuple[Tensor, dict[str, float | int]]:
    """Return a differentiable loss and separately reported scalar components.

    State and future NLLs marginalize independent, zero-mean reference error by
    adding its externally supplied variance to the predictive variance. This
    does not correct biased or wrong labels: such references need repair or an
    external invalidity mask. Variance is in squared normalized state units and
    validity is coordinate-wise ``[batch,time,4]``. Neither field is a model input.

    Total = state + 0.5 * future + observation_weight * observation
            + vision_weight * capture_xy_MSE + process_weight * displacement_MSE.

    Each component is averaged over its own valid support, not summed per token
    or per episode length. The displacement auxiliary is correlated with state
    supervision and is reported as such, not counted as independent evidence or
    as another successful task. A missing-support term is differentiable zero.
    Observation/visual terms with zero weight need not have their labels present.
    """
    vision_weight = finite(vision_weight, "vision_weight")
    observation_weight = finite(observation_weight, "observation_weight")
    process_weight = finite(process_weight, "process_weight")
    target, reference_variance, active = _reference_fields(batch)
    horizon_steps = integer(horizon_steps, "horizon_steps", 1, target.shape[1] - 1)
    mean, logvar, future_mean, future_logvar = (
        _prediction(output, key, target) for key in ("mean", "logvar", "future_mean", "future_logvar")
    )
    current, current_count = _reference_nll(mean, logvar, target, reference_variance, active)
    future, future_count = _reference_nll(
        future_mean[:, :-horizon_steps],
        future_logvar[:, :-horizon_steps],
        target[:, horizon_steps:],
        reference_variance[:, horizon_steps:],
        active[:, horizon_steps:],
    )
    process, process_count = _displacement_loss(
        mean, future_mean, target, reference_variance, active, horizon_steps
    )
    zero = current * 0.0
    observation, observation_count = (
        _observation_loss(output, batch, target) if observation_weight else (zero, 0)
    )
    visual, visual_count = visual_supervision_loss(output, batch) if vision_weight else (zero, 0)

    weighted_future = 0.5 * future
    loss = current + weighted_future
    for weight, term in (
        (observation_weight, observation),
        (vision_weight, visual),
        (process_weight, process),
    ):
        if weight:
            loss = loss + weight * term
    details: dict[str, float | int] = {
        "loss": float(loss.detach()),
        "state_nll": float(current.detach()),
        "future_nll": float(future.detach()),
        "observation_nll": float(observation.detach()),
        "vision_mse_normalized": float(visual.detach()),
        "process_displacement_mse": float(process.detach()),
        "weighted_state_loss": float(current.detach()),
        "weighted_future_loss": float(weighted_future.detach()),
        "weighted_observation_loss": float((observation_weight * observation).detach()),
        "weighted_vision_loss": float((vision_weight * visual).detach()),
        "weighted_process_loss": float((process_weight * process).detach()),
        "state_reference_coordinate_count": current_count,
        "future_reference_coordinate_count": future_count,
        "process_reference_pair_coordinate_count": process_count,
        "observation_target_count": observation_count,
        "vision_visible_capture_count": visual_count,
    }
    return loss, details
