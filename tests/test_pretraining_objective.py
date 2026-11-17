"""Known-reference variance, invalid support, useful process gradients and causality."""

import math

import pytest
import torch

from isac_wm.model import gaussian_nll
from isac_wm.pretraining_data import make_episode
from isac_wm.pretraining_objective import pretraining_loss
from isac_wm.simulator import SceneConfig
from isac_wm.structured import StructuredConfig, build_model
from isac_wm.training import batchify


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _example(steps=6):
    shape = (2, steps, 4)
    batch = {
        "state_target": torch.linspace(-0.2, 0.8, math.prod(shape)).reshape(shape),
        "state_reference_variance": torch.zeros(shape),
        "state_reference_validity": torch.ones(shape),
        "radio_target": torch.full(shape, -0.1),
        "radio_target_valid": torch.ones(shape[:-1]),
    }
    output = {
        "mean": torch.full(shape, 0.4, requires_grad=True),
        "logvar": torch.full(shape, math.log(0.5), requires_grad=True),
        "future_mean": torch.full(shape, 0.7, requires_grad=True),
        "future_logvar": torch.full(shape, math.log(0.8), requires_grad=True),
        "obs_mean": torch.full(shape, 0.2, requires_grad=True),
        "obs_logvar": torch.full(shape, math.log(1.5), requires_grad=True),
    }
    return output, batch


def _state_loss(output, batch, **kwargs):
    return pretraining_loss(output, batch, vision_weight=0, observation_weight=0, **kwargs)


def test_exact_references_recover_the_original_state_future_and_observation_losses():
    output, batch = _example()
    loss, details = pretraining_loss(output, batch, vision_weight=0, process_weight=0)
    current = gaussian_nll(output["mean"], output["logvar"], batch["state_target"]).mean()
    future = gaussian_nll(
        output["future_mean"][:, :-3], output["future_logvar"][:, :-3], batch["state_target"][:, 3:]
    ).mean()
    observation = gaussian_nll(output["obs_mean"], output["obs_logvar"], batch["radio_target"]).mean()
    torch.testing.assert_close(loss, current + 0.5 * future + 0.15 * observation)
    assert details["state_reference_coordinate_count"] == 48
    assert details["future_reference_coordinate_count"] == 24
    assert details["observation_target_count"] == 12


def test_external_reference_variance_is_added_and_controls_the_correct_gradient():
    output, batch = _example()
    batch["state_reference_variance"] = torch.linspace(0, 1, 48).reshape(2, 6, 4)
    loss, details = _state_loss(output, batch, process_weight=0)
    reference_variance = batch["state_reference_variance"]
    variance = output["logvar"].exp() + reference_variance
    future_variance = output["future_logvar"][:, :-3].exp() + reference_variance[:, 3:]
    expected_current = (
        0.5 * (variance.log() + (output["mean"] - batch["state_target"]).square() / variance).mean()
    )
    expected_future = (
        0.5
        * (
            future_variance.log()
            + (output["future_mean"][:, :-3] - batch["state_target"][:, 3:]).square() / future_variance
        ).mean()
    )
    torch.testing.assert_close(loss, expected_current + 0.5 * expected_future)
    assert details["state_nll"] == pytest.approx(float(expected_current.detach()))
    loss.backward()
    expected_gradient = (output["mean"].detach() - batch["state_target"]) / variance.detach() / 48
    torch.testing.assert_close(output["mean"].grad, expected_gradient)


def test_invalid_coordinates_with_nan_labels_are_excluded_before_arithmetic():
    output, batch = _example()
    batch["state_reference_validity"][..., 2] = 0
    batch["state_reference_validity"][0, 0] = 0
    active = batch["state_reference_validity"].bool()
    batch["state_target"][~active] = float("nan")
    batch["state_reference_variance"][~active] = float("nan")
    batch["radio_target_valid"][:, :2] = 0
    batch["radio_target"][:, :2] = float("nan")
    loss, details = pretraining_loss(output, batch, vision_weight=0)
    assert torch.isfinite(loss)
    assert details["state_reference_coordinate_count"] == int(active.sum())
    assert details["future_reference_coordinate_count"] == int(active[:, 3:].sum())
    assert details["process_reference_pair_coordinate_count"] == int((active[:, :-3] & active[:, 3:]).sum())
    loss.backward()
    for prediction in output.values():
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(output["mean"].grad[~active]) == 0
    assert torch.count_nonzero(output["future_mean"].grad[:, :-3][~active[:, 3:]]) == 0
    assert torch.count_nonzero(output["obs_mean"].grad[:, :2]) == 0


def test_entirely_invalid_supervision_returns_differentiable_zero():
    output, batch = _example()
    for key in ("state_target", "state_reference_variance", "radio_target"):
        batch[key].fill_(float("nan"))
    batch["state_reference_validity"].zero_()
    batch["radio_target_valid"].zero_()
    loss, details = pretraining_loss(output, batch, vision_weight=0)
    assert float(loss.detach()) == 0
    assert all(value == 0 for key, value in details.items() if key.endswith("count"))
    loss.backward()
    for prediction in output.values():
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
        assert torch.count_nonzero(prediction.grad) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_reference_variance", -1),
        ("state_reference_variance", float("nan")),
        ("state_reference_validity", 0.5),
        ("state_reference_validity", float("nan")),
        ("state_target", float("nan")),
    ],
)
def test_invalid_active_reference_metadata_is_rejected(field, value):
    output, batch = _example()
    batch[field][0, 0, 0] = value
    with pytest.raises(ValueError):
        _state_loss(output, batch)


@pytest.mark.parametrize("field", ["state_target", "state_reference_variance", "state_reference_validity"])
def test_model_learned_reference_quality_cannot_hide_supervision(field):
    output, batch = _example()
    batch[field].requires_grad_()
    with pytest.raises(ValueError, match="external|learned"):
        _state_loss(output, batch)


@pytest.mark.parametrize("horizon", [0, 6, -1, True, 1.2])
def test_horizon_must_be_an_integer_inside_the_sequence(horizon):
    output, batch = _example()
    with pytest.raises(ValueError, match="horizon_steps"):
        _state_loss(output, batch, horizon_steps=horizon)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vision_weight", -1),
        ("observation_weight", float("nan")),
        ("process_weight", True),
    ],
)
def test_invalid_loss_weights_are_rejected(field, value):
    output, batch = _example()
    with pytest.raises(ValueError, match=field):
        pretraining_loss(output, batch, **{field: value})


@pytest.mark.parametrize("horizon", [3, 10, 20])
def test_all_runner_horizons_align_future_reference_support(horizon):
    output, batch = _example(steps=24)
    _, details = _state_loss(output, batch, horizon_steps=horizon)
    assert details["future_reference_coordinate_count"] == 2 * (24 - horizon) * 4
    assert details["process_reference_pair_coordinate_count"] == 2 * (24 - horizon) * 4


@pytest.mark.parametrize("shared_offset", [0.0, 0.3])
def test_process_supervision_preserves_real_turns_and_does_not_repenalize_shared_bias(shared_offset):
    output, batch = _example()
    turning = torch.tensor([0.0, 0.4, -0.1, 0.8, -0.4, 0.2]).reshape(1, 6, 1).expand(2, 6, 4)
    batch["state_target"] = turning.clone()
    output["mean"] = (turning + shared_offset).clone().requires_grad_()
    shifted = turning.clone()
    shifted[:, :-3] = turning[:, 3:]
    output["future_mean"] = (shifted + shared_offset).requires_grad_()
    _, details = _state_loss(output, batch)
    assert details["process_displacement_mse"] == pytest.approx(0, abs=1e-13)
    assert details["weighted_process_loss"] == pytest.approx(0, abs=1e-13)


def test_process_auxiliary_has_nonzero_gradients_through_both_predictions():
    output, batch = _example()
    baseline, _ = _state_loss(output, batch, process_weight=0)
    process_enabled, _ = _state_loss(output, batch, process_weight=0.4)
    gradients = torch.autograd.grad(process_enabled - baseline, (output["mean"], output["future_mean"]))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    torch.testing.assert_close(gradients[0][:, :-3], -gradients[1][:, :-3])


def test_process_weight_uses_only_external_variance_not_predicted_confidence():
    output, batch = _example()
    _, first = _state_loss(output, batch)
    output["logvar"] = torch.full_like(output["logvar"], 5.0)
    output["future_logvar"] = torch.full_like(output["future_logvar"], 5.0)
    output["confidence"] = torch.tensor(1e-20)
    _, changed_predictions = _state_loss(output, batch)
    assert changed_predictions["process_displacement_mse"] == first["process_displacement_mse"]
    batch["state_reference_variance"].fill_(3)
    _, uncertain_reference = _state_loss(output, batch)
    assert uncertain_reference["process_displacement_mse"] == pytest.approx(
        first["process_displacement_mse"] / 7
    )


def test_duplicating_examples_does_not_multiply_any_loss_or_process_reward():
    output, batch = _example()
    original, original_details = pretraining_loss(output, batch, vision_weight=0)
    repeated_output = {key: value.repeat(2, 1, 1) for key, value in output.items()}
    repeated_batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in batch.items()}
    repeated, repeated_details = pretraining_loss(repeated_output, repeated_batch, vision_weight=0)
    torch.testing.assert_close(original, repeated)
    assert repeated_details["process_displacement_mse"] == pytest.approx(
        original_details["process_displacement_mse"]
    )
    assert (
        repeated_details["state_reference_coordinate_count"]
        == 2 * original_details["state_reference_coordinate_count"]
    )


@pytest.fixture
def actual_model_batch():
    torch.manual_seed(31)
    scene = SceneConfig(steps=12, image_size=16, camera_period=1, camera_dropout=0, occlusion_fraction=0)
    batch = batchify([make_episode(seed, scene, appearance=True) for seed in (51, 52)], torch.device("cpu"))
    model = build_model(StructuredConfig(feature_dim=16, hidden_dim=32, memory_size=4))
    return model, batch


def test_full_objective_reaches_visual_and_process_parameters_and_reports_exact_sum(actual_model_batch):
    model, batch = actual_model_batch
    output = model(batch, forecast_s=0.3, batched_encoding=True, return_visual=True)
    baseline, _ = pretraining_loss(output, batch, process_weight=0)
    loss, details = pretraining_loss(output, batch)
    process_gradients = torch.autograd.grad(
        loss - baseline, tuple(model.parameters()), retain_graph=True, allow_unused=True
    )
    assert any(gradient is not None and bool(torch.count_nonzero(gradient)) for gradient in process_gradients)
    assert all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in process_gradients)
    weighted = sum(
        details[key]
        for key in (
            "weighted_state_loss",
            "weighted_future_loss",
            "weighted_observation_loss",
            "weighted_vision_loss",
            "weighted_process_loss",
        )
    )
    assert details["loss"] == pytest.approx(weighted, abs=2e-6)
    assert details["vision_visible_capture_count"] > 0
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.visual.parameters()
    )
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_quality_labels_do_not_enter_online_inputs_and_future_changes_leave_prefix_fixed(actual_model_batch):
    model, batch = actual_model_batch
    original = model(batch, forecast_s=0.3, batched_encoding=True, return_visual=True)
    changed = {key: value.clone() for key, value in batch.items()}
    for key in (
        "state_target",
        "state_reference_variance",
        "state_reference_validity",
        "radio_target",
        "radio_target_valid",
        "vision_capture_target_xy",
        "vision_target_mask",
        "vision_event_uid",
        "vision_capture_s",
    ):
        changed[key].fill_(999)
    relabelled = model(changed, forecast_s=0.3, batched_encoding=True, return_visual=True)
    for key in original:
        torch.testing.assert_close(original[key], relabelled[key], rtol=0, atol=0)
    cutoff = 6
    changed["image"][:, cutoff:] = torch.randn_like(changed["image"][:, cutoff:])
    changed["radio"][:, cutoff:] = torch.randn_like(changed["radio"][:, cutoff:])
    future_changed = model(changed, forecast_s=0.3, batched_encoding=True, return_visual=True)
    for key in original:
        torch.testing.assert_close(original[key][:, :cutoff], future_changed[key][:, :cutoff], rtol=0, atol=0)
    differentiable_batch = {key: value.clone() for key, value in batch.items()}
    differentiable_batch["image"].requires_grad_()
    prefix = model(differentiable_batch, forecast_s=0.3, batched_encoding=True)["mean"][:, :cutoff]
    image_gradient = torch.autograd.grad(prefix.sum(), differentiable_batch["image"])[0]
    assert torch.count_nonzero(image_gradient[:, cutoff:]) == 0
