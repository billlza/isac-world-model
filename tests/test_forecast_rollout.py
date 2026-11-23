"""Forecast rollout is an explicit, causal intervention with a legacy control."""

from dataclasses import replace

import pytest
import torch

from isac_wm.structured import (
    MAX_FORECAST_STEPS,
    StructuredConfig,
    build_model,
    model_config_from_dict,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(step=0.1, norm="none", architecture="structured-attention"):
    torch.manual_seed(19)
    return build_model(
        StructuredConfig(
            feature_dim=8,
            hidden_dim=16,
            memory_size=4,
            architecture=architecture,
            latent_state_norm=norm,
            forecast_step_s=step,
        )
    )


def prepared_belief(model, batch_size=3):
    """Obtain a nontrivial belief through the real observation-update API."""
    inputs = {
        "image": torch.zeros(batch_size, 3, 16, 16),
        "radio": torch.tensor([[0.4, 0.8, 0.6, 0.1]]).repeat(batch_size, 1),
        "new": torch.tensor([[0.0, 1.0]]).repeat(batch_size, 1),
        "valid": torch.tensor([[0.0, 1.0]]).repeat(batch_size, 1),
        "age": torch.zeros(batch_size, 2),
        "dt": torch.full((batch_size, 1), 0.1),
    }
    memory = model.initial(batch_size, "cpu")
    for _ in range(3):
        output, memory = model.step(inputs, memory)
    return output["hidden"], memory, inputs


def test_config_legacy_roundtrip_and_random_initialization_are_unchanged():
    legacy = StructuredConfig().to_dict()
    assert "forecast_step_s" not in legacy
    assert StructuredConfig(forecast_step_s=0).to_dict() == legacy
    assert model_config_from_dict(legacy).forecast_step_s == 0
    active = legacy | {"forecast_step_s": 0.1}
    assert model_config_from_dict(active).to_dict() == active
    old, new = make_model(0), make_model(0.1)
    assert list(old.state_dict()) == list(new.state_dict())
    for key in old.state_dict():
        torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    with pytest.raises(ValueError, match="structured architectures"):
        model_config_from_dict({"feature_dim": 8, "hidden_dim": 16, "forecast_step_s": 0.1})


@pytest.mark.parametrize("norm", ["none", "layernorm"])
def test_default_forecast_matches_pre_intervention_equations_exactly(norm):
    model = make_model(0, norm)
    packed, _, _ = prepared_belief(model)
    h, mean, covariance = model._unpack(packed)
    # A direct historical physical extrapolation catches an accidental change to
    # the default, even if both omitted and explicit-zero config take that change.
    for horizon in (0, 0.3, 2.0):
        predicted, predicted_cov = model._predict(h, mean, covariance, h.new_full((3, 1), horizon))
        result = model.forecast(packed, horizon)
        torch.testing.assert_close(result["mean"], predicted, rtol=0, atol=0)
        torch.testing.assert_close(result["covariance"], predicted_cov, rtol=0, atol=0)


@pytest.mark.parametrize(
    "architecture,norm",
    [("structured", "none"), ("structured-attention", "none"), ("structured-attention", "layernorm")],
)
def test_forecast_matches_real_observation_free_steps_without_mutating_memory(architecture, norm):
    model = make_model(0.1, norm, architecture)
    packed, memory, inputs = prepared_belief(model)
    saved_packed = packed.detach().clone()
    saved_memory = tuple(value.detach().clone() for value in memory)
    observed = model.forecast(packed, 0.25)
    # NaN cached payloads have no evidential status and must never enter a
    # prediction. The real streaming step supplies an independent integration path.
    missing = {key: value.clone() for key, value in inputs.items()}
    missing["new"].zero_()
    missing["valid"].zero_()
    missing["radio"].fill_(float("nan"))
    missing["image"].fill_(float("nan"))
    evolved = memory
    for interval in (0.1, 0.1, 0.05):
        missing["dt"].fill_(interval)
        missing["age"] += interval
        expected, evolved = model.step(missing, evolved)
    for key in ("mean", "covariance", "logvar"):
        torch.testing.assert_close(observed[key], expected[key], rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(packed, saved_packed, rtol=0, atol=0)
    for current, before in zip(memory, saved_memory, strict=True):
        torch.testing.assert_close(current, before, rtol=0, atol=0)
    for index in (5, 6, 7):
        torch.testing.assert_close(evolved[index], memory[index], rtol=0, atol=0)
    assert torch.linalg.eigvalsh(observed["covariance"]).min() > 0


@pytest.mark.parametrize("norm", ["none", "layernorm"])
def test_each_batch_row_stops_at_its_own_horizon_and_zero_is_exact(norm):
    model = make_model(0.1, norm)
    packed, _, _ = prepared_belief(model)
    horizons = torch.tensor([[0.0], [0.25], [0.3]])
    result = model.forecast(packed, horizons)
    for index, horizon in enumerate(horizons):
        single = model.forecast(packed[index : index + 1], horizon.reshape(()))
        for key in result:
            torch.testing.assert_close(result[key][index : index + 1], single[key], rtol=2e-6, atol=1e-7)
    initial = model.decode_state(packed[:1])
    for key in result:
        torch.testing.assert_close(result[key][:1], initial[key], rtol=0, atol=0)
    zero = model.forecast(packed, 0)
    for key, original in model.decode_state(packed).items():
        torch.testing.assert_close(zero[key], original, rtol=0, atol=0)


@pytest.mark.parametrize("norm", ["none", "layernorm"])
def test_rollout_backpropagates_through_dynamics_and_input_horizon(norm):
    model = make_model(0.1, norm)
    packed, _, _ = prepared_belief(model, 1)
    packed = packed.detach().requires_grad_()
    horizon = torch.tensor(0.25, requires_grad=True)
    result = model.forecast(packed, horizon)
    loss = result["mean"].square().sum() + result["covariance"].square().sum()
    loss.backward()
    assert torch.isfinite(packed.grad).all()
    assert packed.grad[:, : model.config.hidden_dim].abs().sum() > 0
    assert torch.isfinite(horizon.grad) and horizon.grad.abs() > 0
    for module in (model.dynamics, model.acceleration, model.noise):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


def test_future_prediction_responds_to_dynamics_only_in_the_opted_in_model():
    model = make_model(0.1)
    packed, _, _ = prepared_belief(model, 1)
    packed = packed.detach()
    active_before = model.forecast(packed, 0.6)["mean"]
    model.config = replace(model.config, forecast_step_s=0)
    legacy_before = model.forecast(packed, 0.6)["mean"]
    with torch.no_grad():
        model.dynamics[2].bias.add_(2)
    torch.testing.assert_close(model.forecast(packed, 0.6)["mean"], legacy_before, rtol=0, atol=0)
    model.config = replace(model.config, forecast_step_s=0.1)
    assert not torch.allclose(model.forecast(packed, 0.6)["mean"], active_before)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, None, "0.1"])
def test_invalid_rollout_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="forecast_step_s"):
        StructuredConfig(forecast_step_s=value)


@pytest.mark.parametrize("step", [0, 0.1])
@pytest.mark.parametrize(
    "horizon",
    [
        -0.1,
        float("nan"),
        float("inf"),
        True,
        None,
        "0.1",
        1e100,
        torch.tensor(-0.1),
        torch.tensor(float("nan")),
        torch.tensor(1e100, dtype=torch.float64),
        torch.tensor(True),
        torch.tensor(0.1 + 0j),
        torch.ones(3),
        torch.ones(4, 1),
    ],
)
def test_invalid_horizon_is_rejected_explicitly(step, horizon):
    model = make_model(step)
    packed, _, _ = prepared_belief(model)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(packed, horizon)


def test_rollout_budget_is_explicit_and_does_not_silently_fall_back():
    model = make_model(0.25)
    packed, _, _ = prepared_belief(model, 1)
    with pytest.raises(ValueError, match=f"more than {MAX_FORECAST_STEPS}"):
        model.forecast(packed, (MAX_FORECAST_STEPS + 1) * 0.25)
    with torch.no_grad():
        result = model.forecast(packed, MAX_FORECAST_STEPS * 0.25)
    assert all(torch.isfinite(value).all() for value in result.values())
    # The control is an analytic extrapolation and has no internal-step budget.
    model.config = replace(model.config, forecast_step_s=0)
    assert torch.isfinite(model.forecast(packed, 100)["mean"]).all()
