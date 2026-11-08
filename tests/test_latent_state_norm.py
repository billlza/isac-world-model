"""Recurrent-state normalization is an explicit, parameter-free intervention."""

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from isac_wm.artifacts import model_fingerprint, read_checkpoint, save_checkpoint
from isac_wm.model import EventBeliefModel, training_loss
from isac_wm.simulator import SceneConfig, recorded_episode
from isac_wm.structured import (
    LATENT_STATE_NORM_EPS,
    StructuredConfig,
    build_model,
    model_config_from_dict,
)
from isac_wm.training import batchify, load_model, train
from isac_wm.visual_training_diagnostics import validation_snapshot


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(mode="none"):
    return StructuredConfig(feature_dim=8, hidden_dim=16, memory_size=4, latent_state_norm=mode)


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(13)
    return {
        "image": torch.rand(4, 3, 16, 16, generator=generator),
        "radio": torch.tensor([[0.4, 0.8, 0.6, 0.1]]).repeat(4, 1),
        "new": torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=torch.float32),
        "valid": torch.ones(4, 2),
        "age": torch.zeros(4, 2),
        "dt": torch.full((4, 1), 0.1),
    }


@pytest.fixture
def scene():
    return SceneConfig(steps=8, image_size=16, camera_period=1, camera_dropout=0, occlusion_fraction=0)


@pytest.fixture
def batch(scene):
    return batchify([recorded_episode(234, scene)], torch.device("cpu"))


def assert_exact(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_config_serialization_preserves_the_legacy_representation():
    legacy = {
        "feature_dim": 48,
        "hidden_dim": 96,
        "architecture": "structured-attention",
        "memory_size": 16,
        "heads": 4,
        "room_m": 10.0,
    }
    assert StructuredConfig().to_dict() == legacy
    assert StructuredConfig(latent_state_norm="none").to_dict() == legacy
    assert model_config_from_dict(legacy).latent_state_norm == "none"
    active = legacy | {"latent_state_norm": "layernorm"}
    assert model_config_from_dict(active).to_dict() == active


@pytest.mark.parametrize("value", [None, True, 1, "", "batchnorm", "LayerNorm", []])
def test_invalid_normalization_is_rejected(value):
    with pytest.raises(ValueError, match="latent_state_norm"):
        StructuredConfig(latent_state_norm=value)


def test_unsupported_architectures_do_not_silently_ignore_normalization():
    with pytest.raises(ValueError, match="structured-attention"):
        StructuredConfig(architecture="structured", latent_state_norm="layernorm")
    with pytest.raises(ValueError, match="structured architectures"):
        model_config_from_dict({"feature_dim": 8, "hidden_dim": 16, "latent_state_norm": "layernorm"})
    with pytest.raises(ValueError, match="StructuredEventModel"):
        EventBeliefModel(config("layernorm"))


@pytest.mark.parametrize("hidden_dim", [16, 512])
def test_no_new_parameters_buffers_initialization_or_rng_draws(hidden_dim):
    torch.manual_seed(7)
    legacy = build_model(StructuredConfig(hidden_dim=hidden_dim))
    legacy_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    normalized = build_model(StructuredConfig(hidden_dim=hidden_dim, latent_state_norm="layernorm"))
    assert_exact(torch.get_rng_state(), legacy_rng)
    assert list(legacy.state_dict()) == list(normalized.state_dict())
    assert_exact(legacy.state_dict(), normalized.state_dict())
    assert model_fingerprint(legacy) == model_fingerprint(normalized)
    assert list(dict(legacy.named_parameters())) == list(dict(normalized.named_parameters()))
    assert legacy.latent_state_norm is None
    layer = normalized.latent_state_norm
    assert isinstance(layer, nn.LayerNorm)
    assert layer.normalized_shape == (hidden_dim,)
    assert layer.eps == LATENT_STATE_NORM_EPS == 1e-5
    assert layer.elementwise_affine is False
    assert layer.weight is None and layer.bias is None
    assert list(layer.parameters()) == []
    assert list(layer.buffers()) == []
    if hidden_dim == 512:
        assert sum(parameter.numel() for parameter in normalized.parameters()) == 3_883_798


def legacy_transition(model, inputs, memory):
    """Direct pre-intervention transition for valid data, without any norm boundary.

    Keep the branch selection and physical equations explicit: comparing only
    two configurations that both choose the new default would miss a changed
    default transition. This reference intentionally omits diagnostic plumbing.
    """
    h, pv, pr, mean, covariance, tokens, stamps, used, clock = memory
    new = (inputs["new"] * inputs["valid"]).bool()
    age, dt = inputs["age"], inputs["dt"]
    clock = clock + dt
    encodings = model.encode_new(inputs)
    pv = torch.where(new[:, :1], encodings[0], pv)
    pr = torch.where(new[:, 1:], encodings[1], pr)
    prior_h = h + dt * model.dynamics(h)
    features = torch.cat([pv, pr, age, inputs["valid"], new.to(h), dt], -1)
    updated_h = model.fusion(features, prior_h)
    if model.attentive:
        updated_h, tokens, stamps, used = model._retrieve(
            updated_h, encodings, new, age, clock, tokens, stamps, used
        )
    h = torch.where(new.any(-1, keepdim=True), updated_h, prior_h)
    mean, covariance = model._predict(h, mean, covariance, dt)
    variance = (-10 + 7 * model.noise(h)[:, 2:].sigmoid()).exp()
    visual_variance = variance[:, :2] + 0.05 * pv[:, 2:3].clamp(0, 1).pow(4)
    safe_radio = torch.where(new[:, 1:], inputs["radio"], torch.zeros_like(inputs["radio"]))
    radio_xy = (2**0.5) * safe_radio[:, :1] * safe_radio[:, 1:3]
    for measurement, r, a, active in (
        (pv[:, :2], visual_variance, age[:, :1], new[:, :1]),
        (radio_xy, variance[:, 2:], age[:, 1:], new[:, 1:]),
    ):
        mean, covariance = model._correct(
            mean, covariance, measurement, r, a, active, 2 / model.config.room_m
        )
    return h, pv, pr, mean, covariance, tokens, stamps, used, clock


@pytest.mark.parametrize("architecture", ["structured", "structured-attention"])
def test_none_matches_direct_legacy_transition_exactly(inputs, architecture):
    torch.manual_seed(7)
    model = build_model(StructuredConfig(8, 16, architecture, 4))
    memory = model.initial(4, "cpu")
    reference = tuple(value.clone() for value in memory)
    for _ in range(4):
        reference = legacy_transition(model, inputs, reference)
        observed = {}
        output, memory = model.step(inputs, memory, diagnostics=observed)
        assert_exact(memory, reference)
        assert_exact(output["hidden"], model._pack(reference[0], reference[3], reference[4]))
        assert_exact(observed["network"]["hidden_pre_norm"], observed["network"]["hidden"])
        assert_exact(model.forecast(output["hidden"], 0)["mean"], output["mean"])


def test_boundary_follows_branch_selection_and_precedes_all_readouts(inputs, monkeypatch):
    torch.manual_seed(7)
    model = build_model(config("layernorm"))
    memory = list(model.initial(4, "cpu"))
    memory[0] = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 3 + 10
    memory = tuple(memory)
    original = tuple(value.clone() for value in memory)
    retrieval, normalized, readouts = [], [], []
    retrieve = model._retrieve

    def capture_retrieval(*args):
        result = retrieve(*args)
        retrieval.append(result[0].detach().clone())
        return result

    monkeypatch.setattr(model, "_retrieve", capture_retrieval)
    handles = [
        model.latent_state_norm.register_forward_hook(
            lambda module, args, result: normalized.append(
                (args[0].detach().clone(), result.detach().clone())
            )
        ),
        model.acceleration.register_forward_pre_hook(lambda module, args: readouts.append(args[0].clone())),
        model.noise.register_forward_pre_hook(lambda module, args: readouts.append(args[0].clone())),
    ]
    try:
        observed = {}
        output, next_memory = model.step(inputs, memory, diagnostics=observed)
        assert len(normalized) == 1
        before, after = normalized[0]
        active = (inputs["new"] * inputs["valid"]).bool().any(-1)
        assert_exact(before[active], retrieval[0][active])
        expected_prior = memory[0] + inputs["dt"] * model.dynamics(memory[0])
        assert_exact(before[~active], expected_prior[~active])
        manual = (before - before.mean(-1, keepdim=True)) / (
            before.var(-1, correction=0, keepdim=True) + LATENT_STATE_NORM_EPS
        ).sqrt()
        torch.testing.assert_close(after, manual, rtol=2e-5, atol=2e-5)
        assert after.shape == memory[0].shape
        assert bool((after.square().mean(-1).sqrt() <= 1.00001).all())
        assert len(readouts) == 3  # Acceleration, process noise, measurement noise.
        for value in readouts:
            assert_exact(value, after)
        assert_exact(next_memory[0], after)
        assert_exact(output["hidden"][:, : model.config.hidden_dim], after)
        assert_exact(observed["network"]["hidden_pre_norm"], before)
        assert_exact(observed["network"]["hidden"], after)
        assert_exact(memory, original)
        # Forecast does not independently evolve h: reuse the same normalized h.
        model.forecast(output["hidden"], 0.3)
        assert len(normalized) == 1
        assert_exact(readouts[-1], after)
        assert_exact(readouts[-2], after)
    finally:
        for handle in handles:
            handle.remove()


def test_only_latent_state_is_normalized(inputs):
    model = build_model(config("layernorm"))
    memory = list(model.initial(4, "cpu"))
    memory[0] = torch.arange(64, dtype=torch.float32).reshape(4, 16) + 20
    memory[3] = torch.tensor([[0.8, 0.2, -0.3, 0.6]]).repeat(4, 1)
    memory[4] = torch.diag(torch.tensor([0.2, 0.5, 0.3, 0.1])).repeat(4, 1, 1)
    inputs = inputs | {"new": torch.zeros(4, 2), "dt": torch.zeros(4, 1)}
    output, next_memory = model.step(inputs, tuple(memory))
    assert not torch.equal(next_memory[0], memory[0])
    assert_exact(output["mean"], memory[3])
    assert_exact(output["covariance"], memory[4])
    for index in (1, 2, 3, 4, 5, 6, 7, 8):
        assert_exact(next_memory[index], memory[index])


def test_normalized_model_is_differentiable_and_causal(batch):
    torch.manual_seed(7)
    model = build_model(config("layernorm"))
    output = model(batch, return_visual=True)
    loss, _ = training_loss(output, batch, vision_loss_weight=1.0)
    loss.backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert bool(model.fusion.weight_hh.grad.abs().sum() > 0)
    assert bool(model.attention.in_proj_weight.grad.abs().sum() > 0)
    assert bool(model.visual.heat.weight.grad.abs().sum() > 0)
    altered = {name: value.clone() for name, value in batch.items()}
    altered["image"][:, 6:] = 1 - altered["image"][:, 6:]
    altered["radio"][:, 6:] = -altered["radio"][:, 6:]
    changed = model(altered)
    assert_exact(output["mean"][:, :6], changed["mean"][:, :6])
    for name in ("state_target", "radio_target", "radio_target_valid", "vision_capture_target_xy"):
        altered[name].fill_(99)
    labels_changed = model(altered)
    assert_exact(changed["mean"], labels_changed["mean"])
    assert_exact(changed["future_mean"], labels_changed["future_mean"])


def test_observations_are_detached_passive_copies(inputs, batch):
    model = build_model(config("layernorm"))
    memory = model.initial(4, "cpu")
    original, original_memory = model.step(inputs, memory)
    observed = {}
    actual, actual_memory = model.step(inputs, memory, diagnostics=observed)
    assert_exact(original, actual)
    assert_exact(original_memory, actual_memory)
    for key in ("hidden", "hidden_pre_norm"):
        assert observed["network"][key].requires_grad is False
        observed["network"][key].fill_(999)
    assert_exact(original, actual)
    assert_exact(original_memory, actual_memory)

    model.train()
    model.visual.eval()
    modes = [module.training for module in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    parameters = model_fingerprint(model)
    input_copies = {name: value.clone() for name, value in batch.items()}
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    report = validation_snapshot(model, batch, room_m=10.0, dt=0.1, vision_loss_weight=1.0)
    assert parameters == model_fingerprint(model)
    assert modes == [module.training for module in model.modules()]
    assert python_rng == random.getstate()
    for left, right in zip(numpy_rng, np.random.get_state(), strict=True):
        np.testing.assert_equal(left, right)
    assert_exact(torch_rng, torch.get_rng_state())
    assert_exact(batch, input_copies)
    for parameter in model.parameters():
        assert_exact(parameter.grad, torch.full_like(parameter, 0.25))
    network = report["network"]
    assert network["hidden_rms"] == network["hidden_post_norm_rms"]
    assert 0 < network["hidden_post_norm_rms"] <= 1.00001
    assert network["hidden_pre_norm_rms"] > 0
    assert "post-normalization" in network["hidden_scope"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mode", ["legacy-missing", "none", "layernorm"])
def test_old_and_explicit_checkpoint_configs_load_exactly(tmp_path, batch, scene, mode):
    model = build_model(config("layernorm" if mode == "layernorm" else "none")).eval()
    saved_config = model.config.to_dict()
    if mode == "none":
        saved_config["latent_state_norm"] = "none"
    if mode == "legacy-missing":
        assert "latent_state_norm" not in saved_config
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        {
            "schema_version": 2,
            "model_config": saved_config,
            "scene_config": scene.to_dict(),
            "state_dict": model.state_dict(),
        },
    )
    restored, _, _ = load_model(path, "cpu")
    assert restored.config.latent_state_norm == ("layernorm" if mode == "layernorm" else "none")
    assert_exact(model.state_dict(), restored.state_dict())
    assert_exact(model(batch), restored(batch))


def test_active_resume_matches_uninterrupted_and_rejects_mode_change(tmp_path, scene):
    common = dict(
        episodes=2,
        batch_size=1,
        device_name="cpu",
        seed=7,
        scene=scene,
        model_config=config("layernorm"),
        vision_loss_weight=1.0,
        train_seed_base=40000,
        validation_seed_base=50000,
        batch_order_seed=7,
        validation_episodes=2,
        validation_every_updates=2,
        diagnostic_updates=(0, 1, 2, 4),
    )
    train(tmp_path / "whole", epochs=2, **common)
    train(tmp_path / "first", epochs=1, **common)
    checkpoint = tmp_path / "first/resume.pt"
    train(tmp_path / "resumed", epochs=2, resume=checkpoint, **common)
    whole = read_checkpoint(tmp_path / "whole/resume.pt")
    resumed = read_checkpoint(tmp_path / "resumed/resume.pt")
    assert_exact(whole["state_dict"], resumed["state_dict"])
    assert_exact(whole["optimizer"], resumed["optimizer"])
    assert whole["history"] == resumed["history"]
    assert whole["model_config"]["latent_state_norm"] == "layernorm"
    with pytest.raises(ValueError, match="Resume configuration"):
        train(tmp_path / "changed", epochs=2, resume=checkpoint, **(common | {"model_config": config()}))


def test_explicit_none_resume_is_equivalent_to_legacy_default(tmp_path, scene):
    common = dict(
        episodes=2,
        batch_size=1,
        device_name="cpu",
        seed=7,
        scene=scene,
        model_config=config(),
        train_seed_base=40000,
        validation_seed_base=50000,
        batch_order_seed=7,
        validation_episodes=2,
        validation_every_updates=2,
    )
    train(tmp_path / "whole", epochs=2, **common)
    train(tmp_path / "first", epochs=1, **common)
    explicit = read_checkpoint(tmp_path / "first/resume.pt")
    assert "latent_state_norm" not in explicit["model_config"]
    explicit["model_config"]["latent_state_norm"] = "none"
    checkpoint = tmp_path / "explicit-none.pt"
    save_checkpoint(checkpoint, explicit)
    train(tmp_path / "resumed", epochs=2, resume=checkpoint, **common)
    whole = read_checkpoint(tmp_path / "whole/resume.pt")
    resumed = read_checkpoint(tmp_path / "resumed/resume.pt")
    assert_exact(whole["state_dict"], resumed["state_dict"])
    assert_exact(whole["optimizer"], resumed["optimizer"])
    assert whole["history"] == resumed["history"]
    assert "latent_state_norm" not in resumed["model_config"]
    for index, invalid in enumerate(({"latent_state_norm": "unknown"}, {"unknown_field": 1})):
        changed = explicit | {"model_config": explicit["model_config"] | invalid}
        invalid_checkpoint = tmp_path / f"invalid-{index}.pt"
        save_checkpoint(invalid_checkpoint, changed)
        with pytest.raises(ValueError, match="Resume model configuration is invalid"):
            train(tmp_path / f"invalid-{index}", epochs=2, resume=invalid_checkpoint, **common)
