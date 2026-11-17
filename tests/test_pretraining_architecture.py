"""Opt-in visual capacity and physical bounds preserve historical checkpoints."""

from types import MethodType
from unittest.mock import patch

import pytest
import torch
from torch import nn

from isac_wm.artifacts import save_checkpoint
from isac_wm.model import training_loss
from isac_wm.optimizer_policy import describe_optimizer_policy
from isac_wm.simulator import SceneConfig, recorded_episode
from isac_wm.structured import StructuredConfig, build_model, model_config_from_dict
from isac_wm.training import batchify, load_model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def scene():
    return SceneConfig(
        steps=10,
        image_size=16,
        camera_period=1,
        camera_delay_s=0,
        delay_jitter_s=0,
        camera_dropout=0,
        radio_dropout=0,
        occlusion_fraction=0,
    )


@pytest.fixture
def batch(scene):
    return batchify([recorded_episode(seed, scene) for seed in (234, 235)], torch.device("cpu"))


def small_config(**changes):
    return StructuredConfig(**({"feature_dim": 16, "hidden_dim": 32, "memory_size": 4} | changes))


def assert_exact(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


class LegacySpatialEncoder(nn.Module):
    """Literal pre-intervention constructor and calculation, for compatibility."""

    def __init__(self, features, channels=16, depth=1):
        super().__init__()
        assert (channels, depth) == (16, 1)
        self.trunk = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU())
        self.heat = nn.Conv2d(16, 1, 1)
        self.context = nn.Linear(16, features - 3)

    def forward(self, image):
        feature = self.trunk(image)
        scores = self.heat(feature).flatten(1)
        weights = (8 * scores).softmax(-1)
        height, width = image.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=image.device, dtype=image.dtype),
            torch.linspace(0, 1, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        xy = torch.stack([weights @ xx.flatten(), weights @ yy.flatten()], -1)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        entropy = entropy / scores.new_tensor(scores.shape[-1]).log()
        context = self.context((feature.flatten(2) * weights[:, None]).sum(-1))
        return torch.cat([xy, entropy, context], -1)


def legacy_predict(model, h, mean, covariance, dt, *, diagnostics=None):
    """Literal old physical transition; a changed default cannot pass by aliasing."""
    ratio = 2 / model.config.room_m
    acceleration_logits = model.acceleration(h)
    acc = 0.25 * acceleration_logits.tanh()
    position = mean[:, :2] + ratio * (dt * mean[:, 2:] + 0.5 * dt.square() * acc)
    velocity = mean[:, 2:] + dt * acc
    predicted = torch.cat([position, velocity], -1)
    transition = torch.eye(4, device=h.device, dtype=h.dtype).expand(h.shape[0], 4, 4).clone()
    transition[:, 0, 2] = transition[:, 1, 3] = dt[:, 0] * ratio
    noise_logits = model.noise(h)
    q = 1e-5 + 0.02 * noise_logits[:, :2].sigmoid()
    q = torch.cat([q * ratio**2, q], -1) * dt
    predicted_cov = transition @ covariance @ transition.transpose(-1, -2) + torch.diag_embed(q)
    if diagnostics is not None:
        diagnostics["acceleration_logits"] = acceleration_logits.detach().clone()
        diagnostics["noise_logits"] = noise_logits.detach().clone()
    return predicted, predicted_cov


@pytest.mark.parametrize("architecture", ["structured", "structured-attention"])
def test_defaults_preserve_legacy_weights_rng_and_forward(batch, architecture):
    config = small_config(architecture=architecture)
    torch.manual_seed(7)
    with patch("isac_wm.structured.SpatialEncoder", LegacySpatialEncoder):
        legacy = build_model(config).eval()
    legacy._predict = MethodType(legacy_predict, legacy)
    legacy_rng = torch.get_rng_state().clone()

    torch.manual_seed(7)
    current = build_model(config).eval()
    assert_exact(torch.get_rng_state(), legacy_rng)
    assert list(current.state_dict()) == list(legacy.state_dict())
    assert_exact(current.state_dict(), legacy.state_dict())
    current.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        assert_exact(
            current(batch, batched_encoding=True, return_visual=True),
            legacy(batch, batched_encoding=True, return_visual=True),
        )


def test_default_config_omission_and_each_opt_in_roundtrip():
    legacy = {
        "feature_dim": 48,
        "hidden_dim": 96,
        "architecture": "structured-attention",
        "memory_size": 16,
        "heads": 4,
        "room_m": 10.0,
    }
    assert StructuredConfig().to_dict() == legacy
    assert StructuredConfig(visual_channels=16, visual_depth=1, max_acceleration_mps2=0.5).to_dict() == legacy
    assert model_config_from_dict(legacy) == StructuredConfig()
    for field, value in (("visual_channels", 64), ("visual_depth", 3), ("max_acceleration_mps2", 6.0)):
        encoded = legacy | {field: value}
        assert StructuredConfig(**encoded).to_dict() == encoded
        assert model_config_from_dict(encoded).to_dict() == encoded
        with pytest.raises(ValueError, match="structured architectures"):
            model_config_from_dict({"feature_dim": 16, "hidden_dim": 32, field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("visual_channels", 0),
        ("visual_channels", 257),
        ("visual_channels", 1.5),
        ("visual_channels", True),
        ("visual_channels", float("inf")),
        ("visual_depth", 0),
        ("visual_depth", 9),
        ("visual_depth", 1.5),
        ("visual_depth", True),
        ("max_acceleration_mps2", 0),
        ("max_acceleration_mps2", -1),
        ("max_acceleration_mps2", 101),
        ("max_acceleration_mps2", float("nan")),
        ("max_acceleration_mps2", float("inf")),
        ("max_acceleration_mps2", True),
    ],
)
def test_invalid_capacity_or_physical_bound_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        StructuredConfig(**{field: value})


def test_registered_parameter_counts_and_optimizer_cover_expanded_encoder():
    legacy = build_model(StructuredConfig())
    expanded = build_model(StructuredConfig(visual_channels=64, visual_depth=3))
    assert sum(parameter.numel() for parameter in legacy.visual.parameters()) == 1_230
    assert sum(parameter.numel() for parameter in legacy.parameters()) == 172_246
    assert sum(parameter.numel() for parameter in expanded.visual.parameters()) == 78_638
    assert sum(parameter.numel() for parameter in expanded.parameters()) == 249_654
    policy = describe_optimizer_policy(expanded, 3e-4, width_lr_scale=0.5)
    fixed = next(group for group in policy["groups"] if group["name"] == "fixed_width")
    visual_names = {"visual." + name for name, _ in expanded.visual.named_parameters()}
    assert visual_names.issubset(fixed["parameter_names"])


@pytest.mark.parametrize("bound", [0.5, 6.0])
def test_acceleration_bound_has_physical_units_in_prediction_and_forecast(bound):
    model = build_model(small_config(max_acceleration_mps2=bound))
    with torch.no_grad():
        model.acceleration.weight.zero_()
        model.acceleration.bias.copy_(torch.tensor([20.0, -20.0]))
    h = torch.zeros(1, model.config.hidden_dim)
    mean = torch.tensor([[0.5, 0.5, 0.1, -0.2]])
    covariance = torch.eye(4)[None]
    dt = torch.tensor([[0.25]])
    predicted, predicted_cov = model._predict(h, mean, covariance, dt)
    acceleration_from_velocity = (predicted[:, 2:] - mean[:, 2:]) * 2 / dt
    acceleration_from_position = (
        (predicted[:, :2] - mean[:, :2]) * model.config.room_m - mean[:, 2:] * 2 * dt
    ) / (0.5 * dt.square())
    expected = torch.tensor([[bound, -bound]])
    torch.testing.assert_close(acceleration_from_velocity, expected)
    torch.testing.assert_close(acceleration_from_position, expected, atol=1e-5, rtol=1e-5)
    forecast = model.forecast(model._pack(h, mean, covariance), dt)
    assert_exact(forecast["mean"], predicted)
    assert_exact(forecast["covariance"], predicted_cov)
    assert_exact(model._predict(h, mean, covariance, torch.zeros_like(dt)), (mean, covariance))


def test_expanded_encoder_gradients_future_isolation_and_label_isolation(batch):
    torch.manual_seed(7)
    model = build_model(
        small_config(
            visual_channels=64, visual_depth=3, max_acceleration_mps2=6.0, latent_state_norm="layernorm"
        )
    )
    output = model(batch, batched_encoding=True, return_visual=True)
    loss, _ = training_loss(output, batch, vision_loss_weight=1.0)
    loss.backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    for layer in model.visual.trunk:
        if isinstance(layer, nn.Conv2d):
            assert bool(layer.weight.grad.abs().sum() > 0)
    assert bool(model.acceleration.weight.grad.abs().sum() > 0)

    altered = {name: value.clone() for name, value in batch.items()}
    altered["image"][:, 5:] = 1 - altered["image"][:, 5:]
    altered["radio"][:, 5:] = -altered["radio"][:, 5:]
    with torch.no_grad():
        changed = model(altered, batched_encoding=True)
        for name in ("mean", "future_mean", "logvar", "future_logvar"):
            assert_exact(output[name][:, :5], changed[name][:, :5])
        for name in ("state_target", "radio_target", "radio_target_valid", "vision_capture_target_xy"):
            altered[name].fill_(99)
        labels_changed = model(altered, batched_encoding=True)
        assert_exact(changed, labels_changed)

    differentiable = {name: value.clone() for name, value in batch.items()}
    differentiable["image"].requires_grad_()
    prefix = model(differentiable, batched_encoding=True)["mean"][:, :5].sum()
    gradient = torch.autograd.grad(prefix, differentiable["image"])[0]
    assert torch.count_nonzero(gradient[:, 5:]) == 0


@pytest.mark.parametrize("mode", ["legacy-missing", "explicit-defaults", "expanded"])
def test_checkpoint_roundtrip_for_old_defaults_and_expanded_config(tmp_path, batch, scene, mode):
    settings = {"visual_channels": 64, "visual_depth": 3, "max_acceleration_mps2": 6.0}
    model = build_model(small_config(**(settings if mode == "expanded" else {}))).eval()
    encoded = model.config.to_dict()
    if mode == "explicit-defaults":
        encoded |= {"visual_channels": 16, "visual_depth": 1, "max_acceleration_mps2": 0.5}
    if mode == "legacy-missing":
        assert set(settings).isdisjoint(encoded)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        {
            "schema_version": 2,
            "model_config": encoded,
            "scene_config": scene.to_dict(),
            "state_dict": model.state_dict(),
        },
    )
    restored, _, _ = load_model(path, "cpu")
    assert restored.config == model.config
    assert_exact(restored.state_dict(), model.state_dict())
    with torch.no_grad():
        assert_exact(restored(batch), model(batch))
