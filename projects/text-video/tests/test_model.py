"""Behavioral checks for the independently trained text-to-video flow model."""

from dataclasses import replace

import pytest
import torch

from text_video.model import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    ModelConfig,
    TextVideoModel,
    sample,
    tokenize_captions,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return ModelConfig(
        height=16, width=16, frames=4, patch_size=4, hidden=32,
        depth=2, heads=4, text_layers=1, max_text_bytes=32,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(9)
    return TextVideoModel(config)


def video_batch(config, batch=2):
    return torch.randn(batch, 3, config.frames, config.height, config.width)


def activate_output(model):
    # A zero output head intentionally has no conditioning effect before training.
    with torch.no_grad():
        model.output_projection.weight.normal_(std=0.02)


def test_utf8_tokenizer_and_null_condition():
    ids, mask = tokenize_captions(["猫", ""], max_text_bytes=3)
    assert ids[0].tolist() == [BOS_TOKEN, *"猫".encode("utf-8"), EOS_TOKEN]
    assert ids[1].tolist() == [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
    assert mask.dtype == torch.bool
    assert mask[0].tolist() == [False] * 5
    assert mask[1].tolist() == [False, False, True, True, True]


@pytest.mark.parametrize("captions,error", [
    ("cat", TypeError), ([], ValueError), ([3], TypeError), (["\ud800"], ValueError),
])
def test_invalid_captions_fail_explicitly(captions, error):
    with pytest.raises(error):
        tokenize_captions(captions, max_text_bytes=8)


def test_byte_limit_rejects_without_truncating():
    with pytest.raises(ValueError, match="6 UTF-8 bytes.*max_text_bytes=5"):
        tokenize_captions(["红球"], max_text_bytes=5)


@pytest.mark.parametrize("changes", [
    {"height": 15}, {"frames": 0}, {"hidden": 30}, {"depth": True},
    {"dropout": float("nan")}, {"dropout": 1.0}, {"max_text_bytes": 0},
])
def test_invalid_config(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_initial_velocity_shape_and_value(model, config):
    video = video_batch(config)
    output = model(video, torch.tensor([0.0, 1.0]), ["red ball", "红球"])
    assert output.shape == video.shape
    assert output.dtype == video.dtype
    assert torch.count_nonzero(output).item() == 0


def test_flow_loss_trains_head_and_then_backbone(model, config):
    noise = video_batch(config)
    data = video_batch(config).tanh()
    time = torch.tensor([0.2, 0.8])
    mix = time[:, None, None, None, None]
    inputs = noise * (1 - mix) + data * mix
    target = data - noise
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs, time, ["red ball", "蓝球"]) - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    for parameter in (
        model.output_projection.weight,
        model.patch_embed.weight,
        model.text_encoder.embedding.weight,
        model.blocks[0].spatial_attention.in_proj_weight,
        model.blocks[0].temporal_attention.in_proj_weight,
        model.time_embedding[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_text_changes_velocity(model, config):
    activate_output(model)
    model.eval()
    video = video_batch(config, batch=1)
    time = torch.tensor([0.4])
    red = model(video, time, ["红球"])
    blue = model(video, time, ["蓝球"])
    assert (red - blue).abs().max() > 1e-5


def test_padding_does_not_change_other_prompt(model, config):
    activate_output(model)
    model.eval()
    video = video_batch(config)
    time = torch.tensor([0.4, 0.7])
    alone = model(video[:1], time[:1], ["猫"])
    batched = model(video, time, ["猫", "a much longer caption"])
    torch.testing.assert_close(alone, batched[:1], atol=2e-6, rtol=1e-5)


def test_sampling_is_deterministic_and_restores_mode(model):
    activate_output(model)
    model.train()
    rng_state = torch.random.get_rng_state().clone()
    first = model.sample(["red ball", "蓝球"], steps=2, seed=31)
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    second = sample(model, ["red ball", "蓝球"], steps=2, seed=31)
    other = sample(model, ["red ball", "蓝球"], steps=2, seed=32)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == (2, 3, 4, 16, 16)
    assert torch.isfinite(first).all()
    assert first.min() >= -1 and first.max() <= 1
    assert not first.requires_grad


@pytest.mark.parametrize("method", ["heun", "euler"])
def test_sampler_integrates_noise_to_data_direction(model, config, method, monkeypatch):
    def constant_flow(video, time, context, mask):
        assert time.min() >= 0 and time.max() <= 1
        return torch.full_like(video, 0.25)

    monkeypatch.setattr(model, "_velocity", constant_flow)
    result = sample(model, [""], steps=2, guidance_scale=1, seed=14, method=method)
    expected = torch.randn(
        1, 3, config.frames, config.height, config.width,
        generator=torch.Generator().manual_seed(14),
    )
    torch.testing.assert_close(result, (expected + 0.25).clamp(-1, 1))


def test_nonfinite_sampling_fails_and_restores_mode(model, monkeypatch):
    def invalid_flow(video, time, context, mask):
        return torch.full_like(video, float("nan"))

    monkeypatch.setattr(model, "_velocity", invalid_flow)
    model.train()
    with pytest.raises(FloatingPointError, match="non-finite flow"):
        sample(model, ["a"], steps=1)
    assert model.training


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"steps": 1.5}, {"guidance_scale": float("inf")},
    {"guidance_scale": -1}, {"seed": 1.2}, {"method": "unknown"}, {"device": "cuda:0"},
])
def test_invalid_sampling_options(model, kwargs):
    with pytest.raises(ValueError):
        sample(model, ["a"], **kwargs)


def test_invalid_forward_inputs(model, config):
    video = video_batch(config)
    time = torch.tensor([0.2, 0.8])
    with pytest.raises(ValueError, match="shape"):
        model(video[:, :, :-1], time, ["a", "b"])
    with pytest.raises(ValueError, match="batch size"):
        model(video, time, ["a"])
    with pytest.raises(ValueError, match="shape"):
        model(video, time[:, None], ["a", "b"])
    with pytest.raises(ValueError, match="finite values in"):
        model(video, torch.tensor([0.2, 1.01]), ["a", "b"])
    with pytest.raises(ValueError, match="finite values in"):
        model(video, torch.tensor([0.2, float("nan")]), ["a", "b"])
    with pytest.raises(ValueError, match="not truncated"):
        model(video, time, ["a" * 33, "b"])
    bad_video = video.clone()
    bad_video[0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="only finite"):
        model(bad_video, time, ["a", "b"])
