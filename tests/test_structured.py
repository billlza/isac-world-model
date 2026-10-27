"""Causal, probabilistic, compatibility and train-from-scratch contracts."""

import copy

import numpy as np
import pytest
import torch

from isac_wm.artifacts import save_checkpoint
from isac_wm.evaluation import rollout
from isac_wm.model import ModelConfig, training_loss
from isac_wm.runtime import RuntimeConfig
from isac_wm.simulator import SceneConfig, recorded_episode
from isac_wm.structured import StructuredConfig, build_model, model_config_from_dict
from isac_wm.training import INPUT_KEYS, batchify, load_model, seed_all, train


@pytest.fixture
def batch():
    torch.set_num_threads(2)
    seed_all(3)
    return batchify([recorded_episode(i, SceneConfig(steps=12)) for i in (1, 2)], torch.device("cpu"))


@pytest.fixture(params=["structured", "structured-attention"])
def model(request):
    return build_model(StructuredConfig(16, 32, request.param, 4, 4))


def test_shapes_gradients_and_empty_memory(batch, model):
    output = model(batch)
    loss, _ = training_loss(output, batch)
    loss.backward()
    assert output["mean"].shape == (2, 12, 4)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    empty = {k: v.clone() for k, v in batch.items()}
    empty["new"].zero_()
    empty["valid"].zero_()
    output = model(empty)
    assert all(torch.isfinite(v).all() for v in output.values())


def test_future_and_label_isolation(batch, model):
    model.eval()
    altered = {k: v.clone() for k, v in batch.items()}
    for k in ("image", "radio"):
        altered[k][:, 6:] = torch.randn_like(altered[k][:, 6:])
    altered["new"][:, 6:] = 0
    a, b = model(batch), model(altered)
    torch.testing.assert_close(a["mean"][:, :6], b["mean"][:, :6], rtol=0, atol=0)
    for k in ("state_target", "radio_target", "radio_target_valid"):
        altered[k].fill_(99)
    altered["action_target"].fill_(2)
    c = model(altered)
    torch.testing.assert_close(b["mean"], c["mean"], rtol=0, atol=0)
    torch.testing.assert_close(b["future_mean"], c["future_mean"], rtol=0, atol=0)
    batch["image"].requires_grad_()
    gradient = torch.autograd.grad(model(batch)["mean"][:, :6].sum(), batch["image"])[0]
    assert torch.count_nonzero(gradient[:, 6:]) == 0


def test_bounded_memory_no_duplicate_and_covariance(batch, model):
    memory = model.initial(2, "cpu")
    for k in range(12):
        x = {key: batch[key][:, k] for key in INPUT_KEYS}
        output, memory = model.step(x, memory)
        assert memory[5].shape == (2, 4, 32)
        assert torch.linalg.eigvalsh(output["covariance"]).min() >= -1e-7
    old_memory = tuple(v.clone() for v in memory)
    x = {key: batch[key][:, -1].clone() for key in INPUT_KEYS}
    x["new"].zero_()
    with torch.no_grad():
        a, ma = model.step(x, memory)
        x["image"].fill_(float("nan"))
        x["radio"].fill_(float("nan"))
        b, mb = model.step(x, memory)
    torch.testing.assert_close(a["mean"], b["mean"], rtol=0, atol=0)
    for i in (5, 6, 7):
        torch.testing.assert_close(ma[i], old_memory[i], rtol=0, atol=0)
        torch.testing.assert_close(ma[i], mb[i], rtol=0, atol=0)
    for original, current in zip(old_memory, memory, strict=True):
        torch.testing.assert_close(original, current, rtol=0, atol=0)


def test_zero_horizon_and_batch_equivalence(batch, model):
    memory = model.initial(2, "cpu")
    output, _ = model.step({k: batch[k][:, 3] for k in INPUT_KEYS}, memory)
    torch.testing.assert_close(model.forecast(output["hidden"], 0)["mean"], output["mean"], rtol=0, atol=0)
    for bad in (-1, float("nan"), torch.tensor(-1), torch.ones(3)):
        with pytest.raises(ValueError):
            model.forecast(output["hidden"], bad)
    other = copy.deepcopy(model)
    a = model(batch, batched_encoding=False)
    b = other(batch, batched_encoding=True)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=2e-4, atol=2e-5)
    training_loss(a, batch)[0].backward()
    training_loss(b, batch)[0].backward()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=5e-3, atol=2e-4)


def test_resume_and_checkpoint(tmp_path, model):
    config = model.config
    scene = SceneConfig(steps=10)
    kwargs = dict(episodes=4, batch_size=2, device_name="cpu", scene=scene, model_config=config)
    train(tmp_path / "first", epochs=1, **kwargs)
    train(tmp_path / "resumed", epochs=2, resume=tmp_path / "first/resume.pt", **kwargs)
    train(tmp_path / "whole", epochs=2, **kwargs)
    one, _, _ = load_model(tmp_path / "resumed/model.pt", "cpu")
    two, _, _ = load_model(tmp_path / "whole/model.pt", "cpu")
    for key in one.state_dict():
        torch.testing.assert_close(one.state_dict()[key], two.state_dict()[key], rtol=0, atol=0)
    result = rollout(
        60_000_001, scene, "model", "fixed", one, torch.device("cpu"), runtime=RuntimeConfig("fixed", 150)
    )
    assert np.isfinite(result["metrics"]["position_rmse_m"])
    assert type(model_config_from_dict(ModelConfig().to_dict())) is ModelConfig
    payload = {
        "model_config": config.to_dict(),
        "scene_config": scene.to_dict(),
        "state_dict": model.state_dict(),
    }
    payload["scene_config"]["room_m"] = 20
    save_checkpoint(tmp_path / "bad.pt", payload)
    with pytest.raises(ValueError, match="geometry"):
        load_model(tmp_path / "bad.pt", "cpu")


@pytest.mark.parametrize(
    "field,value", [("memory_size", 0), ("heads", 5), ("room_m", 0), ("architecture", "unknown")]
)
def test_config_validation(field, value):
    with pytest.raises(ValueError):
        StructuredConfig(**{field: value})


def test_reject_invalid_fresh_data(batch, model):
    x = {k: batch[k][:, 3].clone() for k in INPUT_KEYS}
    x["new"].fill_(1)
    x["valid"].fill_(1)
    for key, value in (("dt", -1), ("age", float("nan")), ("valid", 0.5), ("image", float("nan"))):
        bad = {k: v.clone() for k, v in x.items()}
        bad[key].fill_(value)
        with pytest.raises(ValueError):
            model.step(bad, model.initial(2, "cpu"))
    with pytest.raises(ValueError):
        model.initial(0, "cpu")
