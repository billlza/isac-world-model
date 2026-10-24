import numpy as np
import pytest
import torch

from isac_wm.model import EventBeliefModel, ModelConfig, radio_from_state, training_loss
from isac_wm.simulator import SceneConfig, ideal_radio, recorded_episode
from isac_wm.training import batchify, load_model, resolve_device, seed_all, train


def setup_batch():
    torch.set_num_threads(2)
    seed_all(3)
    c = SceneConfig(steps=10)
    return batchify([recorded_episode(i, c) for i in (1, 2)], torch.device("cpu"))


def test_forward_shapes_and_finite_gradients():
    b = setup_batch()
    m = EventBeliefModel(ModelConfig(16, 32))
    o = m(b)
    assert o["mean"].shape == (2, 10, 4)
    loss, _ = training_loss(o, b)
    loss.backward()
    for name in ("visual", "radio_encoder", "dynamics", "fusion", "observation_head"):
        grads = [p.grad for p in getattr(m, name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


def test_temporal_causality_no_future_inputs():
    b = setup_batch()
    m = EventBeliefModel(ModelConfig(16, 32)).eval()
    altered = {k: v.clone() for k, v in b.items()}
    for k in ("image", "radio", "age", "valid", "new"):
        altered[k][:, 6:] = torch.rand_like(altered[k][:, 6:])
    with torch.no_grad():
        a = m(b)["mean"]
        z = m(altered)["mean"]
    torch.testing.assert_close(a[:, :6], z[:, :6], rtol=0, atol=0)


def test_training_labels_never_enter_state_estimator():
    b = setup_batch()
    m = EventBeliefModel(ModelConfig(16, 32)).eval()
    altered = {k: v.clone() for k, v in b.items()}
    for k in ("state_target", "radio_target", "radio_target_valid"):
        altered[k].fill_(999)
    altered["action_target"].fill_(2)
    with torch.no_grad():
        a = m(b)
        z = m(altered)
    torch.testing.assert_close(a["mean"], z["mean"], rtol=0, atol=0)
    torch.testing.assert_close(a["future_mean"], z["future_mean"], rtol=0, atol=0)


def test_reused_payload_is_not_new_evidence():
    b = setup_batch()
    m = EventBeliefModel(ModelConfig(16, 32)).eval()
    x = {k: b[k][:, 3].clone() for k in ("image", "radio", "age", "valid", "new", "dt")}
    x["new"].zero_()
    altered = {k: v.clone() for k, v in x.items()}
    altered["image"].fill_(30)
    altered["radio"].fill_(50)
    mem = m.initial(2, "cpu")
    with torch.no_grad():
        a, _ = m.step(x, mem)
        z, _ = m.step(altered, mem)
    torch.testing.assert_close(a["mean"], z["mean"], rtol=0, atol=0)


def test_zero_horizon_equals_current_decode():
    m = EventBeliefModel()
    h = torch.randn(3, 96)
    torch.testing.assert_close(m.decode_state(h)["mean"], m.forecast(h, 0)["mean"])
    with pytest.raises(ValueError):
        m.forecast(h, -1)


def test_geometry_head_matches_simulator():
    state = np.array([3, 4, 0.6, -0.2], np.float32)
    x = torch.tensor(state / np.array([10, 10, 2, 2], np.float32))[None]
    np.testing.assert_allclose(radio_from_state(x)[0].numpy(), ideal_radio(state, 10), rtol=1e-5)


def test_seeded_random_initialization():
    seed_all(1)
    a = EventBeliefModel()
    seed_all(1)
    b = EventBeliefModel()
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k])


def test_small_batch_can_learn():
    b = setup_batch()
    m = EventBeliefModel(ModelConfig(16, 32))
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    before = float(training_loss(m(b), b)[0].detach())
    for _ in range(15):
        opt.zero_grad()
        loss, _ = training_loss(m(b), b)
        loss.backward()
        opt.step()
    after = float(training_loss(m(b), b)[0].detach())
    assert after < before


def test_checkpoint_and_disjoint_split(tmp_path):
    out = tmp_path / "training"
    result = train(
        out,
        epochs=1,
        episodes=4,
        batch_size=2,
        device_name="cpu",
        scene=SceneConfig(steps=10),
        model_config=ModelConfig(16, 32),
    )
    assert not set(result["train_episode_seeds"]) & set(result["validation_episode_seeds"])
    m, c, d = load_model(out / "model.pt", "cpu")
    assert c.steps == 10 and d.type == "cpu"
    assert result["initialization"].startswith("random")
    with pytest.raises(FileExistsError):
        train(out, epochs=1)


def test_invalid_device():
    with pytest.raises(ValueError):
        resolve_device("nonsense")
