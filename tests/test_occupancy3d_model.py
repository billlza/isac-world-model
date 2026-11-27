import numpy as np
import pytest
import torch

from isac_wm.occupancy3d_model import (
    OccupancyReadout,
    StaticRFMemory,
    geometry_loss,
    prefix_features,
    rf_consistency_loss,
    weight_fingerprint,
)


def fixture_arrays():
    rng = np.random.default_rng(12)
    shape = (4, 4, 4)
    coords = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
    a = (rng.normal(size=(12, 64)) + 1j * rng.normal(size=(12, 64))) / 4
    y = rng.normal(size=(2, 12)) + 1j * rng.normal(size=(2, 12))
    return shape, coords, a, y


def test_future_measurements_cannot_change_prefix():
    shape, coords, a, y = fixture_arrays()
    before = prefix_features(y, a, coords, shape, 4, 1)
    changed = y.copy()
    changed[:, 4:] = np.nan
    np.testing.assert_array_equal(before, prefix_features(changed, a, coords, shape, 4, 1))


def test_streaming_matches_batch_and_accepts_late_static_evidence():
    shape, coords, a, y = fixture_arrays()
    memory = StaticRFMemory(a, coords, shape, 4, batch_size=2)
    memory.update(y[:, 4:8], 1, capture_s=1, available_s=1.1, now_s=2)
    memory.update(y[:, :4], 0, capture_s=0, available_s=2.1, now_s=2.1)
    np.testing.assert_allclose(memory.features(), prefix_features(y, a, coords, shape, 4, 2), atol=1e-6)
    with pytest.raises(ValueError, match="Duplicate"):
        memory.update(y[:, :4], 0, capture_s=0, available_s=2.1, now_s=3)


def test_future_arrival_rejected_before_mutation():
    shape, coords, a, y = fixture_arrays()
    memory = StaticRFMemory(a, coords, shape, 4, batch_size=2)
    with pytest.raises(ValueError, match="contract"):
        memory.update(y[:, :4], 0, capture_s=0, available_s=2, now_s=1)
    assert not memory.accepted and not np.any(memory.numerator)


def test_physical_loss_zero_at_exact_solution_and_gradients_present():
    shape, _, a, _ = fixture_arrays()
    truth = torch.rand(2, *shape, requires_grad=True)
    operator = torch.as_tensor(a, dtype=torch.complex64)
    exact_y = truth.detach().flatten(1).to(operator.dtype) @ operator.T
    loss = rf_consistency_loss(truth, exact_y, operator)
    assert float(loss.detach()) < 1e-10
    wrong = rf_consistency_loss(truth * 0.7, exact_y, operator)
    wrong.backward()
    assert truth.grad is not None and bool(torch.isfinite(truth.grad).all())
    assert float(truth.grad.abs().sum()) > 0


def test_random_initialization_and_geometry_training_path():
    torch.manual_seed(7)
    a = OccupancyReadout(channels=4)
    torch.manual_seed(7)
    b = OccupancyReadout(channels=4)
    torch.manual_seed(11)
    c = OccupancyReadout(channels=4)
    assert weight_fingerprint(a) == weight_fingerprint(b)
    assert weight_fingerprint(a) != weight_fingerprint(c)
    logits = a(torch.randn(2, 7, 4, 4, 4))
    geometry_loss(logits, torch.zeros_like(logits)).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in a.parameters())
