import numpy as np
import pytest

from isac_wm.baselines import AirtimeBudget, ReplayKalman, transition
from isac_wm.simulator import SceneConfig, ToyScene, communication_bits, recorded_episode


def test_randomization_reproducible():
    a = recorded_episode(42, SceneConfig(steps=12))
    b = recorded_episode(42, SceneConfig(steps=12))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_policy_does_not_change_world_or_camera():
    a, b = ToyScene(42, SceneConfig()), ToyScene(42, SceneConfig())
    for k in range(20):
        a.radio(k, 1)
        b.radio(k, 2)
    np.testing.assert_array_equal(a.truth_for_evaluation(), b.truth_for_evaluation())
    np.testing.assert_array_equal(a.camera(0).payload, b.camera(0).payload)


def test_different_scenes():
    assert not np.array_equal(
        ToyScene(1, SceneConfig()).truth_for_evaluation(), ToyScene(2, SceneConfig()).truth_for_evaluation()
    )


def test_radio_action_couples_cost_and_delay():
    c = SceneConfig(dropout=0, delay_jitter_s=0)
    s = ToyScene(2, c)
    assert s.radio(0, 0) is None
    assert s.radio(0, 2).available_s > s.radio(0, 1).available_s
    assert communication_bits(c, 0) > communication_bits(c, 1) > communication_bits(c, 2) > 0


def test_inputs_do_not_include_labels():
    e = recorded_episode(0, SceneConfig(steps=12))
    assert e["valid"][0].sum() == 0
    assert e["image"][0].sum() == 0 and e["radio"][0].sum() == 0


def test_camera_occlusion_changes_observation_not_truth():
    c = SceneConfig(dropout=0, camera_period=1)
    s = ToyScene(9, c)
    k = s.occlusion_start
    im = s.camera(k).payload
    assert (im[0] - np.maximum(im[1], im[2])).max() < 0.2
    assert np.isfinite(s.truth_for_evaluation()[k]).all()


def test_budget_never_overspends():
    budget = AirtimeBudget(0.1)
    for k in range(100):
        budget.refill()
        budget.authorize(2)
        assert budget.spent <= 0.1 * (k + 1) + 1e-9


def test_kalman_handles_no_measurement():
    x, p = ReplayKalman(10).estimate(1)
    np.testing.assert_allclose(x[:2], [5, 5])
    assert np.linalg.eigvalsh(p).min() > 0


def test_kalman_replay_causal_and_order_independent():
    c = SceneConfig(dropout=0)
    s = ToyScene(5, c)
    packets = [s.radio(k, 1) for k in range(3)]
    a, b = ReplayKalman(10), ReplayKalman(10)
    a.ingest(packets, 1)
    b.ingest(packets[::-1], 1)
    xa, pa = a.estimate(1)
    xb, pb = b.estimate(1)
    np.testing.assert_allclose(xa, xb)
    np.testing.assert_allclose(pa, pb)
    assert np.linalg.eigvalsh(pa).min() > 0
    with pytest.raises(ValueError):
        b.estimate(0.01)


def test_transition_covariance_psd():
    for dt in [0, 0.01, 0.2, 2]:
        _, q = transition(dt)
        assert np.linalg.eigvalsh(q).min() > -1e-12


@pytest.mark.parametrize("kw", [{"dt": 0}, {"steps": 2}, {"dropout": 1}, {"delay_jitter_s": -1}])
def test_bad_config(kw):
    with pytest.raises(ValueError):
        SceneConfig(**kw)
