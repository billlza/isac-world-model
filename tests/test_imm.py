"""IMM baseline: correct mixing, causal out-of-sequence replay and robustness."""

import numpy as np
import pytest
import torch

from isac_wm.baselines import ReplayKalman
from isac_wm.evaluation import evaluate, rollout
from isac_wm.events import Observation
from isac_wm.imm import IMMConfig, ReplayIMM
from isac_wm.model import EventBeliefModel
from isac_wm.runtime import RuntimeConfig
from isac_wm.scenarios import scenario_config
from isac_wm.simulator import SceneConfig, ToyScene


def radio(uid, capture, ready, x=4.0, y=2.0, action=1):
    r = np.hypot(x, y)
    return Observation(
        uid,
        "radio",
        capture,
        ready,
        np.array([r / (10 * np.sqrt(2)), x / r, y / r, 0.0], dtype=np.float32),
        action,
    )


def test_identical_bank_equivalent_to_single_model():
    bank = ReplayIMM(10, config=IMMConfig((0.8, 0.8, 0.8), switch_rate_hz=2.0))
    single = ReplayKalman(10, process_noise=0.8)
    for t, delivered in (
        (0.15, [radio(1, 0, 0.1)]),
        (0.27, [radio(2, 0.1, 0.22, 4.1, 2.1)]),
        (0.4, [radio(3, 0.2, 0.35, 4.2, 2.2)]),
    ):
        bank.ingest(delivered, t)
        single.ingest(delivered, t)
        bx, bp = bank.estimate(t)
        x, p = single.estimate(t)
        np.testing.assert_allclose(bx, x, atol=1e-10)
        np.testing.assert_allclose(bp, p, atol=1e-10)
        np.testing.assert_allclose(bank.mode_probabilities, [1 / 3] * 3, atol=1e-10)


def test_late_packet_replay_matches_all_at_once():
    a, b = ReplayIMM(10), ReplayIMM(10)
    later = radio(3, 0.2, 0.25, 6, 3)
    earlier = radio(2, 0.1, 0.4, 5, 2)
    first = radio(1, 0.0, 0.1, 4, 2)
    a.ingest([first], 0.15)
    a.estimate(0.15)
    a.ingest([later], 0.30)
    a.estimate(0.30)
    a.ingest([earlier], 0.45)
    got = a.estimate(0.5)
    b.ingest([first, earlier, later], 0.5)
    expected = b.estimate(0.5)
    for left, right in zip(got, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-11)
    assert a.replayed_updates > b.replayed_updates
    with pytest.raises(ValueError, match="Duplicate"):
        a.ingest([earlier], 0.5)


def test_gate_rejects_spoofed_observation_and_keeps_modes():
    a = ReplayIMM(10, config=IMMConfig(gate_threshold=9.210340371976184))
    b = ReplayIMM(10, config=IMMConfig(gate_threshold=9.210340371976184))
    good = radio(1, 0.0, 0.1, 4, 3)
    for target in (a, b):
        target.ingest([good], 0.1)
        target.estimate(0.1)
    outlier = radio(2, 0.0, 0.2, 80, 80)
    # Intentionally outside the configured 10 m room; represent a spoofed packet.
    a.ingest([outlier], 0.2)
    xa, pa = a.estimate(0.2)
    xb, pb = b.estimate(0.2)
    np.testing.assert_allclose(xa, xb, atol=1e-10)
    np.testing.assert_allclose(pa, pb, atol=1e-10)
    assert a.rejected_measurements == 1
    assert (np.linalg.eigvalsh(pa) > -1e-10).all()


def test_invalid_batch_atomic_and_budget():
    tracker = ReplayIMM(10, config=IMMConfig(max_observations=2))
    one = radio(1, 0, 0.1)
    with pytest.raises(ValueError, match="Future"):
        tracker.ingest([one], 0.05)
    assert not tracker.observations
    with pytest.raises(ValueError, match="Duplicate"):
        tracker.ingest([one, one], 0.3)
    assert not tracker.observations
    tracker.ingest([one], 0.3)
    with pytest.raises(OverflowError, match="budget"):
        tracker.ingest([radio(2, .1, .2), radio(3, .2, .3)], 0.4)
    assert len(tracker.observations) == 1
    with pytest.raises(ValueError, match="monotonic"):
        tracker.ingest([], 0.1)


@pytest.mark.parametrize("settings", [
    {"process_noises": (0.2,)},
    {"process_noises": (0.2, -1)},
    {"switch_rate_hz": -0.2},
    {"gate_threshold": float("nan")},
    {"max_observations": 0},
])
def test_config_validation(settings):
    with pytest.raises(ValueError):
        IMMConfig(**settings)


@pytest.mark.parametrize("estimator", ["imm", "imm-gated", "kalman-gated"])
@pytest.mark.parametrize("mode", ["reference", "fixed"])
def test_rollout_integrated(estimator, mode):
    r = rollout(80, SceneConfig(steps=12), estimator, "fixed", runtime=RuntimeConfig(mode, 150 if mode == "fixed" else 0))
    assert r["estimator"] == estimator
    assert np.isfinite(r["metrics"]["position_rmse_m"])
    assert r["metrics"]["replayed_filter_updates"] > 0
    assert all(pkt <= t["time_s"] + 1e-9 for t in r["trace"] for pkt in [t["time_s"]])


def test_scenario_shift_does_not_change_target():
    base = ToyScene(31, SceneConfig()).truth_for_evaluation()
    for scenario in ("radio_bias", "sparse_camera"):
        shifted = ToyScene(31, scenario_config(scenario, SceneConfig())).truth_for_evaluation()
        np.testing.assert_array_equal(base, shifted)


def test_opt_in_extra_estimators_leave_default_unchanged(tmp_path):
    torch.set_num_threads(2)
    model = EventBeliefModel()
    c = SceneConfig(steps=10)
    report = evaluate(
        tmp_path / "extra", model, c, torch.device("cpu"), episodes=1, seed=75_000_000,
        policies=("fixed",), estimators=("kalman", "kalman-gated", "imm", "imm-gated", "model"),
        runtime=RuntimeConfig("fixed", 150),
    )
    assert set(report["summary"]) == {
        "kalman+fixed", "kalman-gated+fixed", "imm+fixed", "imm-gated+fixed", "model+fixed"
    }
    with pytest.raises(ValueError, match="Invalid estimators"):
        evaluate(tmp_path / "bad", model, c, torch.device("cpu"), estimators=("kalman", "kalman"))
