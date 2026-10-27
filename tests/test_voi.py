"""Causal delay-aware covariance VoI: numerical, resource and execution tests."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from isac_wm.evaluation import evaluate, rollout
from isac_wm.runtime import RuntimeConfig
from isac_wm.simulator import ACTIONS, SceneConfig, ToyScene
from isac_wm.voi import VoIConfig, choose_sensing_by_voi


X = np.array([3., 4., 0.2, -0.1])


def decision(p=0.5, tokens=0.6, cfg=None, *, scene=None, wait=0.0):
    cov = np.diag([p, p, 0.2, 0.2])
    return choose_sensing_by_voi(
        X, cov, scene or SceneConfig(steps=12), tokens,
        planned_actuation_wait_s=wait, configuration=cfg,
    )


def test_low_uncertainty_skips_high_uncertainty_senses():
    assert decision(0.00001).action == 0
    result = decision(2.0)
    assert result.action in (1, 2)
    assert result.gains_m2[2] > 0
    assert result.lost_mbit[2] > result.lost_mbit[1] > 0


def test_delayed_measurement_has_less_value_at_common_horizon():
    base = decision(0.5)
    late = decision(0.5, scene=SceneConfig(steps=12, radio_delay_extra_s=0.3))
    assert late.gains_m2[1] < base.gains_m2[1]
    assert late.gains_m2[2] < base.gains_m2[2]
    assert late.delivery_ticks[1] > base.delivery_ticks[1]
    assert decision(1, wait=2.0).action == 0
    assert decision(1, scene=SceneConfig(steps=12, radio_delay_extra_s=2.0)).action == 0


def test_price_and_budget_restrict_actions_without_nonfinite_json():
    none = decision(10, tokens=0.05)
    assert none.action == 0
    assert none.scores_m2[1:] == (None, None)
    assert none.to_dict()["score_m2"] == [0.0, None, None]
    only_quick = decision(10, tokens=0.1)
    assert only_quick.action == 1
    assert only_quick.scores_m2[2] is None
    expensive = decision(1, cfg=VoIConfig(tradeoff_m2_per_mbit=1e6))
    assert expensive.action == 0


def test_physical_noise_action_is_modeled_without_oracle():
    r = decision(0.5)
    assert r.gains_m2[1] > 0
    assert r.gains_m2[2] > 0
    assert r.delivery_ticks[2] > r.delivery_ticks[1]
    assert r.lost_mbit[1] == pytest.approx(ACTIONS[1].airtime_fraction *
            (SceneConfig().dt * SceneConfig().bandwidth_hz * np.log2(11))/1e6)


@pytest.mark.parametrize("bad", [
    (np.zeros(2), np.eye(4)),
    (np.zeros(4), np.eye(2)),
    (np.zeros(4), np.diag([-1., 1., 1., 1.])),
    (np.zeros(4), np.full((4, 4), float("nan"))),
    (np.zeros(4), np.triu(np.ones((4, 4)))),
])
def test_invalid_state_fails_closed(bad):
    with pytest.raises(ValueError):
        choose_sensing_by_voi(*bad, SceneConfig(), 0.2)


@pytest.mark.parametrize("cfg", [
    {"horizon_s": 0}, {"horizon_s": float("nan")},
    {"tradeoff_m2_per_mbit": -1}, {"acceleration_spectral_density": -1},
    {"conservatively_delay_jitter": 1},
])
def test_invalid_policy_config_fails_closed(cfg):
    with pytest.raises(ValueError):
        VoIConfig(**cfg)


def test_future_truth_does_not_change_actions(monkeypatch):
    scene = SceneConfig(steps=14)
    base = rollout(190, scene, "imm-gated", "voi")
    original = ToyScene.truth_for_evaluation

    def corrupt_scoring_truth(self):
        altered = original(self)
        altered[:, :2] += 1e4
        return altered

    monkeypatch.setattr(ToyScene, "truth_for_evaluation", corrupt_scoring_truth)
    changed = rollout(190, scene, "imm-gated", "voi")
    assert [r["action"] for r in base["trace"]] == [r["action"] for r in changed["trace"]]
    assert [r["voi_decision"] for r in base["trace"]] == [r["voi_decision"] for r in changed["trace"]]
    assert base["metrics"]["position_rmse_m"] != changed["metrics"]["position_rmse_m"]


@pytest.mark.parametrize("mode,ms", [("reference", 0), ("fixed", 150)])
def test_integrated_voi_is_auditable_and_budgeted(mode, ms):
    config = SceneConfig(steps=16)
    result = rollout(901, config, "imm-gated", "voi", runtime=RuntimeConfig(mode, ms))
    assert result["metrics"]["sensing_airtime_fraction"] <= 0.1 + 1e-9
    assert all(row["action"] in (0, 1, 2) for row in result["trace"])
    assert any(row.get("voi_decision") for row in result["trace"])
    for row in result["trace"]:
        diag = row.get("voi_decision")
        if diag is not None:
            assert len(diag["score_m2"]) == 3
            assert len(diag["delivery_ticks"]) == 3
            assert diag["interpretation"].startswith("Gaussian")
    if mode == "fixed":
        assert all(
            job["published_s"] is None or job["published_s"] >= job["finish_s"]
            for job in result["jobs"]
        )
        assert all(job["executed_action"] is None or job["published_s"] is not None for job in result["jobs"])


def test_policy_in_normal_evaluation_and_serialization(tmp_path):
    from isac_wm.model import EventBeliefModel, ModelConfig
    model = EventBeliefModel(ModelConfig(16, 32)).eval()
    report = evaluate(
        tmp_path / "voi", model, replace(SceneConfig(), steps=12),
        torch.device("cpu"), episodes=1, policies=("fixed", "voi"),
        estimators=("kalman", "model"),
    )
    assert "kalman+voi" in report["summary"]
    assert "model+voi" in report["summary"]
    assert (tmp_path / "voi/evaluation.json").is_file()


def test_explicit_no_sensing_negative_control():
    result = rollout(505, SceneConfig(steps=14), "imm-gated", "skip")
    assert result["metrics"]["sensing_airtime_fraction"] == 0.0
    assert all(row["action"] == 0 for row in result["trace"])


def test_changed_voi_tradeoff_is_bound_to_calibration(tmp_path):
    from isac_wm.calibration import calibrate, validate_calibration
    from isac_wm.model import EventBeliefModel, ModelConfig

    model = EventBeliefModel(ModelConfig(16, 32))
    scene = SceneConfig(steps=10)
    cfg = VoIConfig(tradeoff_m2_per_mbit=4.0)
    cal = calibrate(tmp_path / "cal.json", model, scene, torch.device("cpu"),
                    episodes=10, seed=17_000_000, policies=("voi",), voi_config=cfg)
    validate_calibration(cal, model, scene, [30_000_000], voi_config=cfg)
    with pytest.raises(ValueError, match="VoI configuration mismatch"):
        validate_calibration(cal, model, scene, [30_000_000], voi_config=VoIConfig())
