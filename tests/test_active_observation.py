"""Physical identifiability, causal release, resource caps and learned observer."""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from isac_wm.active_observation import (
    METHODS,
    SCENARIOS,
    Anchor,
    ExperimentConfig,
    HypothesisMemory,
    ObservationNetwork,
    SyntheticWorld,
    choose_action,
    expected_error_reduction,
    observation_geometry,
    run_episode,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("active_study", ROOT / "scripts/active_observation_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture(scope="module")
def physical():
    world = SyntheticWorld.create(871, "same")
    anchor, report = world.initial_rf()
    return world, anchor, report


def packet(memory, *, uid=1, lateral=4.0, capture=1.0, observation=None):
    mean, sigma = memory.predict(lateral, capture)
    return {
        "uid": uid,
        "lateral_m": lateral,
        "capture_s": capture,
        "available_s": capture + 0.25,
        "prediction_mean_deg": mean.tolist(),
        "prediction_sigma_deg": sigma.tolist(),
        "observation_deg": float(mean[0]) if observation is None else observation,
    }


def test_geometric_identifiability_depends_on_viewpoint():
    cfg = ExperimentConfig()
    anchor = Anchor(18, 12, 0, 0.1)
    at_origin, origin_sigma, _ = observation_geometry(anchor, 0, 1, cfg)
    shifted, sigma, _ = observation_geometry(anchor, 4, 1, cfg)
    np.testing.assert_allclose(at_origin, [12, 12])
    assert abs(shifted[0] - shifted[1]) > 2 * max(sigma)
    assert expected_error_reduction(np.array([0.5, 0.5]), at_origin, origin_sigma) < 0.001
    assert expected_error_reduction(np.array([0.5, 0.5]), shifted, sigma) > 0.2


def test_no_update_before_arrival_no_duplicate_and_no_partial_failure():
    memory = HypothesisMemory(Anchor(18, 12, 0, 0.1), ExperimentConfig())
    event = packet(memory)
    prior = memory.weights().copy()
    with pytest.raises(ValueError, match="Future"):
        memory.release(event, event["capture_s"])
    np.testing.assert_array_equal(memory.weights(), prior)
    assert not memory.seen
    malformed = copy.deepcopy(event)
    malformed["prediction_sigma_deg"][0] = -1
    with pytest.raises(ValueError):
        memory.release(malformed, event["available_s"])
    assert not memory.seen
    memory.release(event, event["available_s"])
    assert memory.weights()[0] > prior[0]
    with pytest.raises(ValueError, match="Duplicate"):
        memory.release(event, event["available_s"])
    with pytest.raises(ValueError, match="past"):
        memory.predict(0, 1.0)


def test_ambiguous_views_never_confirm_even_with_extreme_neural_confidence():
    class MisleadingObserver(ObservationNetwork):
        def predict(self, features, base_mean, base_sigma, memory):
            return base_mean + torch.tensor([0.0, 10.0]), torch.full_like(base_sigma, 0.01)

    memory = HypothesisMemory(Anchor(18, 12, 0, 0.1), ExperimentConfig(), MisleadingObserver())
    for i in range(4):
        event = packet(memory, uid=i, lateral=0, capture=1.0 + i)
        memory.release(event, event["available_s"])
    assert memory.weights()[0] > 0.999
    assert memory.status() == "unknown"
    assert memory.informative == 0


def test_new_evidence_can_reject_all_candidates_and_missing_detection_is_unknown():
    memory = HypothesisMemory(Anchor(18, 12, 0, 0.1), ExperimentConfig())
    event = packet(memory)
    event["observation_deg"] = None
    memory.release(event, event["available_s"])
    assert memory.accepted == 0 and memory.status() == "unknown"
    for i in (2, 3):
        event = packet(memory, uid=i, capture=float(i), observation=-45)
        memory.release(event, event["available_s"])
    assert memory.status() == "candidate_set_rejected"
    chosen, _ = choose_action(
        memory, "geometry_voi", decision_s=4, position_m=4, remaining_m=8, deadline_s=8, step=3, seed=1
    )
    assert chosen is None


def test_new_evidence_corrects_initial_wrong_preference():
    memory = HypothesisMemory(Anchor(18, 12, 0, 0.1), ExperimentConfig())
    assert memory.weights()[0] == pytest.approx(0.8)
    for i in range(3):
        event = packet(memory, uid=i, lateral=4, capture=1 + i)
        event["observation_deg"] = event["prediction_mean_deg"][1]
        memory.release(event, event["available_s"])
    assert memory.status() == "farther_object"


@pytest.mark.parametrize("method", METHODS)
def test_real_phy_and_renderer_follow_causal_budgets(physical, method):
    world, anchor, report = physical
    network = ObservationNetwork(method != "learned_no_memory") if method.startswith("learned") else None
    output = run_episode(world, method, ExperimentConfig(), anchor=anchor, phy_report=report, network=network)
    times = [r["time_s"] for r in output["trace"]]
    assert times == sorted(times)
    assert output["metrics"]["travel_m"] <= output["config"]["travel_budget_m"]
    assert output["metrics"]["capture_count"] <= output["config"]["steps"]
    for previous, row in zip(output["trace"], output["trace"][1:]):
        if row["phase"] != "arrival":
            assert row["weights"] == previous["weights"]
            assert row["observation_deg"] is None
    for event in output["measurements"]:
        assert anchor.available_s <= event["decision_s"] < event["capture_s"] < event["available_s"]
        assert event["available_s"] <= output["trace"][-1]["time_s"]
    poisoned = copy.deepcopy(report)
    poisoned["truth_for_evaluation_only"] = {"range_m": -999, "bearing_deg": -999}
    second = run_episode(
        world, method, ExperimentConfig(), anchor=anchor, phy_report=poisoned, network=network
    )
    assert output == second  # Neither decision nor inference reads report truth.


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_failure_scenarios_are_exercised(physical, scenario):
    original, anchor, report = physical
    world = SyntheticWorld.create(original.seed, scenario)
    output = run_episode(world, "fixed", ExperimentConfig(), anchor=anchor, phy_report=report)
    if scenario in {"unresolvable", "camera_loss"}:
        assert output["metrics"]["final_status"] == "unknown"
    if scenario == "camera_loss":
        assert output["metrics"]["missing_detections"] == 4
    if scenario == "late":
        assert output["metrics"]["late_packets"] > 0
        assert any(row["phase"] == "waiting" for row in output["trace"])
    if scenario == "outside_candidates":
        assert output["metrics"]["candidate_set_rejected"]


def test_action_cannot_spend_unavailable_resources_or_use_late_evidence():
    cfg = ExperimentConfig(delivery_s=10)
    memory = HypothesisMemory(Anchor(18, 12, 0, 0.1), cfg)
    chosen, candidates = choose_action(
        memory, "fixed", decision_s=1, position_m=0, remaining_m=0, deadline_s=3, step=0, seed=1
    )
    assert chosen is None and not candidates
    with pytest.raises(ValueError):
        choose_action(
            memory, "fixed", decision_s=0, position_m=0, remaining_m=0, deadline_s=3, step=0, seed=1
        )


def test_observer_predicts_before_seeing_target_and_memory_changes_only_from_evidence():
    torch.manual_seed(61)
    net = ObservationNetwork()
    features = torch.zeros(2, 4)
    base = torch.tensor([5.0, 9.0])
    sigma = torch.ones(2)
    hidden = torch.zeros(2, 12)
    before, _ = net.predict(features, base, sigma, hidden)
    observation = torch.tensor(6.0, requires_grad=True)
    after_hidden = net.observe(features, base, observation, hidden)
    after, _ = net.predict(features, base, sigma, after_hidden)
    assert not torch.allclose(before, after)
    assert torch.autograd.grad(before.sum(), observation, allow_unused=True)[0] is None
    assert torch.autograd.grad(after.sum(), observation)[0].abs() > 0
    net.use_memory = False
    without, _ = net.predict(features, base, sigma, after_hidden)
    np.testing.assert_allclose(without.detach(), before.detach())
    assert not torch.count_nonzero(net.observe(features, base, observation, hidden))


@pytest.mark.parametrize(
    "values",
    [
        {"steps": 1},
        {"packet_bits": 0},
        {"delivery_s": float("nan")},
        {"max_baseline_m": -1},
        {"confirmation_score": 1.0},
        {"rejection_sigma": 1},
        {"camera_speed_mps": 0.1},
    ],
)
def test_invalid_config_is_explicit(values):
    with pytest.raises(ValueError):
        ExperimentConfig(**values)


def test_invalid_anchor_geometry_packets_and_method(physical):
    world, anchor, report = physical
    for change in ({"range_m": -1}, {"bearing_deg": float("inf")}, {"range_sigma_m": 0}):
        with pytest.raises(ValueError):
            replace(anchor, **change)
    memory = HypothesisMemory(anchor, ExperimentConfig())
    for lateral, capture in ((float("nan"), 2), (0, 0)):
        with pytest.raises(ValueError):
            observation_geometry(anchor, lateral, capture, memory.cfg)
    with pytest.raises(ValueError):
        observation_geometry(replace(anchor, radial_speed_mps=-99), 1, 10, memory.cfg)
    event = packet(memory, capture=1)
    event["prediction_mean_deg"] = [1]
    with pytest.raises(ValueError):
        memory.release(event, 2)
    with pytest.raises(ValueError):
        run_episode(world, "unknown", memory.cfg, anchor=anchor, phy_report=report)
    with pytest.raises(ValueError):
        run_episode(world, "learned_memory", memory.cfg, anchor=anchor, phy_report=report)
    with pytest.raises(ValueError):
        SyntheticWorld.create(-1, "same")
    with pytest.raises(ValueError):
        SyntheticWorld.create(1, "fiction")


def test_small_training_and_portable_replay(tmp_path):
    torch.set_num_threads(2)
    out = tmp_path / "study"
    result = study.run_study(
        out, train_episodes=2, validation_episodes=2, test_episodes=1, epochs=1, seed=11500
    )
    assert len(result["episodes"]) == len(SCENARIOS) * len(METHODS)
    splits = result["provenance"]["splits"]
    assert not (set(splits["train"]) & set(splits["validation"]))
    assert not ((set(splits["train"]) | set(splits["validation"])) & set(splits["test"]))
    portable = json.loads((out / "matlab_replay.json").read_text())
    assert len(portable["summary"]) == len(SCENARIOS) * len(METHODS)
    assert portable["provenance"]["status"] == "completed"
    assert (out / "matlab_replay.json").stat().st_size < 1_000_000
    assert (out / "learned_memory.pt").stat().st_size < 100_000
    assert set(portable["provenance"]["checkpoint_sha256"]) == {"learned_memory", "learned_no_memory"}
    for row in portable["summary"]:
        if row["method"] == "always_unknown":
            assert row["mean_predictive_nll"] is None and row["mean_predictive_nll_count"] == 0
    with pytest.raises(FileExistsError):
        study.run_study(out)


def test_failed_study_retains_error_and_never_claims_success(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate generator failure")

    monkeypatch.setattr(study, "dataset", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        study.run_study(
            tmp_path / "failed", train_episodes=2, validation_episodes=2, test_episodes=1, epochs=1
        )
    manifest = json.loads((tmp_path / "failed" / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["failure_type"] == "RuntimeError"
    assert not (tmp_path / "failed" / "study.json").exists()


def test_study_rejects_bad_dimensions_and_incomplete_results(tmp_path):
    with pytest.raises(ValueError):
        study.run_study(tmp_path / "bad", epochs=0)
    with pytest.raises(ValueError):
        study.run_study(tmp_path / "bad", seed=2**32)
    with pytest.raises(ValueError):
        study.aggregate([])
