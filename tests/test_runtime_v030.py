"""Regression tests for real-time causality and execution-contract binding."""

import json
import sys

import numpy as np
import pytest
import torch

from isac_wm import cli
from isac_wm.calibration import calibrate, load_calibration, validate_calibration
from isac_wm.evaluation import evaluate, rollout
from isac_wm.events import Observation
from isac_wm.model import EventBeliefModel, ModelConfig
from isac_wm.runtime import RuntimeConfig
from isac_wm.simulator import SceneConfig, recorded_episode
from isac_wm.streaming import propagate
from isac_wm.training import batchify, seed_all


@pytest.fixture
def small_model():
    torch.set_num_threads(2)
    seed_all(13)
    return EventBeliefModel(ModelConfig(16, 32)).eval()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"compute_ms": 12},
        {"compute_ms": True},
        {"mode": "fixed", "compute_ms": -1},
        {"mode": "fixed", "compute_ms": float("nan")},
        {"mode": "fixed", "compute_ms": float("inf")},
        {"mode": "fixed", "queue_capacity": 0},
        {"mode": "fixed", "queue_capacity": True},
        {"queue_capacity": 3},
        {"mode": "measured", "compute_ms": 1},
        {"mode": "fixed", "compute_ms": 3_600_001},
        {"output_acceleration_noise": -1},
    ],
)
def test_invalid_runtime(kwargs):
    with pytest.raises(ValueError):
        RuntimeConfig(**kwargs)


def test_runtime_contract_roundtrip():
    r = RuntimeConfig("fixed", 120, 20)
    assert RuntimeConfig.from_dict(r.to_dict()) == r
    assert r.service_seconds(0.002) == 0.12
    assert RuntimeConfig("measured").service_seconds(0.002) == 0.002
    with pytest.raises(ValueError):
        RuntimeConfig.from_dict({"protocol_version": "unknown"})


@pytest.mark.parametrize("estimator", ["kalman", "model"])
def test_fixed_zero_equivalent_to_reference(small_model, estimator):
    c = SceneConfig(steps=12)
    args = (731, c, estimator, "fixed", small_model, torch.device("cpu"))
    a = rollout(*args)
    b = rollout(*args, runtime=RuntimeConfig("fixed"))
    for x, y in zip(a["trace"], b["trace"]):
        np.testing.assert_allclose(x["estimate"], y["estimate"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(x["position_variance"], y["position_variance"], rtol=1e-6)
        assert x["action"] == y["action"]
        assert x["arrived_ids"] == y["arrived_ids"]


@pytest.mark.parametrize("delay_ms", [50, 150, 250])
@pytest.mark.parametrize("estimator", ["kalman", "model"])
def test_delayed_publication_and_actions(small_model, estimator, delay_ms):
    c = SceneConfig(steps=16)
    r = rollout(
        765, c, estimator, "fixed", small_model, torch.device("cpu"), runtime=RuntimeConfig("fixed", delay_ms)
    )
    for job in r["jobs"]:
        assert job["input_cutoff_s"] == job["start_s"]
        assert job["max_input_available_s"] is None or job["max_input_available_s"] <= job["start_s"] + 1e-10
        if job["published_s"] is not None:
            assert job["published_s"] + 1e-12 >= job["finish_s"]
            assert job["published_s"] - job["finish_s"] < c.dt + 1e-9
    for row in r["trace"]:
        if row["output_available"]:
            assert row["output_ready_s"] <= row["time_s"] + 1e-12
            assert row["output_state_time_s"] == row["time_s"]
        else:
            assert row["status"] == "no_output" and not row["accepted"]
        if row["action"]:
            assert row["new_output"]
            assert row["step"] * 3 + row["action"] in row["captured_ids"]
    assert r["trace"][0]["action"] == 0
    assert r["metrics"]["sensing_airtime_fraction"] <= 0.1 + 1e-9
    assert r["metrics"]["unfinished_jobs"] == 1


def test_cameras_continue_during_busy_worker():
    c = SceneConfig(steps=20, camera_dropout=0, occlusion_fraction=0)
    a = rollout(98, c, "kalman", "fixed", runtime=RuntimeConfig("fixed", 0))
    b = rollout(98, c, "kalman", "fixed", runtime=RuntimeConfig("fixed", 350))

    def cameras(r):
        return [i for row in r["trace"] for i in row["captured_ids"] if i % 3 == 0]

    assert cameras(a) == cameras(b)
    assert b["metrics"]["busy_tick_fraction"] > 0
    assert b["metrics"]["completed_jobs"] < a["metrics"]["completed_jobs"]


def test_overflow_explicit_and_bounded():
    c = SceneConfig(steps=30, camera_period=1, camera_dropout=0, occlusion_fraction=0)
    r = rollout(42, c, "kalman", "fixed", runtime=RuntimeConfig("fixed", 800, queue_capacity=1))
    assert r["metrics"]["queue_peak"] <= 1
    assert r["metrics"]["overflow_dropped_count"] > 0
    assert sum(len(x["overflow_dropped_ids"]) for x in r["trace"]) == r["metrics"]["overflow_dropped_count"]


def test_job_never_completes_not_silently_scored_as_new_output():
    r = rollout(9, SceneConfig(steps=8), "kalman", "fixed", runtime=RuntimeConfig("fixed", 10000))
    assert r["metrics"]["no_output_fraction"] == 1
    assert r["metrics"]["acceptance_fraction"] == 0
    assert r["metrics"]["started_jobs"] == 1
    assert r["metrics"]["action_skip_count"] == 8


def test_time_advance_uses_model_not_future_truth(monkeypatch):
    from isac_wm.simulator import ToyScene

    c = SceneConfig(steps=12)
    a = rollout(99, c, "kalman", "fixed", runtime=RuntimeConfig("fixed", 150))
    original = ToyScene.truth_for_evaluation
    monkeypatch.setattr(ToyScene, "truth_for_evaluation", lambda self: original(self) + 500)
    b = rollout(99, c, "kalman", "fixed", runtime=RuntimeConfig("fixed", 150))
    for x, y in zip(a["trace"], b["trace"]):
        assert x["estimate"] == y["estimate"]
        assert x["action"] == y["action"]
    assert a["metrics"]["position_rmse_m"] != b["metrics"]["position_rmse_m"]


def test_covariance_propagation_full_and_positive():
    state = np.array([1.0, 2.0, 3.0, 4.0])
    cov = np.eye(4)
    out, p = propagate(state, cov, 0.2, 0.8)
    np.testing.assert_allclose(out, [1.6, 2.8, 3, 4])
    np.testing.assert_allclose(p, p.T)
    assert np.linalg.eigvalsh(p).min() > 0
    with pytest.raises(ValueError):
        propagate(state, cov, -0.1, 0.8)


def test_prefix_bitwise_causality_when_future_batch_shape_changes(small_model):
    c = SceneConfig(steps=10)
    b = batchify([recorded_episode(30, c), recorded_episode(31, c)], torch.device("cpu"))
    altered = {k: v.clone() for k, v in b.items()}
    altered["new"][:, 6:] = 1
    altered["valid"][:, 6:] = 1
    altered["image"][:, 6:] = torch.rand_like(altered["image"][:, 6:])
    with torch.no_grad():
        a, z = small_model(b), small_model(altered)
    torch.testing.assert_close(a["mean"][:, :6], z["mean"][:, :6], rtol=0, atol=0)


def test_batched_encoding_has_zero_future_input_gradient(small_model):
    c = SceneConfig(steps=10)
    b = batchify([recorded_episode(30, c)], torch.device("cpu"))
    b["image"] = b["image"].requires_grad_()
    b["radio"] = b["radio"].requires_grad_()
    out = small_model(b, batched_encoding=True)
    out["mean"][:, :6].sum().backward()
    assert torch.count_nonzero(b["image"].grad[:, 6:]) == 0
    assert torch.count_nonzero(b["radio"].grad[:, 6:]) == 0


def test_calibration_bound_to_runtime(tmp_path, small_model):
    c = SceneConfig(steps=8)
    rt = RuntimeConfig("fixed", 50)
    path = tmp_path / "calibration.json"
    d = calibrate(path, small_model, c, torch.device("cpu"), 9, 2000, 0.2, ("fixed",), runtime=rt)
    loaded = load_calibration(path)
    assert loaded["runtime"] == rt.to_dict()
    validate_calibration(d, small_model, c, [3000], runtime=rt)
    for other in (RuntimeConfig(), RuntimeConfig("fixed", 60), RuntimeConfig("fixed", 50, 2)):
        with pytest.raises(ValueError, match="runtime"):
            validate_calibration(d, small_model, c, [3000], runtime=other)
    evaluate(
        tmp_path / "eval",
        small_model,
        c,
        torch.device("cpu"),
        1,
        3000,
        policies=("fixed",),
        calibration=d,
        runtime=rt,
    )


def test_measured_timing_cannot_reuse_calibration(tmp_path, small_model):
    with pytest.raises(ValueError, match="fixed"):
        calibrate(
            tmp_path / "x",
            small_model,
            SceneConfig(steps=8),
            torch.device("cpu"),
            runtime=RuntimeConfig("measured"),
        )
    with pytest.raises(ValueError, match="diagnostic"):
        rollout(2, SceneConfig(steps=8), "kalman", "fixed", quantile=1, runtime=RuntimeConfig("measured"))


@pytest.mark.parametrize("field", ["capture_s", "available_s"])
def test_timestamp_bool_rejected(field):
    kw = dict(uid=3, modality="radio", capture_s=0.0, available_s=1.0, payload=np.ones(4), action=1)
    kw[field] = True
    with pytest.raises(ValueError):
        Observation(**kw)


def test_stress_exit_nonzero_and_retains_failure(tmp_path, monkeypatch, small_model):
    monkeypatch.setattr(
        cli, "load_model", lambda *a: (small_model, SceneConfig(steps=8), torch.device("cpu"))
    )

    def broken(*a, **kw):
        raise ValueError("injected scenario failure")

    monkeypatch.setattr(cli, "evaluate", broken)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isac-wm",
            "stress",
            "--checkpoint",
            "unused.pt",
            "--out",
            str(tmp_path / "stress"),
            "--scenarios",
            "nominal",
        ],
    )
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
    result = json.loads((tmp_path / "stress" / "stress.json").read_text())
    assert result["scenarios"]["nominal"]["status"] == "failed"
