"""Read-only replay identities, numerical paths and descriptive error tails."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from isac_wm import checkpoint_replay as replay
from isac_wm.artifacts import (
    model_fingerprint,
    read_checkpoint,
    read_resume_checkpoint,
    save_resume_checkpoint,
    sha256_file,
)
from isac_wm.simulator import SceneConfig
from isac_wm.structured import StructuredConfig, build_model
from isac_wm.training import train


def replay_fixture(
    manifest,
    kind,
    source_commit,
    checkpoint_sha,
    manifest_sha,
    state_fingerprint,
    final_score=None,
):
    """Synthetic schema fixture only; zero errors are not an experimental result."""
    seeds = manifest["validation_episode_seeds"]
    steps = manifest["scene"]["steps"]
    zero = [0.0] * (steps - 5)
    trajectories = [
        {
            "seed": seed,
            "position_error_m": zero.copy(),
            "position_error": replay.error_summary(zero),
            "visible_captures": [],
            "strata_masks": {
                "fresh_visible_vision": [False] * len(zero),
                "fresh_occluded_vision": [False] * len(zero),
                "no_fresh_vision": [True] * len(zero),
                "radio_fresh": [False] * len(zero),
                "radio_not_fresh": [True] * len(zero),
                "camera_unavailable": [True] * len(zero),
                "camera_age_lt_0.2s": [False] * len(zero),
                "camera_age_0.2_to_0.5s": [False] * len(zero),
                "camera_age_0.5_to_1s": [False] * len(zero),
                "camera_age_ge_1s": [False] * len(zero),
            },
        }
        for seed in seeds
    ]
    path = {
        "reference_position_rmse_m": 0.0,
        "position_error": replay.error_summary(zero * len(seeds)),
        "trajectory_rmse": replay.error_summary([0.0] * len(seeds)),
        "visual_capture_error": replay.error_summary([]),
        "strata": {
            name: replay.error_summary([0.0] * (sum(mask) * len(seeds)))
            for name, mask in trajectories[0]["strata_masks"].items()
        },
        "trajectories": trajectories,
    }
    arrays = {
        "mean": np.zeros((len(seeds), steps, 4)),
        "logvar": np.zeros((len(seeds), steps, 4)),
        "visual_xy": np.zeros((len(seeds), steps, 2)),
    }
    prefix = {key: values[:4] for key, values in arrays.items()}
    score = manifest["best_validation_rmse_m"] if kind == "best" or final_score is None else final_score
    report = {
        "schema_version": 1,
        "kind": "checkpoint-validation-replay",
        "status": "completed",
        "profile": "development",
        "device": "cpu",
        "runtime": {"device": "cpu"},
        "checkpoint_kind": kind,
        "checkpoint_source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "state_dict_fingerprint": state_fingerprint,
        "weights_update": manifest["best_update"] if kind == "best" else manifest["optimizer_updates"],
        "selected_best_update": manifest["best_update"],
        "model_config": manifest["model"],
        "scene_config": manifest["scene"],
        "validation_episode_seeds": seeds,
        "fresh_test_consumed": False,
        "deployment_certification": False,
        "unchanged_model": True,
        "rng_preserved": True,
        "protocol": {
            "scored_tick_start": 5,
            "steps": steps,
            "reference_batch_size": 8,
            "prefix_episodes": min(4, len(seeds)),
            "error_thresholds_m": list(replay.ERROR_THRESHOLDS_M),
        },
        "paths": {"batched": copy.deepcopy(path), "streaming": copy.deepcopy(path)},
        "comparisons": {
            "full_streaming_vs_batched": replay.compare_outputs(
                arrays, arrays, seeds, manifest["scene"]["room_m"]
            ),
            "first4_streaming_vs_batched": replay.compare_outputs(
                prefix, prefix, seeds[:4], manifest["scene"]["room_m"]
            ),
            "first4_batched4_vs_batched8": replay.compare_outputs(
                prefix, prefix, seeds[:4], manifest["scene"]["room_m"]
            ),
        },
        "recorded_score": {
            "expected_rmse_m": score,
            "replayed_rmse_m": 0.0,
            "delta_m": -score,
            "atol_m": replay.SCORE_ATOL_M,
            "rtol": replay.SCORE_RTOL,
            "within_registered_tolerance": abs(score) <= replay.SCORE_ATOL_M + replay.SCORE_RTOL * abs(score),
        },
    }
    report["paths"]["streaming"]["correction_diagnostics"] = {
        modality: {
            "applied_event_count": 0,
            "correction_length": replay.error_summary([]),
            "innovation_length": replay.error_summary([]),
        }
        for modality in ("vision", "radio")
    }
    return report


@pytest.fixture(scope="module")
def trained_cell(tmp_path_factory):
    torch.set_num_threads(2)
    folder = tmp_path_factory.mktemp("read-only-replay") / "cell"
    manifest = train(
        folder,
        epochs=2,
        episodes=8,
        batch_size=8,
        device_name="cpu",
        seed=7,
        scene=SceneConfig(steps=12, image_size=16),
        model_config=StructuredConfig(
            feature_dim=8, hidden_dim=16, memory_size=4, heads=4, latent_state_norm="layernorm"
        ),
        learning_rate=0.0003,
        train_seed_base=89200000,
        validation_seed_base=89210000,
        batch_order_seed=37,
        validation_episodes=8,
        validation_every_updates=1,
        vision_loss_weight=1.0,
        diagnostic_updates=(0, 1, 2),
    )
    return folder, manifest


def arguments(folder, kind="best"):
    filename = "model.pt" if kind == "best" else "resume.pt"
    return {
        "checkpoint": folder / filename,
        "manifest": folder / "manifest.json",
        "kind": kind,
        "device_name": "cpu",
        "profile": "development",
        "source_commit": "1" * 40,
        "expected_checkpoint_sha256": sha256_file(folder / filename),
        "expected_manifest_sha256": sha256_file(folder / "manifest.json"),
    }


def validation_arguments(args):
    return {key: value for key, value in args.items() if key not in {"checkpoint", "manifest", "output"}}


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_tail_aggregation_thresholds_and_empty_support():
    values = [0, 0.1, 0.2, 0.5, 1, 2]
    result = replay.error_summary(values)
    assert result["count"] == 6
    assert result["rmse_m"] == pytest.approx(math.sqrt(5.3 / 6))
    assert result["p95_m"] == pytest.approx(1.75)
    assert result["p99_m"] == pytest.approx(1.95)
    assert result["max_m"] == 2
    assert result["exceedance"]["0.1"]["count"] == 4
    assert result["exceedance"]["1"]["count"] == 1
    empty = replay.error_summary([])
    assert empty["count"] == 0 and empty["rmse_m"] is None
    assert empty["exceedance"]["1"]["fraction"] is None
    assert replay.error_summary([1e308, 1e308])["rmse_m"] == 1e308


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-1]])
def test_invalid_tail_values_fail(values):
    with pytest.raises((ValueError, FloatingPointError)):
        replay.error_summary(values)


def test_numeric_disagreement_is_measured_with_first_divergence():
    reference = {
        "mean": np.zeros((2, 3, 4)),
        "logvar": np.zeros((2, 3, 4)),
        "visual_xy": np.zeros((2, 3, 2)),
    }
    candidate = copy.deepcopy(reference)
    candidate["mean"][0, 1, 0] = 0.001
    candidate["mean"][1, 0, 1] = 0.002
    result = replay.compare_outputs(reference, candidate, [7, 11], 10)
    assert result["within_registered_tolerance"] is False
    assert result["exceeding_tick_count"] == 2
    assert result["first_divergence"] == {"seed": 11, "tick": 0}
    assert result["position_delta"]["max_m"] == 0.02
    replay._validate_comparison(result, [7, 11], 3)
    candidate["mean"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        replay.compare_outputs(reference, candidate, [7, 11], 10)


def test_actual_checkpoint_replays_reference_and_preserves_rng_and_source(
    trained_cell, tmp_path, monkeypatch
):
    folder, manifest = trained_cell
    args = arguments(folder)
    before = rng_state()
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("Replay called an optimizer"))
    output = tmp_path / "replay.json"
    result = replay.evaluate_checkpoint(**args, output=output)
    assert_rng_equal(before, rng_state())
    assert result["recorded_score"]["delta_m"] == 0
    assert result["paths"]["batched"]["reference_position_rmse_m"] == manifest["best_validation_rmse_m"]
    assert result["paths"]["batched"]["position_error"]["count"] == 8 * 7
    assert result["paths"]["batched"]["visual_capture_error"]["count"] > 0
    assert result["checkpoint_sha256"] == sha256_file(args["checkpoint"])
    assert result["manifest_sha256"] == sha256_file(args["manifest"])
    assert replay.validate_replay_report(result, manifest, **validation_arguments(args))
    assert json.loads(output.read_text()) == result
    second = replay.evaluate_checkpoint(**args)
    for field in ("paths", "comparisons", "recorded_score", "state_dict_fingerprint"):
        assert second[field] == result[field]
    with pytest.raises(ValueError, match="already exists"):
        replay.evaluate_checkpoint(**args, output=output)


def test_final_reads_current_tensors_not_embedded_best(trained_cell, tmp_path):
    folder, manifest = trained_cell
    payload = read_resume_checkpoint(folder / "resume.pt")
    payload["state_dict"] = {key: value.clone() for key, value in payload["state_dict"].items()}
    payload["state_dict"]["noise.bias"].add_(0.25)
    current = replay.state_dict_fingerprint(payload["state_dict"])
    selected = replay.state_dict_fingerprint(payload["best_checkpoint"]["state_dict"])
    assert current != selected
    checkpoint = tmp_path / "resume.pt"
    save_resume_checkpoint(checkpoint, payload)
    args = arguments(folder, "final") | {
        "checkpoint": checkpoint,
        "expected_checkpoint_sha256": sha256_file(checkpoint),
    }
    result = replay.evaluate_checkpoint(**args)
    assert result["weights_update"] == manifest["optimizer_updates"]
    assert result["state_dict_fingerprint"] == current
    assert result["state_dict_fingerprint"] != selected
    assert result["recorded_score"]["within_registered_tolerance"] is False
    assert replay.validate_replay_report(result, manifest, **validation_arguments(args))


def test_private_model_replay_preserves_modes_and_tensors(trained_cell):
    folder, manifest = trained_cell
    model = build_model(StructuredConfig(**manifest["model"]))
    model.load_state_dict(read_checkpoint(folder / "model.pt")["state_dict"])
    model.train()
    model.visual.eval()
    modes = [module.training for module in model.modules()]
    fingerprint, rng = model_fingerprint(model), rng_state()
    replay._replay_model(model, manifest, torch.device("cpu"), "development")
    assert [module.training for module in model.modules()] == modes
    assert model_fingerprint(model) == fingerprint
    assert_rng_equal(rng, rng_state())


def test_nonfinite_failure_is_retained_and_rng_is_restored(trained_cell, tmp_path, monkeypatch):
    folder, _ = trained_cell
    original = replay._cpu_output

    def inject_failure(output):
        output["mean"] = output["mean"].clone()
        output["mean"][0, 0, 0] = float("nan")
        return original(output)

    monkeypatch.setattr(replay, "_cpu_output", inject_failure)
    output = tmp_path / "failed.json"
    before = rng_state()
    with pytest.raises(FloatingPointError, match="Non-finite model output"):
        replay.evaluate_checkpoint(**arguments(folder), output=output)
    assert_rng_equal(before, rng_state())
    failure = json.loads(output.read_text())
    assert failure["status"] == "failed"
    assert failure["error"]["type"] == "FloatingPointError"


def test_source_hash_kind_and_byte_bound_rejections(trained_cell, tmp_path, monkeypatch):
    folder, _ = trained_cell
    args = arguments(folder)
    with pytest.raises(ValueError, match="Manifest SHA"):
        replay.evaluate_checkpoint(**(args | {"expected_manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="differs from study"):
        replay.evaluate_checkpoint(**(args | {"expected_checkpoint_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="continuation fields"):
        replay.evaluate_checkpoint(**(args | {"kind": "final"}))
    monkeypatch.setattr(replay, "MAX_CHECKPOINT_BYTES", 1)
    called = []
    original = replay.sha256_file

    def track_hash(path):
        called.append(Path(path))
        return original(path)

    monkeypatch.setattr(replay, "sha256_file", track_hash)
    with pytest.raises(ValueError, match="byte budget"):
        replay.evaluate_checkpoint(**args)
    assert args["checkpoint"] not in called


def test_schema_and_new_optimizer_policy_bound_to_checkpoint(trained_cell):
    folder, manifest = trained_cell
    payload = read_checkpoint(folder / "model.pt")
    bad = copy.deepcopy(payload)
    bad["schema_version"] = 1
    with pytest.raises(ValueError, match="schema version"):
        replay._validate_payload(bad, manifest, "best")
    changed = manifest | {"optimizer_policy": "grouped-visual", "width_lr_scale": 0.5}
    with pytest.raises(ValueError, match="optimizer_policy"):
        replay._validate_payload(payload, changed, "best")
    aligned = payload | {"optimizer_policy": "grouped-visual", "width_lr_scale": 0.5}
    replay._validate_payload(aligned, changed, "best")


def test_pure_report_validator_rejects_edited_tail_and_identity(trained_cell):
    folder, manifest = trained_cell
    args = arguments(folder)
    result = replay_fixture(
        manifest,
        "best",
        args["source_commit"],
        args["expected_checkpoint_sha256"],
        args["expected_manifest_sha256"],
        replay.state_dict_fingerprint(read_checkpoint(folder / "model.pt")["state_dict"]),
    )
    assert replay.validate_replay_report(result, manifest, **validation_arguments(args))
    bad = copy.deepcopy(result)
    bad["paths"]["batched"]["position_error"]["p99_m"] = 1
    with pytest.raises(ValueError, match="summary differs"):
        replay.validate_replay_report(bad, manifest, **validation_arguments(args))
    bad = copy.deepcopy(result)
    bad["weights_update"] += 1
    with pytest.raises(ValueError, match="weights_update"):
        replay.validate_replay_report(bad, manifest, **validation_arguments(args))
    bad = copy.deepcopy(result)
    bad["fresh_test_consumed"] = True
    with pytest.raises(ValueError, match="fresh_test_consumed"):
        replay.validate_replay_report(bad, manifest, **validation_arguments(args))


def test_model_input_boundary_excludes_truth_and_future_labels():
    values = {key: object() for key in (*replay.INPUT_KEYS, "action_target")}
    values.update(state_target=object(), vision_capture_target_xy=object(), radio_target=object())
    assert set(replay._model_inputs(values)) == {*replay.INPUT_KEYS, "action_target"}


def test_profile_rejects_formal_data_in_development_and_wrong_device(trained_cell):
    _, manifest = trained_cell
    changed = copy.deepcopy(manifest)
    changed["validation_episode_seeds"] = list(replay.FORMAL_VALIDATION_SEEDS[:8])
    with pytest.raises(ValueError, match="formal episodes"):
        replay._validate_protocol(changed, "development", "cpu")
    with pytest.raises(ValueError, match="profile/device"):
        replay._validate_protocol(manifest, "formal", "cpu")
