"""RF-gap causal boundaries, input parity and exogenous trajectory checks."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from isac_wm.gap_recovery import GapRecoverySpec, generate_gap_episode
from isac_wm.simulator import SceneConfig, ToyScene
from isac_wm.training import INPUT_KEYS, batchify


def _scene(**changes):
    return replace(
        SceneConfig(steps=40, image_size=16, dropout=0, radio_dropout=0, delay_jitter_s=0),
        **changes,
    )


@pytest.mark.parametrize("gap_steps", [3, 6, 10])
def test_capture_and_delivery_gaps_are_distinct_and_recovery_counts_are_exact(gap_steps):
    episode = generate_gap_episode(12, _scene(), GapRecoverySpec(gap_steps=gap_steps))
    arrays, meta = episode.arrays, episode.metadata
    assert meta["forecast_anchor_step"] == 12  # capture 1.1 s arrives at 1.16 s
    assert meta["forecast_anchor_capture_s"] == pytest.approx(1.1)
    assert meta["forecast_anchor_available_s"] == pytest.approx(1.16)
    assert meta["capture_gap_start_s"] == pytest.approx(1.2)
    assert meta["first_recovery_step"] == 13 + gap_steps
    assert meta["gap_no_new_steps"] == list(range(13, 13 + gap_steps))
    assert arrays["new"][12, 1] == 1  # delayed pre-gap evidence is retained
    assert np.all(arrays["new"][arrays["phase_gap"], 1] == 0)
    assert np.all(arrays["valid"][arrays["phase_gap"], 1] == 1)
    for tick in meta["gap_no_new_steps"]:
        np.testing.assert_array_equal(arrays["radio"][tick], arrays["radio"][12])
    for count, tick in enumerate(meta["recovery_packet_steps"], 1):
        assert arrays["recovery_packet_count"][tick] == count
    phases = sum(arrays[key].astype(int) for key in ("phase_history", "phase_gap", "phase_recovery"))
    np.testing.assert_array_equal(phases, 1)


def test_only_rf_position_evidence_is_shared_with_baselines_and_counts_balance():
    scene = _scene(delay_jitter_s=0.35, radio_delay_extra_s=0.1)
    episode = generate_gap_episode(53, scene, GapRecoverySpec(gap_steps=10))
    arrays, meta = episode.arrays, episode.metadata
    assert np.all(arrays["image"] == 0)
    assert np.all(arrays["valid"][:, 0] == 0)
    assert np.all(arrays["new"][:, 0] == 0)
    assert np.all(arrays["radio"][:, 3] == 0)
    assert np.all(arrays["radio_target"][:, 3] == 0)
    for tick, observations in enumerate(episode.delivered):
        assert len(observations) == int(arrays["new"][tick, 1])
        for obs in observations:
            assert obs.available_s <= tick * scene.dt + 1e-10
            assert not (meta["capture_gap_start_s"] <= obs.capture_s < meta["capture_gap_end_s"])
            np.testing.assert_array_equal(obs.payload, arrays["radio"][tick])
            assert obs.capture_s == meta["accepted_radio_capture_s"][tick]
            assert obs.available_s == meta["accepted_radio_available_s"][tick]
        assert meta["rf_raw_arrivals_per_tick"][tick] == (
            len(observations)
            + meta["rf_ignored_older_per_tick"][tick]
            + meta["rf_superseded_same_tick"][tick]
        )
    assert (
        meta["rf_acquired_packets"]
        == sum(meta["rf_raw_arrivals_per_tick"]) + meta["rf_pending_after_final_tick"]
    )
    assert sum(meta["rf_ignored_older_per_tick"]) + sum(meta["rf_superseded_same_tick"]) > 0


def test_deterministic_pairing_and_capture_schedule_do_not_change_truth():
    scene = _scene()
    first = generate_gap_episode(71, scene, GapRecoverySpec(gap_steps=3))
    repeated = generate_gap_episode(71, scene, GapRecoverySpec(gap_steps=3))
    longer = generate_gap_episode(71, scene, GapRecoverySpec(gap_steps=10))
    assert first.metadata == repeated.metadata
    for key in first.arrays:
        np.testing.assert_array_equal(first.arrays[key], repeated.arrays[key])
    np.testing.assert_array_equal(first.arrays["state_target"], longer.arrays["state_target"])
    for key in INPUT_KEYS:
        np.testing.assert_array_equal(first.arrays[key][:16], longer.arrays[key][:16])
    assert np.all(first.arrays["state_reference_variance"] == 0)
    assert np.all(first.arrays["state_reference_validity"] == 1)
    batch = batchify([first.arrays, repeated.arrays], torch.device("cpu"))
    assert batch["state_target"].shape == (2, scene.steps, 4)
    assert all(value.shape[0] == scene.steps for value in first.arrays.values())


def test_truth_supervision_and_camera_are_not_online_input_dependencies():
    scene, spec = _scene(), GapRecoverySpec()
    episode = generate_gap_episode(109, scene, spec)
    altered_truth = np.full((scene.steps, 4), 999, np.float32)
    with (
        patch.object(ToyScene, "truth_for_evaluation", return_value=altered_truth),
        patch.object(ToyScene, "camera", side_effect=AssertionError("camera sampled")),
    ):
        changed = generate_gap_episode(109, scene, spec)
    assert not np.array_equal(episode.arrays["state_target"], changed.arrays["state_target"])
    for key in INPUT_KEYS:
        np.testing.assert_array_equal(episode.arrays[key], changed.arrays[key])


def test_missing_history_or_recovery_is_an_explicit_protocol_error():
    with pytest.raises(ValueError, match="No pre-gap"):
        generate_gap_episode(8, _scene(radio_dropout=1), GapRecoverySpec())
    with pytest.raises(ValueError, match="No recovery"):
        generate_gap_episode(8, _scene(steps=20, radio_delay_extra_s=1.0), GapRecoverySpec(gap_steps=3))
    with pytest.raises(ValueError, match="leave history and recovery"):
        generate_gap_episode(8, _scene(), GapRecoverySpec(gap_start_step=35, gap_steps=4))
    with pytest.raises(ValueError, match="gap_steps"):
        GapRecoverySpec(gap_steps=-1)


def test_zero_capture_gap_is_an_explicit_no_gap_control():
    episode = generate_gap_episode(9, _scene(), GapRecoverySpec(gap_steps=0))
    assert not episode.arrays["phase_gap"].any()
    assert episode.metadata["capture_gap_steps"] == []
    assert episode.metadata["gap_no_new_duration_s"] == 0
