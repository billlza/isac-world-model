"""Pretraining scenario coverage, capture identity, and supervision isolation."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from isac_wm.pretraining_data import make_episode, scene_profiles
from isac_wm.simulator import SceneConfig, recorded_episode


def _scene(**changes):
    return replace(
        SceneConfig(steps=32, image_size=16, camera_dropout=0, radio_dropout=0),
        **changes,
    )


def _copy_episode(episode):
    return {key: value.copy() for key, value in episode.items()}


def test_profiles_are_ordered_and_keep_the_batch_coordinate_contract():
    base = _scene(steps=41, dt=0.07, image_size=20, room_m=12.0)
    original = base.to_dict()
    profiles = scene_profiles(base)
    assert tuple(profiles) == (
        "nominal",
        "maneuver",
        "delay",
        "occlusion",
        "radio_outliers",
        "radio_bias",
        "sparse_camera",
        "maneuver_delay",
        "occlusion_outliers",
        "maneuver_bias",
        "combined",
    )
    assert profiles["nominal"] is base
    assert base.to_dict() == original
    for config in profiles.values():
        for name in ("steps", "dt", "image_size", "room_m"):
            assert getattr(config, name) == getattr(base, name)
        assert not (config.camera_dropout == 1 and config.radio_dropout == 1)
    assert profiles["maneuver"].acceleration_std > base.acceleration_std
    assert profiles["delay"].camera_delay_s > base.camera_delay_s
    assert profiles["occlusion"].occlusion_fraction > base.occlusion_fraction
    assert profiles["radio_outliers"].radio_outlier_probability > 0
    assert profiles["radio_bias"].radio_range_bias_m != 0
    assert profiles["combined"].acceleration_std == profiles["maneuver"].acceleration_std
    assert (
        profiles["combined"].radio_outlier_probability == profiles["radio_outliers"].radio_outlier_probability
    )


def test_complete_blackout_is_not_presented_as_a_training_success_profile():
    with pytest.raises(ValueError, match="observability"):
        scene_profiles(_scene(camera_dropout=1, radio_dropout=1))


def test_default_trace_is_unchanged_and_exact_reference_fields_are_arrays():
    config = _scene()
    previous = recorded_episode(123, config)
    current = make_episode(123, config)
    assert set(current) == set(previous) | {"state_reference_variance", "state_reference_validity"}
    for key, value in previous.items():
        np.testing.assert_array_equal(current[key], value)
    for key, expected in (("state_reference_variance", 0), ("state_reference_validity", 1)):
        assert current[key].shape == (config.steps, 4)
        assert current[key].dtype == np.float32
        assert np.all(current[key] == expected)
    assert all(isinstance(value, np.ndarray) for value in current.values())


def test_sensor_faults_do_not_become_label_uncertainty():
    nominal = make_episode(8, _scene())
    faulty = make_episode(
        8,
        _scene(radio_outlier_probability=1, radio_range_bias_m=3, radio_delay_extra_s=0.3),
    )
    np.testing.assert_array_equal(faulty["state_target"], nominal["state_target"])
    assert not np.array_equal(faulty["radio_target"], nominal["radio_target"])
    assert np.all(faulty["state_reference_variance"] == 0)
    assert np.all(faulty["state_reference_validity"] == 1)


def test_appearance_is_deterministic_and_changes_only_images():
    config = _scene()
    plain = make_episode(503, config)
    first = make_episode(503, config, appearance=True)
    repeated = make_episode(503, config, appearance=True)
    assert not np.array_equal(first["image"], plain["image"])
    assert first["image"].dtype == np.float32
    assert np.isfinite(first["image"]).all()
    assert np.all((first["image"] >= 0) & (first["image"] <= 1))
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
        if key != "image":
            np.testing.assert_array_equal(first[key], plain[key])


def test_stale_augmented_frames_are_bitwise_identical_and_empty_frames_stay_empty():
    episode = make_episode(105, _scene(camera_period=3, camera_delay_s=0.3), appearance=True)
    stale_count = 0
    for tick in range(len(episode["image"])):
        if not episode["valid"][tick, 0]:
            assert np.all(episode["image"][tick] == 0)
        elif not episode["new"][tick, 0]:
            stale_count += 1
            assert episode["vision_event_uid"][tick] == -1
            np.testing.assert_array_equal(episode["image"][tick], episode["image"][tick - 1])
    assert stale_count > 3


def test_augmentation_is_keyed_by_capture_uid_not_delivery_tick():
    quick = make_episode(199, _scene(camera_delay_s=0, delay_jitter_s=0), appearance=True)
    delayed = make_episode(199, _scene(camera_delay_s=0.6, delay_jitter_s=0), appearance=True)
    captures = []
    for episode in (quick, delayed):
        captures.append(
            {
                int(uid): episode["image"][tick]
                for tick, uid in enumerate(episode["vision_event_uid"])
                if uid >= 0
            }
        )
    common = captures[0].keys() & captures[1].keys()
    assert len(common) >= 5
    for uid in common:
        np.testing.assert_array_equal(captures[0][uid], captures[1][uid])


def test_appearance_ignores_truth_and_has_no_future_observation_dependency():
    config = _scene()
    original = recorded_episode(77, config)
    altered_labels = _copy_episode(original)
    for key in ("state_target", "radio_target", "vision_capture_target_xy", "vision_target_mask"):
        altered_labels[key].fill(999)
    altered_future = _copy_episode(altered_labels)
    cutoff = 16
    altered_future["image"][cutoff:] = 0.95
    altered_future["radio"][cutoff:] = -123
    augmented = []
    for episode in (original, altered_labels, altered_future):
        with patch("isac_wm.pretraining_data.recorded_episode", return_value=_copy_episode(episode)):
            augmented.append(make_episode(77, config, appearance=True))
    for key in ("image", "radio", "new", "valid", "age", "dt"):
        np.testing.assert_array_equal(augmented[0][key], augmented[1][key])
        np.testing.assert_array_equal(augmented[0][key][:cutoff], augmented[2][key][:cutoff])


def test_missing_camera_remains_missing_under_appearance_augmentation():
    episode = make_episode(818, _scene(camera_dropout=1), appearance=True)
    assert np.all(episode["image"] == 0)
    assert np.all(episode["vision_target_mask"] == 0)
    assert np.all(episode["state_reference_validity"] == 1)


def test_appearance_requires_an_explicit_boolean():
    with pytest.raises(ValueError, match="boolean"):
        make_episode(1, _scene(), appearance=1)
