import numpy as np
import pytest

from isac_wm.occupancy3d_data import (
    C_M_S,
    GridSpec,
    RFConfig,
    generate_scenes,
    make_operator,
    observe_scenes,
    point_scattering_matrix,
)


@pytest.fixture
def grid():
    return GridSpec(shape=(6, 6, 4))


@pytest.fixture
def rf():
    return RFConfig(n_poses=4, n_frequencies=16, bandwidth_hz=200e6, heldout_n_poses=3)


def test_coordinates_follow_declared_xyz_flattening(grid):
    xyz = grid.coordinates_m.reshape(*grid.shape, 3)
    np.testing.assert_allclose(xyz[0, 0, 0], [-2.5, -2.5, 0.375])
    np.testing.assert_allclose(xyz[1, 0, 0] - xyz[0, 0, 0], [1, 0, 0])
    np.testing.assert_allclose(xyz[0, 0, 1] - xyz[0, 0, 0], [0, 0, 0.75])


def test_point_amplitude_and_round_trip_phase_are_physical():
    xyz = np.array([[2.0, 0, 0], [4.0, 0, 0]])
    frequencies = np.array([24e9, 24e9 + 1e6])
    matrix = point_scattering_matrix(xyz, np.zeros((1, 3)), frequencies)
    np.testing.assert_allclose(np.abs(matrix[:, 0]), 0.25, rtol=1e-6)
    np.testing.assert_allclose(np.abs(matrix[:, 1]), 0.0625, rtol=1e-6)
    expected_phase = np.exp(-1j * 4 * np.pi * 1e6 * 2 / C_M_S)
    np.testing.assert_allclose(matrix[1, 0] / matrix[0, 0], expected_phase, rtol=1e-6)


def test_single_scatterer_forward_matches_kernel(grid, rf):
    op = make_operator(grid, rf)
    occupied = np.zeros(grid.shape, dtype=np.float32)
    occupied[2, 3, 1] = 1
    expected_index = np.ravel_multi_index((2, 3, 1), grid.shape)
    np.testing.assert_allclose(op.forward(occupied), op.matrix[:, expected_index], rtol=1e-6)


def test_height_diversity_breaks_exact_mirror_symmetry(grid, rf):
    planar = make_operator(grid, rf, "same_height")
    diverse = make_operator(grid, rf, "height_diverse")
    index = np.arange(grid.n_voxels).reshape(grid.shape)
    mirrored = index[:, :, ::-1].ravel()
    np.testing.assert_array_equal(planar.matrix, planar.matrix[:, mirrored])
    assert np.linalg.norm(diverse.matrix - diverse.matrix[:, mirrored]) > 1.0
    assert planar.matrix.shape == diverse.matrix.shape
    assert planar.normalization == diverse.normalization
    np.testing.assert_array_equal(planar.sensors_m[:, :2], diverse.sensors_m[:, :2])
    np.testing.assert_array_equal(planar.frequencies_hz, diverse.frequencies_hz)


def test_mirror_families_stay_together_and_splits_are_disjoint(grid):
    train = generate_scenes(12, 101, grid)
    valid = generate_scenes(8, 202, grid, excluded_hashes=set(train.scene_hashes))
    test = generate_scenes(8, 303, grid, excluded_hashes=set(train.scene_hashes + valid.scene_hashes))
    for scenes in (train, valid, test):
        assert len(set(scenes.scene_hashes)) == len(scenes.occupancy)
        for i in range(0, len(scenes.occupancy), 2):
            np.testing.assert_array_equal(scenes.occupancy[i, :, :, ::-1], scenes.occupancy[i + 1])
            assert scenes.pair_ids[i] == scenes.pair_ids[i + 1]
    assert not (set(train.scene_hashes) & set(valid.scene_hashes))
    assert not (set(train.scene_hashes) & set(test.scene_hashes))
    assert not (set(train.pair_ids) & set(valid.pair_ids))
    assert not (set(train.pair_ids) & set(test.pair_ids))


def test_generation_is_reproducible_and_hash_covers_labels(grid):
    first = generate_scenes(8, 11, grid)
    again = generate_scenes(8, 11, grid)
    np.testing.assert_array_equal(first.occupancy, again.occupancy)
    assert first.dataset_hash == again.dataset_hash
    assert first.metadata() == again.metadata()
    changed = generate_scenes(8, 12, grid)
    assert first.dataset_hash != changed.dataset_hash


def test_model_inputs_contain_measurements_and_instrument_information_only(grid, rf):
    scenes = generate_scenes(8, 14, grid)
    observations = observe_scenes(scenes, make_operator(grid, rf), 100)
    inputs = observations.model_inputs()
    assert set(inputs) == {"rf", "sensors_m", "frequencies_hz"}
    assert inputs["rf"] is observations.y
    assert observations.y.shape == (8, rf.n_poses * rf.n_frequencies)
    assert observations.metadata()["normalization_uses_scene_truth"] is False


def test_noise_scale_is_fixed_not_chosen_from_scene_truth(grid, rf):
    first = generate_scenes(8, 1, grid)
    second = generate_scenes(8, 2, grid)
    op = make_operator(grid, rf)
    a, b = observe_scenes(first, op, 33), observe_scenes(second, op, 33)
    assert a.noise_std == b.noise_std == rf.noise_std
    np.testing.assert_allclose(a.y - a.clean_y, b.y - b.clean_y, atol=5e-7, rtol=0)
    again = observe_scenes(first, op, 33)
    np.testing.assert_array_equal(a.y, again.y)


def test_heldout_sensor_positions_are_not_training_positions(grid, rf):
    for layout in ("same_height", "height_diverse"):
        train = make_operator(grid, rf, layout)
        heldout = make_operator(grid, rf, layout, heldout=True)
        assert heldout.matrix.shape[0] == rf.heldout_n_poses * rf.n_frequencies
        distances = np.linalg.norm(train.sensors_m[:, None] - heldout.sensors_m[None], axis=-1)
        assert distances.min() > 0.05
        assert train.normalization == heldout.normalization


def test_mismatch_changes_measurements_without_changing_occupancy_labels(grid, rf):
    scenes = generate_scenes(8, 1, grid)
    before = scenes.occupancy.copy()
    op = make_operator(grid, rf, "height_diverse")
    nominal = observe_scenes(scenes, op, 12, noise_std=0)
    mismatch = observe_scenes(
        scenes, op, 12, noise_std=0, susceptibility=np.full(grid.shape, 1.2, dtype=np.float32)
    )
    np.testing.assert_allclose(mismatch.clean_y, nominal.clean_y * 1.2, atol=5e-6, rtol=1e-6)
    np.testing.assert_array_equal(scenes.occupancy, before)
    assert not mismatch.unit_susceptibility


def test_default_ranges_fit_the_stepped_frequency_interval():
    for layout in ("same_height", "height_diverse"):
        for heldout in (False, True):
            info = make_operator(layout=layout, heldout=heldout).metadata()
            assert info["maximum_sensor_voxel_range_m"] < info["stepped_frequency_unambiguous_range_m"]
            assert info["conventional_range_resolution_m"] == pytest.approx(0.299792458)


def test_invalid_or_aliased_configuration_is_rejected(grid):
    with pytest.raises(ValueError, match="unambiguous"):
        make_operator(grid, RFConfig(n_frequencies=8, bandwidth_hz=1e9))
    with pytest.raises(ValueError, match="even"):
        generate_scenes(7, 0, grid)
    with pytest.raises(ValueError, match="Unknown"):
        make_operator(grid, layout="invented")
    with pytest.raises(ValueError, match="coincides"):
        point_scattering_matrix(np.zeros((1, 3)), np.zeros((1, 3)), np.array([24e9]))
