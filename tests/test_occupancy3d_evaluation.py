"""Checks against analytic inverse solutions and hand-computed geometry cases."""

import unittest

import numpy as np

from isac_wm.occupancy3d_evaluation import (
    complex_nmse,
    fista_box_inverse,
    fixed_height_25d,
    matched_filter,
    occupancy_metrics,
    paired_family_bootstrap,
    select_fista_configuration,
    select_fixed_height_configuration,
    select_occupancy_threshold,
)


class Occupancy3DEvaluationTest(unittest.TestCase):
    def test_complex_phase_backprojection_and_inverse(self):
        # These columns are orthogonal over C; a real-only treatment is wrong.
        operator = np.array([[1, 1j], [1j, 1]], dtype=np.complex128)
        truth = np.array([[0.2, 0.8], [0.6, 0.1]])
        measurements = truth @ operator.T
        np.testing.assert_allclose(matched_filter(operator, measurements), truth, atol=1e-12)
        np.testing.assert_allclose(
            fista_box_inverse(operator, measurements, ridge=0, iterations=40), truth, atol=1e-9
        )
        np.testing.assert_allclose(complex_nmse(measurements, measurements), 0)
        self.assertGreater(float(complex_nmse(measurements.real[0], measurements[0])), 0)

    def test_box_constraints_and_known_ridge_solution(self):
        operator = np.diag(np.array([1j, 2], dtype=np.complex128))
        observations = operator @ np.array([0.4, 0.8])
        expected = np.array([0.4 / 1.5, 3.2 / 4.5])
        np.testing.assert_allclose(
            fista_box_inverse(operator, observations, ridge=0.5, iterations=100), expected, atol=2e-7
        )
        observations = operator @ np.array([2.0, -1.0])
        np.testing.assert_allclose(
            fista_box_inverse(operator, observations, ridge=0, iterations=80), [1, 0], atol=1e-9
        )

    def test_fixed_profile_has_no_per_scene_height_freedom(self):
        shape = (2, 1, 3)
        profile = np.array([1.0, 1.0, 0.0])
        truth = np.array([0.2, 0.7])[:, None, None] * profile
        operator = np.eye(6, dtype=np.complex128)
        reconstructed = fixed_height_25d(operator, truth.ravel(), shape, profile, ridge=0, iterations=50)
        self.assertEqual(reconstructed.shape, shape)
        np.testing.assert_allclose(reconstructed, truth, atol=1e-9)
        np.testing.assert_array_equal(reconstructed[:, :, 2], 0)

    def test_height_union_penalizes_missed_and_false_positive_columns(self):
        truth = np.zeros((2, 2, 3))
        truth[0, 0, :2] = 1
        predicted = np.zeros_like(truth)
        predicted[0, 0, 0] = 1
        predicted[1, 1, 0] = 1
        metrics = occupancy_metrics(predicted, truth, (1, 1, 0.4))
        self.assertAlmostEqual(metrics["iou_3d"], 1 / 3)
        self.assertAlmostEqual(metrics["brier"], 2 / 12)
        self.assertAlmostEqual(metrics["height_column_rmse_m"], 0.4)
        self.assertEqual(metrics["false_positive_columns"], 1)
        self.assertEqual(metrics["missed_columns"], 0)
        self.assertEqual(metrics["height_evaluated_columns"], 2)
        missed = occupancy_metrics(np.zeros_like(truth), truth, (1, 1, 0.4))
        self.assertAlmostEqual(missed["height_column_rmse_m"], 0.8)
        self.assertEqual(missed["missed_columns"], 1)
        self.assertIsNone(missed["surface_chamfer_m"])
        self.assertEqual(missed["empty_predicted_surface"], 1)

    def test_exact_geometry_and_known_surface_translation(self):
        truth = np.zeros((3, 2, 2))
        truth[0, 0, 0] = 1
        exact = occupancy_metrics(truth, truth, (0.5, 0.75, 1.0))
        self.assertEqual(exact["iou_3d"], 1)
        self.assertEqual(exact["surface_chamfer_m"], 0)
        self.assertEqual(exact["surface_f1"], 1)
        shifted = np.zeros_like(truth)
        shifted[1, 0, 0] = 1
        metrics = occupancy_metrics(shifted, truth, (0.5, 0.75, 1.0), surface_tolerance_m=0.25)
        self.assertAlmostEqual(metrics["surface_chamfer_m"], 0.5)
        self.assertEqual(metrics["surface_f1"], 0)

    def test_unknown_voxels_do_not_become_free_labels_or_height_truth(self):
        truth = np.zeros((2, 2, 3))
        truth[0, 0, 0] = 1
        probability = truth.copy()
        probability[0, 0, 2] = 1
        valid = np.ones_like(truth, dtype=bool)
        valid[0, 0, 2] = False
        metrics = occupancy_metrics(probability, truth, (1, 1, 1), valid_mask=valid)
        self.assertEqual(metrics["iou_3d"], 1)
        self.assertEqual(metrics["brier"], 0)
        self.assertEqual(metrics["height_evaluated_columns"], 0)
        with self.assertRaises(ValueError):
            occupancy_metrics(probability, truth, (1, 1, 1), valid_mask=np.zeros_like(valid))

    def test_threshold_selection_uses_macro_validation_iou(self):
        probability = np.array([[0.2, 0.4], [0.1, 0.9]])
        truth = np.array([[0, 1], [0, 1]])
        result = select_occupancy_threshold(probability, truth, candidates=[0.1, 0.3, 0.5])
        self.assertEqual(result["threshold"], 0.3)
        self.assertEqual(result["validation_macro_iou_3d"], 1)

    def test_bootstrap_resamples_families_not_individual_mirrors(self):
        result = paired_family_bootstrap([0, 0, 0, 10], [0, 0, 0, 0], ["a", "a", "a", "b"], repetitions=300)
        duplicated = paired_family_bootstrap([0, 10], [0, 0], ["a", "b"], repetitions=300)
        self.assertEqual(result["mean_difference_a_minus_b"], 5)
        self.assertEqual(result["independent_families"], 2)
        self.assertEqual(result["lower"], duplicated["lower"])
        self.assertEqual(result["upper"], duplicated["upper"])

    def test_configuration_selection_records_each_validation_candidate(self):
        operator = np.eye(2, dtype=np.complex128)
        target = np.array([[0, 1], [1, 0]])
        result = select_fista_configuration(
            operator,
            target.astype(complex),
            target,
            operator,
            target.astype(complex),
            iteration_candidates=(20, 40),
            ridge_candidates=(0.0, 0.1),
        )
        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(result["selected"]["ridge"], 0)
        self.assertEqual(result["selection_split"], "validation")
        self.assertGreater(result["selected"]["validation_macro_iou_3d"], 0.99)

    def test_fixed_profile_selection_evaluates_expanded_3d_geometry(self):
        operator = np.eye(4, dtype=np.complex128)
        truth = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]).reshape(2, 2, 1, 2)
        measurements = truth.reshape(2, 4).astype(complex)
        result = select_fixed_height_configuration(
            operator,
            measurements,
            truth,
            operator,
            measurements,
            (2, 1, 2),
            {"all_heights": np.ones(2), "floor": np.array([1.0, 0.0])},
            iteration_candidates=(20,),
            ridge_candidates=(0.0,),
        )
        self.assertEqual(result["selected"]["profile_name"], "floor")
        self.assertEqual(result["selected"]["validation_macro_iou_3d"], 1)
        self.assertEqual(len(result["candidates"]), 2)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            complex_nmse(np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            matched_filter(np.zeros((3, 2)), np.ones(3))
        with self.assertRaises(ValueError):
            occupancy_metrics(np.full((2, 2, 2), np.nan), np.zeros((2, 2, 2)), (1, 1, 1))
        with self.assertRaises(ValueError):
            paired_family_bootstrap([0, np.nan], [0, 0], [1, 2])


if __name__ == "__main__":
    unittest.main()
