"""Analytic checks of the frozen-map initialization used by RF correction."""

import numpy as np

from isac_wm.occupancy3d_evaluation import fista_box_inverse


def test_initial_map_retains_an_unobserved_component_without_mutation():
    operator = np.array([[1.0, 1.0]], dtype=np.complex128)
    observed = np.array([[1.0]], dtype=np.complex128)
    learned_prior = np.array([[0.9, 0.1]])
    saved = learned_prior.copy()
    refined = fista_box_inverse(operator, observed, initial=learned_prior, iterations=24, ridge=0)
    zero_initialized = fista_box_inverse(operator, observed, iterations=24, ridge=0)
    np.testing.assert_allclose(refined, [[0.9, 0.1]], atol=1e-10)
    np.testing.assert_allclose(zero_initialized, [[0.5, 0.5]], atol=1e-8)
    np.testing.assert_array_equal(learned_prior, saved)
    np.testing.assert_allclose(refined @ operator.T, observed, atol=1e-10)


def test_rf_correction_updates_observed_component_from_inaccurate_initialization():
    operator = np.diag(np.array([1j, 1.0]))
    target = np.array([[0.0, 1.0]])
    observed = target @ operator.T
    initial = np.array([[0.8, 0.2]])
    refined = fista_box_inverse(operator, observed, initial=initial, iterations=24, ridge=0)
    np.testing.assert_allclose(refined, target, atol=1e-9)
