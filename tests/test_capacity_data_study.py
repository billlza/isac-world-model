import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capacity_data_study import (
    INITIALIZATIONS,
    METRICS,
    PARAMETERS,
    WIDTHS,
    cell_name,
    cells,
    design,
    epoch_budget,
    factorial_contrasts,
    registered_seeds,
    run_baseline,
    summarize,
)
from recovery_challenge_core import observed_trace

from isac_wm.baselines import ReplayKalman
from isac_wm.imm import IMMConfig, ReplayIMM
from isac_wm.simulator import SceneConfig


class CapacityDataStudyTests(unittest.TestCase):
    def test_budget_is_updates_and_selection_opportunities_not_epochs(self):
        spec = design("formal")
        self.assertEqual(len(cells(spec)), 8)
        self.assertEqual([epoch_budget(spec, n) for n in (96, 1536)], [128, 8])
        for _, n, _ in cells(spec):
            self.assertEqual(epoch_budget(spec, n) * n // spec["batch_size"], 1536)
        self.assertEqual(spec["optimizer_updates"] // spec["validation_every_updates"], 16)
        self.assertEqual(spec["validation_episodes"], 192)
        small = set(range(spec["train_seed_base"], spec["train_seed_base"] + 96))
        pools = registered_seeds(spec)
        self.assertLess(small, pools["train"])
        smoke = design("smoke")
        smoke_pool = set().union(*registered_seeds(smoke).values())
        self.assertFalse(smoke_pool.intersection(set().union(*pools.values())))
        for old_test in (93621000, 94761000):
            self.assertFalse(pools["test"].intersection(range(old_test, old_test + 24)))
        self.assertEqual(smoke["widths"], [96, 512])
        for width in WIDTHS:
            self.assertEqual(14 * width**2 + 410 * width + 3862, PARAMETERS[width])

    def test_capacity_data_and_initialization_effects_are_separate(self):
        spec = design("formal")
        base = {100: 0.2, 101: 0.7, 102: 1.3}
        panel = {
            (w, n, i): {
                seed: value + 2 * (w == 512) + 3 * (n == 1536) + 5 * (i == 11) + 7 * (w == 512 and n == 1536)
                for seed, value in base.items()
            }
            for w, n, i in cells(spec)
        }
        contrasts = factorial_contrasts(panel, spec)
        self.assertAlmostEqual(contrasts["wide_minus_narrow_at_data_96_mean_over_inits"]["mean"], 2)
        self.assertAlmostEqual(contrasts["wide_minus_narrow_at_data_1536_mean_over_inits"]["mean"], 9)
        self.assertAlmostEqual(contrasts["large_minus_small_data_at_width_512_init_7"]["mean"], 10)
        self.assertAlmostEqual(contrasts["init_11_minus_7_at_width_96_data_96"]["mean"], 5)
        np.testing.assert_allclose(
            contrasts["capacity_by_data_interaction_mean_over_inits"]["trajectory_bootstrap_95"], [7, 7]
        )
        panel[96, 96, 7][100] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            factorial_contrasts(panel, spec)
        panel[96, 96, 7].pop(100)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            factorial_contrasts(panel, spec)

    def test_baseline_forecast_matches_future_prediction_without_advancing_replay_clock(self):
        cfg = SceneConfig(steps=64)
        _, ready, truth, meta = observed_trace(81720000, cfg, True)
        anchor = meta["first_post_blackout_camera_delivery_tick"]
        for cls in (ReplayKalman, lambda room: ReplayIMM(room, config=IMMConfig(gate_threshold=9.21))):
            estimator = cls(cfg.room_m)
            for k, packets in enumerate(ready[:20]):
                now = k * cfg.dt
                estimator.ingest(packets, now)
                state = estimator.estimate(now)[0]
                future = copy.deepcopy(estimator).estimate(now + 3 * cfg.dt)[0][:2]
                np.testing.assert_allclose(state[:2] + 3 * cfg.dt * state[2:], future, rtol=1e-10, atol=1e-10)
            metrics = run_baseline(cls(cfg.room_m), cfg, ready, truth, anchor)
            self.assertTrue(np.isfinite(metrics[METRICS[1]]))
            self.assertEqual(metrics["forecast_origin_ticks"], [5, 60])

    def test_summary_includes_bad_initializations_and_rejects_missing_outcomes(self):
        spec = design("smoke")
        rows = []
        for seed in range(spec["test_seed_base"], spec["test_seed_base"] + 2):
            for condition in ("control", "maneuver"):
                scores = {
                    cell_name(w, n, i): {
                        METRICS[0]: 0.1 if i == 7 else 0.9,
                        METRICS[1]: 0.2 if i == 7 else 1.0,
                        "not_recovered_by_1s": i == 11,
                        "end_window_relapse": False,
                    }
                    for w, n, i in cells(spec)
                }
                for baseline in ("kalman", "gated-imm"):
                    scores[baseline] = {
                        METRICS[0]: 0.3,
                        METRICS[1]: 0.4,
                        "not_recovered_by_1s": False,
                        "end_window_relapse": False,
                    }
                rows.append({"seed": seed, "condition": condition, "scores": scores})
        result = summarize(rows, spec)
        self.assertEqual(len(result["maneuver"]["methods"]), 10)
        marginal = result["maneuver"]["paired_baseline_comparisons"][METRICS[0]][
            "a96-data-8-mean-over-inits"
        ]["gated-imm"]
        self.assertAlmostEqual(marginal["mean"], 0.2)
        self.assertEqual(
            result["maneuver"]["methods"][cell_name(96, 8, INITIALIZATIONS[1])]["unrecovered_by_1s"], 2
        )
        with self.assertRaisesRegex(ValueError, "repeated"):
            summarize(rows + [rows[0]], spec)
        del rows[0]["scores"]["gated-imm"]
        with self.assertRaisesRegex(ValueError, "Every case"):
            summarize(rows, spec)


if __name__ == "__main__":
    unittest.main()
