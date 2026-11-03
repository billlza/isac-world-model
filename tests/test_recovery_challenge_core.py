import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_challenge_core import ChallengeScene, observed_trace, score

from isac_wm.simulator import SceneConfig


class TestRecoveryChallenge(unittest.TestCase):
    def test_prefix_identical_before_turn_and_delivery_anchor(self):
        cfg = SceneConfig(steps=64)
        a = observed_trace(83_000_005, cfg, False)
        b = observed_trace(83_000_005, cfg, True)
        for i in range(24):
            for key in a[0][i]:
                np.testing.assert_array_equal(a[0][i][key], b[0][i][key])
        np.testing.assert_array_equal(a[2][:25, :2], b[2][:25, :2])
        self.assertFalse(np.array_equal(a[2][31, :2], b[2][31, :2]))
        self.assertEqual(a[3], b[3])
        self.assertEqual(a[3]["first_post_blackout_camera_delivery_tick"], 34)
        self.assertTrue(all(o.available_s <= i * cfg.dt + 1e-10 for i, xs in enumerate(a[1]) for o in xs))
        self.assertIsNone(ChallengeScene(83_000_005, cfg, True).camera(24))

    def test_non_recovery_is_not_silently_given_zero_seconds(self):
        truth = np.zeros((64, 2))
        est = truth.copy()
        est[:, 0] = 0.5
        r = score(est, truth, 34)
        self.assertAlmostEqual(r["position_error_auc_m_s"], 0.5)
        self.assertIsNone(r["recovery_confirmed_s"])
        self.assertTrue(r["not_recovered_by_1s"])
        est[37:40] = 0
        self.assertAlmostEqual(score(est, truth, 34)["recovery_confirmed_s"], 0.5)
        with self.assertRaisesRegex(ValueError, "complete"):
            score(est, truth, 55)
