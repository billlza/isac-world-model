import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crossed_seed_study import CELLS, design, factorial_auc


class CrossedSeedTests(unittest.TestCase):
    def test_main_effects_and_interaction_are_not_confounded(self):
        # Fixed increments of 2 and 3, plus 5 only when both factors are set to 11.
        base = {9000: 0.3, 9001: 1.2, 9002: 0.9}
        cells = {}
        for i, d in CELLS:
            cells[i, d] = {
                k: v + 2 * (i == 11) + 3 * (d == 11) + 5 * (i == d == 11) for k, v in base.items()
            }
        differences = factorial_auc(cells)
        self.assertAlmostEqual(differences["init_11_minus_7_average"]["mean_m_s"], 4.5)
        self.assertAlmostEqual(differences["data_11_minus_7_average"]["mean_m_s"], 5.5)
        np.testing.assert_allclose(differences["interaction"]["trajectory_bootstrap_95_m_s"], [5.0, 5.0])
        del cells[7, 11][9000]
        with self.assertRaisesRegex(ValueError, "mismatched"):
            factorial_auc(cells)

    def test_development_never_scores_a_registered_test_trajectory(self):
        formal, smoke = design("formal"), design("smoke")
        def corpus(s):
            result = set()
            for b in (7, 11):
                result.update(range(s["train_offset"] + b * 100_000, s["train_offset"] + b * 100_000 + s["train_episodes"]))
            ranges = [(s["validation_base"], max(8, s["train_episodes"] // 8))]
            ranges.extend((v, s["challenge_episodes"]) for v in s["splits"].values())
            for start, n in ranges:
                rows = set(range(start, start + n))
                self.assertFalse(result.intersection(rows))
                result.update(rows)
            return result
        self.assertFalse(corpus(formal).intersection(corpus(smoke)))

    def test_resume_must_preserve_seed_plan(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Installed PyTorch needed to exercise actual optimizer resume")
        from isac_wm.model import ModelConfig
        from isac_wm.simulator import SceneConfig
        from isac_wm.training import train

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(epochs=1, episodes=2, batch_size=2, seed=7, scene=SceneConfig(steps=16),
                        device_name="cpu", model_config=ModelConfig(feature_dim=8, hidden_dim=8))
            plan = dict(train_seed_base=1000, validation_seed_base=2000, batch_order_seed=37)
            first = train(root / "train", **args, **plan)
            self.assertEqual(first["train_episode_seeds"], [1000, 1001])
            self.assertEqual(first["validation_episode_seeds"], list(range(2000, 2008)))
            args.update(epochs=2, resume=root / "train" / "resume.pt")
            with self.assertRaisesRegex(ValueError, "seed plan"):
                train(root / "changed", **args, **(plan | {"batch_order_seed": 7}))
            resumed = train(root / "correct", **args, **plan)
            self.assertEqual(resumed["completed_epochs"], 2)
