"""Update-budget controls exercise real optimization and checkpoint continuation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipIf(importlib.util.find_spec("torch") is None, "Installed PyTorch is required")
class ControlledTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch

        from isac_wm.artifacts import read_checkpoint
        from isac_wm.model import ModelConfig
        from isac_wm.simulator import SceneConfig
        from isac_wm.training import train

        self.torch, self.read_checkpoint, self.train = torch, read_checkpoint, train
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.common = dict(
            episodes=4,
            batch_size=2,
            device_name="cpu",
            seed=7,
            scene=SceneConfig(steps=8, image_size=16),
            model_config=ModelConfig(feature_dim=8, hidden_dim=8),
            train_seed_base=50_000,
            validation_seed_base=60_000,
            batch_order_seed=37,
            validation_episodes=3,
            validation_every_updates=3,
        )

    def tearDown(self):
        self.torch.set_num_threads(self.previous_threads)
        self.tmp.cleanup()

    def run_train(self, name, epochs, **kwargs):
        return self.train(self.root / name, epochs=epochs, **(self.common | kwargs))

    def read_history(self, name):
        return json.loads((self.root / name / "history.json").read_text())

    def assert_same_continuation(self, first, second):
        a = self.read_checkpoint(self.root / first / "resume.pt")
        b = self.read_checkpoint(self.root / second / "resume.pt")
        for field in ("state_dict", "optimizer", "torch_rng"):
            self.torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)
        for field in (
            "completed_epochs",
            "optimizer_updates",
            "best_rmse",
            "history",
            "epoch_history",
            "numpy_rng_json",
            "pending_validation_loss_sum",
            "pending_validation_loss_count",
        ):
            self.assertEqual(a[field], b[field], field)
        x = self.read_checkpoint(self.root / first / "model.pt")
        y = self.read_checkpoint(self.root / second / "model.pt")
        self.torch.testing.assert_close(x["state_dict"], y["state_dict"], rtol=0, atol=0)
        self.assertEqual(x["best_update"], y["best_update"])

    def test_different_data_sizes_get_same_updates_validation_corpus_and_opportunities(self):
        small = self.run_train("small", 3)
        large = self.run_train("large", 1, episodes=12)
        for name, result, completed_epochs in (("small", small, 3), ("large", large, 1)):
            self.assertEqual(result["optimizer_updates"], 6)
            self.assertEqual(result["planned_optimizer_updates"], 6)
            self.assertEqual(result["validation_updates"], [3, 6])
            self.assertEqual(result["validation_opportunities"], 2)
            self.assertEqual(result["planned_validation_opportunities"], 2)
            self.assertEqual(result["validation_episode_seeds"], [60_000, 60_001, 60_002])
            self.assertEqual(result["completed_epochs"], completed_epochs)
            self.assertEqual(result["model_selection_status"], "selected")
            self.assertIn(result["best_update"], [3, 6])
            self.assertNotIn("cuda_peak_allocated_bytes", result)
            self.assertGreater(result["training_seconds"], 0)
            epochs = json.loads((self.root / name / "epoch_history.json").read_text())
            self.assertEqual(len(epochs), completed_epochs)
        self.assertEqual(small["initial_parameters_sha256"], large["initial_parameters_sha256"])
        self.assertEqual(small["train_episode_seeds"], large["train_episode_seeds"][:4])
        # Both validate at step 3, with different epoch progress; neither rounds
        # validation to an epoch boundary or adds an initial untrained candidate.
        self.assertEqual(self.read_history("small")[0]["completed_epochs"], 1)
        self.assertEqual(self.read_history("large")[0]["completed_epochs"], 0)

    def test_resume_before_first_validation_and_no_unscheduled_final_candidate(self):
        full = self.run_train("full", 2)
        part = self.run_train("part", 1)
        self.assertEqual(part["completed_epochs"], 1)
        self.assertEqual(part["optimizer_updates"], 2)
        self.assertEqual(part["validation_opportunities"], 0)
        self.assertEqual(part["validation_updates"], [])
        self.assertEqual(part["model_selection_status"], "no_validation_yet")
        self.assertIsNone(part["best_update"])
        self.assertIsNone(part["best_validation_rmse_m"])
        self.assertFalse((self.root / "part/model.pt").exists())
        self.assertTrue((self.root / "part/resume.pt").is_file())
        resumed = self.run_train("resumed", 2, resume=self.root / "part/resume.pt")
        self.assertEqual(full["validation_updates"], [3])
        self.assertEqual(resumed["validation_updates"], [3])
        self.assertEqual(resumed["optimizer_updates"], 4)
        self.assertEqual(resumed["optimizer_updates_this_call"], 2)
        self.assert_same_continuation("full", "resumed")

    def test_resume_after_selection_keeps_partial_interval_and_matches_uninterrupted(self):
        self.run_train("full", 3)
        self.run_train("part", 2)
        partial = self.read_checkpoint(self.root / "part/resume.pt")
        self.assertEqual(partial["pending_validation_loss_count"], 1)
        resumed = self.run_train("resumed", 3, resume=self.root / "part/resume.pt")
        self.assertEqual(resumed["validation_updates"], [3, 6])
        self.assert_same_continuation("full", "resumed")

    def test_resume_preserves_earlier_best_when_later_validation_deteriorates(self):
        with patch("isac_wm.training._validation_position_rmse", side_effect=[1.0, 2.0, 3.0]):
            self.run_train("part", 1, episodes=2, validation_every_updates=1)
            resumed = self.run_train(
                "resumed", 3, episodes=2, validation_every_updates=1, resume=self.root / "part/resume.pt"
            )
        old = self.read_checkpoint(self.root / "part/model.pt")
        best = self.read_checkpoint(self.root / "resumed/model.pt")
        self.torch.testing.assert_close(old["state_dict"], best["state_dict"], rtol=0, atol=0)
        self.assertEqual(resumed["best_update"], 1)
        self.assertEqual(resumed["best_validation_rmse_m"], 1.0)
        self.assertEqual(resumed["validation_updates"], [1, 2, 3])

    def test_epoch_validation_and_legacy_resume_contract_still_work(self):
        from isac_wm.artifacts import save_checkpoint

        common = dict(validation_episodes=None, validation_every_updates=None)
        full = self.run_train("full", 2, **common)
        self.run_train("part", 1, **common)
        legacy = self.read_checkpoint(self.root / "part/resume.pt")
        for field in (
            "optimizer_updates",
            "epoch_history",
            "pending_validation_loss_sum",
            "pending_validation_loss_count",
            "validation_episodes",
            "validation_every_updates",
        ):
            del legacy[field]
        for field in ("best_update", "validation_episodes", "validation_every_updates"):
            del legacy["best_checkpoint"][field]
        save_checkpoint(self.root / "legacy.pt", legacy)
        resumed = self.run_train("resumed", 2, resume=self.root / "legacy.pt", **common)
        self.assertEqual(full["validation_updates"], [2, 4])
        self.assertEqual(resumed["validation_updates"], [2, 4])
        self.assertEqual(resumed["validation_episode_seeds"], list(range(60_000, 60_008)))
        self.assertEqual(
            set(self.read_history("resumed")[0]), {"epoch", "train_loss", "validation_position_rmse_m"}
        )
        a = self.read_checkpoint(self.root / "full/resume.pt")
        b = self.read_checkpoint(self.root / "resumed/resume.pt")
        self.torch.testing.assert_close(a["state_dict"], b["state_dict"], rtol=0, atol=0)
        self.assertEqual(a["history"], b["history"])

    def test_resume_rejects_validation_count_schedule_and_update_count_changes(self):
        from isac_wm.artifacts import save_checkpoint

        self.run_train("part", 1)
        source = self.root / "part/resume.pt"
        for name, change in (
            ("count", {"validation_episodes": 4}),
            ("interval", {"validation_every_updates": 2}),
            ("mode", {"validation_every_updates": None}),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "validation protocol"):
                self.run_train(name, 2, resume=source, **change)
        with self.assertRaisesRegex(ValueError, "total epoch target"):
            self.run_train("no_new_epoch", 1, resume=source)
        saved = self.read_checkpoint(source)
        saved["optimizer_updates"] += 1
        save_checkpoint(self.root / "broken.pt", saved)
        with self.assertRaisesRegex(ValueError, "epoch boundary"):
            self.run_train("bad_count", 2, resume=self.root / "broken.pt")

    def test_invalid_control_values_are_rejected_before_dataset_generation(self):
        for i, change in enumerate(
            [
                {"validation_episodes": 0},
                {"validation_episodes": True},
                {"validation_every_updates": 0},
                {"validation_every_updates": 1.5},
                {"patience": 1},
            ]
        ):
            with (
                self.subTest(change=change),
                patch("isac_wm.training.recorded_episode") as generator,
                self.assertRaises(ValueError),
            ):
                self.run_train(f"invalid-{i}", 1, **change)
            generator.assert_not_called()

    def test_explicit_validation_count_is_in_memory_budget_and_split_guard(self):
        with patch("isac_wm.training.recorded_episode") as generator:
            with self.assertRaises(MemoryError):
                self.run_train("budget", 1, validation_episodes=1000, max_dataset_bytes=1_000_000)
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.run_train("overlap", 1, validation_seed_base=49_998)
        generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
