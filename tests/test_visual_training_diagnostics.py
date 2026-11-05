"""Passive diagnostic observation and resume protocol regression tests."""

import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


@unittest.skipIf(importlib.util.find_spec("torch") is None, "Installed PyTorch is required")
class DiagnosticPassivityTests(unittest.TestCase):
    def setUp(self):
        import torch

        from isac_wm.artifacts import model_fingerprint
        from isac_wm.simulator import SceneConfig, recorded_episode
        from isac_wm.structured import StructuredConfig, build_model
        from isac_wm.training import batchify
        from isac_wm.visual_training_diagnostics import validation_snapshot

        self.torch, self.fingerprint, self.observe = torch, model_fingerprint, validation_snapshot
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = build_model(StructuredConfig(hidden_dim=8))
        self.batch = batchify(
            [
                recorded_episode(
                    234,
                    SceneConfig(
                        steps=8, image_size=16, camera_period=1, camera_dropout=0, occlusion_fraction=0
                    ),
                )
            ],
            torch.device("cpu"),
        )

    def tearDown(self):
        self.torch.set_num_threads(self.threads)

    def test_observer_preserves_parameters_gradients_modes_inputs_and_rng(self):
        torch = self.torch
        self.model.train()
        self.model.visual.eval()
        modes = [module.training for module in self.model.modules()]
        original = {key: value.clone() for key, value in self.batch.items()}
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        parameters = self.fingerprint(self.model)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        observed = self.observe(self.model, self.batch, room_m=10.0, dt=0.1, vision_loss_weight=1.0)
        self.assertEqual(self.fingerprint(self.model), parameters)
        self.assertEqual([module.training for module in self.model.modules()], modes)
        self.assertEqual(random.getstate(), python_rng)
        for a, b in zip(np.random.get_state(), numpy_rng, strict=True):
            np.testing.assert_equal(a, b)
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        for parameter in self.model.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25), rtol=0, atol=0)
        for key in self.batch:
            torch.testing.assert_close(self.batch[key], original[key], rtol=0, atol=0)
        self.assertEqual(
            observed["visual_visible_capture_count"], int(self.batch["vision_target_mask"].sum())
        )
        self.assertEqual(observed["event_counts"]["snapshots"], 8)
        self.assertGreater(observed["visual_position_rmse_m"], 0)
        self.assertIn("attention_residual_to_hidden_rms", observed["network"])
        self.assertIn("acceleration_tanh_saturation_fraction", observed["logits"])
        self.assertIn("vision_noise_sigmoid_upper_saturation_fraction", observed["logits"])
        self.assertIn("radio_noise_sigmoid_upper_saturation_fraction", observed["logits"])
        for modality in ("vision", "radio"):
            self.assertEqual(
                observed["corrections"][modality]["applied_event_count"],
                observed["event_counts"][modality + "_new_valid"],
            )
            self.assertGreaterEqual(observed["corrections"][modality]["position_correction_rms_m"], 0)
        # No Tensor/autograd object or image data is retained in the report.
        encoded = json.dumps(observed, allow_nan=False)
        self.assertNotIn('"image"', encoded)

    def test_empty_visible_support_is_count_zero_and_rmse_null(self):
        batch = {key: value.clone() for key, value in self.batch.items()}
        batch["vision_target_mask"].zero_()
        observed = self.observe(self.model, batch, room_m=10.0, dt=0.1, vision_loss_weight=1.0)
        self.assertEqual(observed["visual_visible_capture_count"], 0)
        self.assertIsNone(observed["visual_position_rmse_m"])
        self.assertEqual(observed["losses"]["vision_mse_normalized"], 0)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "Installed PyTorch is required")
class VisualTrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        import torch

        from isac_wm.artifacts import read_checkpoint, save_checkpoint
        from isac_wm.simulator import SceneConfig
        from isac_wm.structured import StructuredConfig
        from isac_wm.training import train

        self.torch, self.train, self.read, self.save = torch, train, read_checkpoint, save_checkpoint
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.common = dict(
            episodes=2,
            batch_size=1,
            device_name="cpu",
            seed=7,
            scene=SceneConfig(
                steps=8, image_size=16, camera_period=1, camera_dropout=0, occlusion_fraction=0
            ),
            model_config=StructuredConfig(hidden_dim=8),
            train_seed_base=40000,
            validation_seed_base=50000,
            batch_order_seed=7,
            validation_episodes=2,
            validation_every_updates=2,
        )

    def tearDown(self):
        self.torch.set_num_threads(self.threads)
        self.tmp.cleanup()

    def run_train(self, name, epochs=1, **changes):
        return self.train(self.root / name, epochs=epochs, **(self.common | changes))

    def test_diagnostics_do_not_change_zero_weight_optimization_or_selection(self):
        ordinary = self.run_train("plain")
        observed = self.run_train("observed", diagnostic_updates=(0, 1, 2))
        for name in ("resume.pt", "model.pt"):
            a, b = self.read(self.root / "plain" / name), self.read(self.root / "observed" / name)
            self.torch.testing.assert_close(a["state_dict"], b["state_dict"], rtol=0, atol=0)
        a, b = self.read(self.root / "plain/resume.pt"), self.read(self.root / "observed/resume.pt")
        self.torch.testing.assert_close(a["optimizer"], b["optimizer"], rtol=0, atol=0)
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["numpy_rng_json"], b["numpy_rng_json"])
        self.torch.testing.assert_close(a["torch_rng"], b["torch_rng"], rtol=0, atol=0)
        self.assertEqual(
            ordinary["initial_visual_parameters_sha256"], observed["initial_visual_parameters_sha256"]
        )
        self.assertEqual(ordinary["initial_parameters_sha256"], observed["initial_parameters_sha256"])
        report = json.loads((self.root / "observed/visual_diagnostics.json").read_text())
        self.assertEqual(report["protocol"], observed["diagnostic_protocol"])
        self.assertEqual([row["optimizer_updates"] for row in report["snapshots"]], [0, 1, 2])
        self.assertIsNone(report["snapshots"][0]["training"])
        self.assertGreater(report["snapshots"][1]["training"]["branches"]["visual"]["gradient_norm"], 0)
        self.assertGreater(report["snapshots"][1]["training"]["branches"]["visual"]["relative_update"], 0)
        self.assertEqual(observed["event_counts_seen"], observed["training_event_counts_per_epoch"])

    def test_resume_visual_weight_and_diagnostics_matches_uninterrupted(self):
        common = dict(vision_loss_weight=1.0, diagnostic_updates=(0, 1, 2, 4))
        full = self.run_train("full", epochs=2, **common)
        self.run_train("part", **common)
        resumed = self.run_train("resumed", epochs=2, resume=self.root / "part/resume.pt", **common)
        a, b = self.read(self.root / "full/resume.pt"), self.read(self.root / "resumed/resume.pt")
        self.torch.testing.assert_close(a["state_dict"], b["state_dict"], rtol=0, atol=0)
        self.torch.testing.assert_close(a["optimizer"], b["optimizer"], rtol=0, atol=0)
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(full["event_counts_seen"], resumed["event_counts_seen"])
        self.assertEqual(resumed["diagnostic_observed_updates"], [0, 1, 2, 4])
        self.assertEqual(resumed["vision_loss_weight"], 1.0)
        for name, changes in (
            ("weight", {"vision_loss_weight": 0.0}),
            ("schedule", {"diagnostic_updates": (0, 2, 4)}),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Resume .* (weight|protocol)"):
                self.run_train(name, epochs=2, resume=self.root / "part/resume.pt", **(common | changes))

    def test_legacy_resume_defaults_to_zero_weight_and_protocol_changes_are_rejected(self):
        self.run_train("part")
        source = self.read(self.root / "part/resume.pt")
        for name in (
            "vision_loss_weight",
            "vision_supervision_protocol",
            "diagnostic_protocol",
            "visual_diagnostics_snapshots",
            "event_counts_seen",
            "initial_visual_parameters_sha256",
        ):
            source.pop(name)
        self.save(self.root / "legacy.pt", source)
        resumed = self.run_train("legacy", epochs=2, resume=self.root / "legacy.pt")
        self.assertEqual(resumed["vision_loss_weight"], 0.0)
        source = self.read(self.root / "part/resume.pt")
        source["vision_supervision_protocol"]["mask"] = "current target including stale frames"
        self.save(self.root / "wrong.pt", source)
        with self.assertRaisesRegex(ValueError, "visual-supervision protocol"):
            self.run_train("wrong", epochs=2, resume=self.root / "wrong.pt")

    def test_invalid_schedules_or_weights_fail_before_data_generation(self):
        for index, changes in enumerate(
            [
                {"vision_loss_weight": -1},
                {"vision_loss_weight": float("nan")},
                {"diagnostic_updates": (2, 1)},
                {"diagnostic_updates": (1, 1)},
                {"diagnostic_updates": (True,)},
                {"diagnostic_updates": []},
            ]
        ):
            with self.subTest(changes=changes), patch("isac_wm.training.recorded_episode") as generate:
                with self.assertRaises(ValueError):
                    self.run_train(f"invalid-{index}", **changes)
                generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
