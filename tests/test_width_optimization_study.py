"""Registered Phase D comparisons and provenance; fixtures are not research results."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import normalized_capacity_study as phase_c
import width_optimization_study as study
from test_normalized_capacity_study import evidence_fixture, numerical_fixture

HAS_TORCH = importlib.util.find_spec("torch") is not None


def records_fixture(spec):
    """Synthetic validated metadata used to exercise paired-arm failure detection."""
    records = {}
    for key in study.cells(spec):
        width, scale, seed = key
        evidence = evidence_fixture(spec, (width, seed))
        manifest = evidence[0]
        manifest.update(width_lr_scale=scale, optimizer_policy=study.expected_optimizer_policy(width, scale))
        records[study.cell_name(*key)] = {
            "status": "completed",
            "hidden_width": width,
            "width_lr_scale": scale,
            "initialization": seed,
            "parameters": study.PARAMETERS_BY_WIDTH[width],
            "visual_encoder_parameters": study.VISUAL_PARAMETERS,
            "latent_state_norm": "layernorm",
            "learning_rate": study.LEARNING_RATE,
            "vision_loss_weight": study.VISION_WEIGHT,
            "initial_parameters_sha256": manifest["initial_parameters_sha256"],
            "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
            "optimizer_policy": manifest["optimizer_policy"],
            "numerical_settings": numerical_fixture(),
            "metrics": study.validate_training_evidence(*evidence, spec, key, "cuda"),
        }
    return records


class RegistrationTests(unittest.TestCase):
    def test_single_optimizer_intervention_preserves_capacity_protocol(self):
        spec, previous = study.design("formal"), phase_c.design("formal")
        self.assertEqual(
            study.cells(spec),
            ((512, 1.0, 7), (512, 1.0, 11), (2048, 1.0, 7), (2048, 1.0, 11), (2048, 0.5, 7), (2048, 0.5, 11)),
        )
        self.assertEqual(len({study.cell_name(*key) for key in study.cells(spec)}), 6)
        for field in (
            "scene",
            "training_episodes",
            "train_seed_base",
            "batch_size",
            "optimizer_updates",
            "validation_episodes",
            "validation_seed_base",
            "validation_every_updates",
            "selection_metric",
            "diagnostic_updates",
            "learning_rate",
            "vision_loss_weight",
            "minibatch_order_seed",
            "normalization",
        ):
            self.assertEqual(spec[field], previous[field], field)
        self.assertEqual(study.validation_updates(spec), [96, 192, 288, 384])
        self.assertEqual(study.epoch_budget(spec), 2)
        self.assertEqual(study.PARAMETERS_BY_WIDTH[2048], 59_563_798)
        self.assertFalse(spec["deployment_validated"])
        self.assertEqual(spec["checkpoint_replay"]["exceedance_thresholds_m"], [0.1, 0.25, 0.5, 1.0])
        self.assertIn("not calibrated", spec["checkpoint_replay"]["threshold_scope"])
        self.assertNotIn("test_seed_base", spec)

    def test_development_uses_all_real_models_and_disjoint_data(self):
        spec, formal = study.design("development"), study.design("formal")
        self.assertEqual(study.cells(spec), study.cells(formal))
        self.assertEqual(spec["models_by_width"], formal["models_by_width"])
        self.assertEqual(spec["optimizer_updates"], 4)
        self.assertEqual(spec["training_episodes"], 8)
        self.assertEqual(spec["validation_episodes"], 8)
        self.assertEqual(spec["batch_size"], 8)
        self.assertEqual(spec["diagnostic_updates"], [0, 1, 2, 3, 4])
        self.assertEqual(spec["scene"]["steps"], 64)
        pools, old_pools = study.registered_seeds(spec), study.registered_seeds(formal)
        train_seeds, validation_seeds = pools["train"], pools["validation"]
        old_train, old_val = old_pools["train"], old_pools["validation"]
        self.assertFalse(train_seeds & validation_seeds)
        self.assertFalse((train_seeds | validation_seeds) & (old_train | old_val))
        for previous_profile in (phase_c.design("development"),):
            old = study.registered_seeds(previous_profile)
            a, b = old["train"], old["validation"]
            self.assertFalse((train_seeds | validation_seeds) & (a | b))

    def test_missing_arm_and_accidental_width_change_are_rejected(self):
        spec = study.design("formal")
        spec["optimizer_arms"].pop()
        with self.assertRaisesRegex(ValueError, "registered optimizer arms"):
            study.cells(spec)
        for key in ((1024, 1.0, 7), (512, 0.5, 7), (2048, 0.25, 7), (2048, True, 7), (2048, 0.5, 8)):
            with self.assertRaises(ValueError):
                study.cell_name(*key)

    def test_historical_evidence_change_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "study").mkdir()
            (root / "study/study.json").write_text('{"status": "completed"}')
            with self.assertRaisesRegex(ValueError, "changed after audit"):
                study.historical_source(root)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for actual parameter policy partition")
class PairingTests(unittest.TestCase):
    def test_identical_init_across_optimizer_arms_is_required_not_deduplicated(self):
        spec = study.design("formal")
        records = records_fixture(spec)
        study.validate_pairing(records, spec)
        self.assertEqual(len({r["initial_parameters_sha256"] for r in records.values()}), 4)
        altered = copy.deepcopy(records)
        altered[study.cell_name(2048, 0.5, 7)]["initial_parameters_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "changed initial model"):
            study.validate_pairing(altered, spec)
        altered = copy.deepcopy(records)
        altered[study.cell_name(2048, 0.5, 7)]["metrics"]["event_counts_seen"]["snapshots"] += 1
        with self.assertRaisesRegex(ValueError, "observation cadence"):
            study.validate_pairing(altered, spec)

    def test_policy_mismatch_is_not_recorded_as_success(self):
        spec = study.design("development")
        key = (2048, 0.5, 7)
        evidence = evidence_fixture(spec, (2048, 7))
        evidence[0].update(width_lr_scale=0.5, optimizer_policy=study.expected_optimizer_policy(2048, 1.0))
        with self.assertRaisesRegex(ValueError, "parameter partition"):
            study.validate_training_evidence(*evidence, spec, key, "cuda")

    def test_paired_contrast_sign_retains_worse_initialization(self):
        spec = study.design("formal")
        records = records_fixture(spec)
        for key, value in {
            (2048, 1.0, 7): 0.5,
            (2048, 1.0, 11): 0.4,
            (2048, 0.5, 7): 0.3,
            (2048, 0.5, 11): 0.5,
        }.items():
            records[study.cell_name(*key)]["metrics"]["best_validation_rmse_m"] = value
        contrast = study.summarize(records, spec)["contrasts"]["best_validation_rmse_m"][
            "a2048_half_minus_base"
        ]
        self.assertAlmostEqual(contrast["by_initialization"]["7"], -0.2)
        self.assertAlmostEqual(contrast["by_initialization"]["11"], 0.1)
        self.assertAlmostEqual(contrast["mean_over_initializations"], -0.05)

    def test_training_failure_preserves_started_and_unstarted_cells(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "failure-evidence"
            with (
                patch.object(study, "source_revision", return_value="9" * 40),
                patch("isac_wm.training.train", side_effect=RuntimeError("injected training failure")),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    study.run(out, "cpu", "development")
            report = json.loads((out / "study.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"]["stage"], "training")
            self.assertEqual(len(report["cells"]), 6)
            self.assertEqual(report["cells"][study.cell_name(512, 1.0, 7)]["status"], "failed")
            self.assertEqual(sum(c["status"] == "not_started" for c in report["cells"].values()), 5)
            self.assertFalse(report["fresh_test_consumed"])


if __name__ == "__main__":
    unittest.main()
