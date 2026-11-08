"""Registered Phase B budgets and evidence semantics; no trained-result fixtures."""

import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vision_learning_study as phase_a
from recurrent_state_study import (
    LEARNING_RATE,
    NORM_EPS,
    VISION_WEIGHT,
    cell_name,
    cells,
    design,
    registered_model,
    registered_seeds,
    summarize,
    validate_selected_checkpoint,
    validate_training_evidence,
    validation_updates,
)
from test_vision_learning_study import evidence_fixture as phase_a_evidence_fixture


def evidence_fixture(spec, key, device="cuda"):
    """Synthetic records for validation tests, never actual training evidence."""
    norm, initialization = key
    evidence = phase_a_evidence_fixture(spec, (LEARNING_RATE, VISION_WEIGHT, initialization), device)
    evidence[0]["model"] = registered_model(norm)
    for snapshot in evidence[3]["snapshots"]:
        post = 0.5 if norm == "none" else 0.9
        snapshot["validation"]["network"] = {
            "hidden_pre_norm_rms": 0.5 if norm == "none" else 2.0,
            "hidden_post_norm_rms": post,
            "hidden_rms": post,
        }
    return evidence


def selected_checkpoint_fixture(manifest, spec, key):
    """Metadata-only synthetic checkpoint; no tensor/model execution claim."""
    norm, initialization = key
    return {
        "schema_version": 2,
        "model_config": registered_model(norm),
        "scene_config": spec["scene"],
        "seed": initialization,
        "best_update": manifest["best_update"],
        "best_epoch": math.ceil(manifest["best_update"] / (spec["training_episodes"] / spec["batch_size"])),
        "state_dict": {"synthetic_placeholder": "metadata-only fixture"},
        **{
            field: copy.deepcopy(manifest[field])
            for field in (
                "train_episode_seeds",
                "validation_episode_seeds",
                "validation_episodes",
                "validation_every_updates",
                "vision_loss_weight",
                "vision_supervision_protocol",
                "initial_visual_parameters_sha256",
                "diagnostic_protocol",
                "seed_plan",
            )
        },
    }


class RecurrentStateStudyTests(unittest.TestCase):
    def test_formal_four_cells_preserve_phase_a_data_and_budgets(self):
        spec, reference = design("formal"), phase_a.design("formal")
        self.assertEqual(cells(spec), (("none", 7), ("none", 11), ("layernorm", 7), ("layernorm", 11)))
        for field in (
            "training_episodes",
            "train_seed_base",
            "validation_episodes",
            "validation_seed_base",
            "batch_size",
            "optimizer_updates",
            "validation_every_updates",
            "scene",
            "model",
            "parameters",
            "visual_encoder_parameters",
            "initializations",
        ):
            self.assertEqual(spec[field], reference[field], field)
        self.assertEqual(validation_updates(spec), [96, 192, 288, 384])
        self.assertEqual(spec["diagnostic_updates"], sorted(reference["diagnostic_updates"] + [288]))
        self.assertEqual(spec["learning_rate"], 3e-4)
        self.assertEqual(spec["vision_loss_weight"], 1.0)
        self.assertEqual(spec["intervention"]["eps"], NORM_EPS)
        self.assertFalse(spec["intervention"]["elementwise_affine"])
        self.assertNotIn("test_seed_base", spec)
        self.assertIn("no fresh test", spec["test_policy"])
        self.assertEqual(registered_model("none"), phase_a.MODEL)
        self.assertEqual(registered_model("layernorm"), phase_a.MODEL | {"latent_state_norm": "layernorm"})

    def test_development_uses_full_model_and_disjoint_data(self):
        spec, formal = design("development"), design("formal")
        self.assertEqual(spec["model"], formal["model"])
        self.assertEqual((spec["batch_size"], spec["steps"], spec["optimizer_updates"]), (8, 64, 8))
        self.assertEqual(validation_updates(spec), [2, 4, 6, 8])
        pools, formal_pools = registered_seeds(spec), registered_seeds(formal)
        for left in pools.values():
            for right in formal_pools.values():
                self.assertFalse(left.intersection(right))
        for profile in ("formal", "development"):
            current = design(profile)
            for key in cells(current):
                device = "cuda" if profile == "formal" else "cpu"
                metrics = validate_training_evidence(
                    *evidence_fixture(current, key, device), current, key, device
                )
                self.assertEqual(metrics["best_update"], current["optimizer_updates"])
                self.assertEqual(metrics["final_minus_best_validation_rmse_m"], 0)
                self.assertLess(metrics["late_validation_change_m"], 0)

    def test_wrong_model_or_missing_late_diagnostic_is_rejected(self):
        spec, key = design("formal"), ("layernorm", 11)
        evidence = evidence_fixture(spec, key)
        evidence[0]["model"] = registered_model("none")
        with self.assertRaisesRegex(ValueError, "model"):
            validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, key)
        evidence[3]["snapshots"] = [s for s in evidence[3]["snapshots"] if s["optimizer_updates"] != 288]
        with self.assertRaisesRegex(ValueError, "diagnostic snapshot"):
            validate_training_evidence(*evidence, spec, key, "cuda")

    def test_hidden_diagnostic_contract_is_enforced(self):
        spec = design("formal")
        for norm, field, value in (
            ("none", "hidden_pre_norm_rms", 0.4),
            ("layernorm", "hidden_pre_norm_rms", float("nan")),
            ("layernorm", "hidden_post_norm_rms", float("inf")),
            ("layernorm", "hidden_rms", 0.7),
        ):
            key = (norm, 7)
            evidence = evidence_fixture(spec, key)
            evidence[3]["snapshots"][-1]["validation"]["network"][field] = value
            with self.subTest(norm=norm, field=field), self.assertRaises(ValueError):
                validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, ("layernorm", 7))
        last = evidence[3]["snapshots"][-1]["validation"]
        last["network"].update(hidden_post_norm_rms=1.1, hidden_rms=1.1)
        with self.assertRaisesRegex(ValueError, "normalization bound"):
            validate_training_evidence(*evidence, spec, ("layernorm", 7), "cuda")
        evidence = evidence_fixture(spec, ("none", 7))
        evidence[3]["snapshots"][-1]["validation"]["event_counts"]["vision_new_valid"] += 1
        with self.assertRaisesRegex(ValueError, "cadence"):
            validate_training_evidence(*evidence, spec, ("none", 7), "cuda")

    def test_best_and_endpoint_metrics_keep_late_regression_visible(self):
        spec, key = design("formal"), ("layernorm", 11)
        evidence = evidence_fixture(spec, key)
        for row, value in zip(evidence[1], (0.5, 0.4, 0.49, 1.15)):
            row["validation_position_rmse_m"] = value
        evidence[0].update(best_update=192, best_validation_rmse_m=0.4)
        metrics = validate_training_evidence(*evidence, spec, key, "cuda")
        self.assertEqual(metrics["best_update"], 192)
        self.assertEqual(metrics["last_validation_rmse_m"], 1.15)
        self.assertAlmostEqual(metrics["final_minus_best_validation_rmse_m"], 0.75)
        self.assertAlmostEqual(metrics["late_validation_change_m"], 0.75)

    def test_paired_summary_retains_both_initializations_and_endpoints(self):
        spec = design("formal")
        records = {}
        for key in cells(spec):
            metrics = validate_training_evidence(*evidence_fixture(spec, key), spec, key, "cuda")
            metrics.update(
                best_validation_rmse_m=0.3,
                last_validation_rmse_m=0.4,
                final_minus_best_validation_rmse_m=0.1,
                late_validation_change_m=-0.2,
            )
            records[cell_name(*key)] = {"status": "completed", "metrics": metrics}
        records[cell_name("layernorm", 11)]["metrics"].update(
            last_validation_rmse_m=1.2, final_minus_best_validation_rmse_m=0.9, late_validation_change_m=0.6
        )
        summary = summarize(records, spec)
        self.assertEqual(len(summary["cells"]), 4)
        contrast = summary["contrasts"]["last_validation_rmse_m"]
        self.assertAlmostEqual(contrast["layernorm_minus_none_init_11"], 0.8)
        self.assertAlmostEqual(contrast["layernorm_minus_none_mean_over_inits"], 0.4)
        invalid = copy.deepcopy(records)
        invalid[cell_name("none", 7)]["metrics"]["last_validation_rmse_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            summarize(invalid, spec)
        records.pop(cell_name("layernorm", 11))
        with self.assertRaisesRegex(ValueError, "Every registered"):
            summarize(records, spec)

    def test_checkpoint_metadata_binds_selected_update_and_intervention(self):
        spec, key = design("formal"), ("layernorm", 7)
        manifest = evidence_fixture(spec, key)[0]
        payload = selected_checkpoint_fixture(manifest, spec, key)
        validate_selected_checkpoint(payload, manifest, spec, key)
        for field, value in (
            ("best_update", 96),
            ("best_epoch", 1),
            ("model_config", registered_model("none")),
            ("seed", 11),
            ("state_dict", {}),
            ("vision_loss_weight", 0),
            ("train_episode_seeds", [0]),
            ("diagnostic_protocol", {}),
        ):
            altered = copy.deepcopy(payload)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_selected_checkpoint(altered, manifest, spec, key)

    def test_unregistered_profiles_and_conditions_fail(self):
        with self.assertRaises(ValueError):
            design("smoke")
        with self.assertRaises(ValueError):
            cell_name("batchnorm", 7)
        with self.assertRaises(ValueError):
            cell_name("none", 99)


if __name__ == "__main__":
    unittest.main()
