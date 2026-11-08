"""Protocol and evidence checks do not require Torch or a GPU."""

import copy
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vision_learning_study import (
    MODEL,
    PARAMETERS,
    SUPERVISION_PROTOCOL,
    VISUAL_PARAMETERS,
    cell_name,
    cells,
    design,
    epoch_budget,
    registered_seeds,
    summarize,
    validate_training_evidence,
    validation_updates,
)


def evidence_fixture(spec, key, device="cuda"):
    """A synthetic record exercises validation; it is not a trained-model result."""
    lr, weight, initialization = key
    updates = validation_updates(spec)
    epochs = epoch_budget(spec)
    pools = registered_seeds(spec)
    diagnostic_protocol = {
        "updates": spec["diagnostic_updates"],
        "validation_episode_seeds": sorted(pools["validation"])[:4],
        "max_validation_episodes": 4,
        "scope": "diagnostic-only; not checkpoint selection",
        "validation_steps": 64,
        "saturation_threshold": 0.99,
    }
    manifest = {
        "seed": initialization,
        "episodes": spec["training_episodes"],
        "epochs": epochs,
        "completed_epochs": epochs,
        "batch_size": spec["batch_size"],
        "parameters": PARAMETERS,
        "visual_encoder_parameters": VISUAL_PARAMETERS,
        "model": MODEL.copy(),
        "learning_rate": lr,
        "vision_loss_weight": weight,
        "early_stopping_patience": 0,
        "optimizer_updates": spec["optimizer_updates"],
        "validation_updates": updates,
        "validation_opportunities": 4,
        "validation_episodes": spec["validation_episodes"],
        "validation_every_updates": spec["validation_every_updates"],
        "resume_source": None,
        "model_selection_status": "selected",
        "train_episode_seeds": sorted(pools["train"]),
        "validation_episode_seeds": sorted(pools["validation"]),
        "seed_plan": {
            "initialization": initialization,
            "train_seed_base": spec["train_seed_base"],
            "validation_seed_base": spec["validation_seed_base"],
            "batch_order_seed": 37,
        },
        "runtime": {"device": device},
        "scene": spec["scene"].copy(),
        "best_update": updates[-1],
        "best_validation_rmse_m": 1 / updates[-1],
        "training_seconds": 8.0,
        "initial_parameters_sha256": hashlib.sha256(f"initial-{initialization}".encode()).hexdigest(),
        "initial_visual_parameters_sha256": hashlib.sha256(f"visual-{initialization}".encode()).hexdigest(),
        "vision_supervision_protocol": SUPERVISION_PROTOCOL.copy(),
        "event_counts_seen": {
            "snapshots": spec["training_episodes"] * epochs * spec["steps"],
            "vision_new_valid": 64,
            "radio_new_valid": 96,
            "evidence_update_ticks": 100,
            "prediction_only_ticks": spec["training_episodes"] * epochs * spec["steps"] - 100,
            "vision_visible_captures": 40,
        },
        "diagnostic_protocol": diagnostic_protocol,
    }
    history = [{"optimizer_updates": u, "validation_position_rmse_m": 1 / u} for u in updates]
    epoch_history = [
        {"epoch": e, "optimizer_updates": e * spec["training_episodes"] // spec["batch_size"]}
        for e in range(1, epochs + 1)
    ]
    diagnostics = {
        "schema_version": 1,
        "protocol": diagnostic_protocol,
        "snapshots": [
            {
                "optimizer_updates": u,
                "validation": {
                    "visual_position_rmse_m": 1 / (u + 1),
                    "visual_visible_capture_count": 40,
                    "event_counts": {"snapshots": 256, "vision_new_valid": 80},
                },
            }
            for u in spec["diagnostic_updates"]
        ],
    }
    return manifest, history, epoch_history, diagnostics


class VisionLearningStudyTests(unittest.TestCase):
    def test_full_factorial_and_identical_update_selection_budgets(self):
        formal, smoke = design("formal"), design("smoke")
        for profile, spec in (("formal", formal), ("smoke", smoke)):
            self.assertEqual(len(cells(spec)), 8)
            self.assertEqual(len(set(cell_name(*k) for k in cells(spec))), 8)
            self.assertEqual(len(validation_updates(spec)), 4)
            self.assertEqual(
                epoch_budget(spec) * spec["training_episodes"] // spec["batch_size"],
                spec["optimizer_updates"],
            )
            self.assertEqual(spec["model"]["hidden_dim"], 512)
            self.assertEqual(spec["visual_encoder_parameters"], 1230)
            self.assertNotIn("test_seed_base", spec)
            self.assertIn("no fresh test", spec["test_policy"])
            for key in cells(spec):
                metrics = validate_training_evidence(
                    *evidence_fixture(spec, key, "cuda" if profile == "formal" else "cpu"),
                    spec,
                    key,
                    "cuda" if profile == "formal" else "cpu",
                )
                self.assertEqual(metrics["best_update"], spec["optimizer_updates"])
        self.assertEqual(epoch_budget(formal), 2)
        self.assertEqual(epoch_budget(smoke), 4)
        self.assertEqual(formal["diagnostic_updates"], [0, 1, 2, 4, 8, 16, 32, 64, 96, 192, 384])
        self.assertEqual(smoke["diagnostic_updates"], [0, 1, 2, 4, 8])
        self.assertEqual(14 * 512**2 + 410 * 512 + 3862, PARAMETERS)

    def test_development_does_not_consume_formal_or_previous_test_trajectories(self):
        formal, smoke = registered_seeds(design("formal")), registered_seeds(design("smoke"))
        development = set().union(*smoke.values())
        self.assertFalse(development & set().union(*formal.values()))
        self.assertEqual(len(formal["train"]), 1536)
        self.assertEqual(len(formal["validation"]), 192)
        self.assertEqual(min(formal["train"]), 700000)
        self.assertEqual(min(formal["validation"]), 74611000)
        for base in (81700000, 81710000, 81720000, 93621000, 94761000, 95871000):
            self.assertFalse(development.intersection(range(base, base + 192)))
        altered = design("formal") | {"validation_seed_base": 700000}
        with self.assertRaisesRegex(ValueError, "disjoint"):
            registered_seeds(altered)

    def test_missing_mismatched_or_repeated_training_evidence_is_rejected(self):
        spec, key = design("formal"), cells(design("formal"))[0]
        cases = [
            ("learning_rate", 0.01),
            ("vision_loss_weight", 2.0),
            ("parameters", 172246),
            ("visual_encoder_parameters", 1231),
            ("resume_source", "old.pt"),
            ("validation_opportunities", 3),
            ("optimizer_updates", 383),
            ("train_episode_seeds", [1]),
            ("initial_parameters_sha256", "unknown"),
            ("event_counts_seen", None),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                evidence = evidence_fixture(spec, key)
                evidence[0][field] = value
                with self.assertRaises(ValueError):
                    validate_training_evidence(*evidence, spec, key, "cuda")
        for index in (1, 2):
            evidence = evidence_fixture(spec, key)
            evidence[index].pop()
            with self.assertRaises(ValueError):
                validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, key)
        evidence[3]["snapshots"][-1] = copy.deepcopy(evidence[3]["snapshots"][0])
        with self.assertRaisesRegex(ValueError, "diagnostic snapshot"):
            validate_training_evidence(*evidence, spec, key, "cuda")

    def test_bad_initializations_remain_in_the_descriptive_summary(self):
        spec = design("formal")
        records = {}
        for lr, weight, seed in cells(spec):
            value = 1 + 2 * (lr == 0.0003) + 3 * weight + 4 * (seed == 11) + 5 * weight * (lr == 0.0003)
            records[cell_name(lr, weight, seed)] = {
                "status": "completed",
                "metrics": dict.fromkeys(
                    ("best_validation_rmse_m", "last_validation_rmse_m", "last_visual_position_rmse_m"), value
                ),
            }
        summary = summarize(records, spec)
        self.assertEqual(len(summary["cells"]), 8)
        contrasts = summary["contrasts"]["best_validation_rmse_m"]
        self.assertEqual(contrasts["vision_1_minus_0_lr_0.001_init_11"], 3)
        self.assertEqual(contrasts["learning_rate_by_supervision_interaction_mean_over_inits"], 5)
        self.assertIn("No population confidence", summary["interpretation"])
        records.pop(cell_name(0.0003, 1.0, 11))
        with self.assertRaisesRegex(ValueError, "Every registered"):
            summarize(records, spec)


if __name__ == "__main__":
    unittest.main()
