"""Phase C registration and evidence tests; synthetic fixtures are not model results."""

import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recurrent_state_study as phase_b
from normalized_capacity_study import (
    INITIALIZATIONS,
    LEARNING_RATE,
    NORM_EPS,
    PARAMETERS_BY_WIDTH,
    VISION_WEIGHT,
    VISUAL_PARAMETERS,
    WIDTHS,
    cell_name,
    cells,
    design,
    epoch_budget,
    numerical_settings,
    registered_model,
    registered_seeds,
    run,
    summarize,
    validate_checkpoint_weights,
    validate_numerical_settings,
    validate_pairing,
    validate_selected_checkpoint,
    validate_training_evidence,
    validation_updates,
)
from test_vision_learning_study import evidence_fixture as phase_a_evidence_fixture


def numerical_fixture(device="cuda"):
    """Schema-only execution settings for validator tests, never measured runtime."""
    return {
        "deterministic_algorithms": False,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_enabled": True,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": None,
        "nvidia_tf32_override": None,
        "cuda_runtime": "12.8" if device == "cuda" else None,
        "cudnn_version": 91002 if device == "cuda" else None,
    }


def counts_fixture(snapshots):
    return {
        "snapshots": snapshots,
        "vision_new_valid": 80,
        "radio_new_valid": 96,
        "evidence_update_ticks": 120,
        "prediction_only_ticks": snapshots - 120,
        "vision_visible_captures": 40,
    }


def evidence_fixture(spec, key, device="cuda"):
    """Synthetic schema fixture; values do not represent a trained model."""
    width, initialization = key
    evidence = phase_a_evidence_fixture(spec, (LEARNING_RATE, VISION_WEIGHT, initialization), device)
    manifest, history, epoch_history, diagnostics = evidence
    manifest.update(
        model=registered_model(width),
        parameters=PARAMETERS_BY_WIDTH[width],
        initialization="random; no pretrained weights",
        initial_parameters_sha256=hashlib.sha256(f"synthetic-{width}-{initialization}".encode()).hexdigest(),
        planned_optimizer_updates=spec["optimizer_updates"],
        planned_validation_opportunities=4,
        validation_mode="fixed_optimizer_updates",
        diagnostic_observed_updates=spec["diagnostic_updates"],
        optimizer_updates_this_call=spec["optimizer_updates"],
        completed_epochs_this_call=epoch_budget(spec),
        dataset_estimated_bytes=1_000_000,
        diagnostic_overhead_seconds=1.0,
        training_timing_scope="synthetic timing scope for validator tests",
    )
    if device == "cuda":
        manifest.update(
            cuda_peak_allocated_bytes=PARAMETERS_BY_WIDTH[width] * 16,
            cuda_peak_reserved_bytes=PARAMETERS_BY_WIDTH[width] * 20,
            cuda_memory_scope="synthetic allocator scope for validator tests",
        )
    for rows in (history, epoch_history):
        for row in rows:
            row["train_loss"] = -0.5
    for snapshot in diagnostics["snapshots"]:
        snapshot["validation"]["network"] = {
            "hidden_pre_norm_rms": 2.0,
            "hidden_post_norm_rms": 0.9,
            "hidden_rms": 0.9,
        }
        snapshot["validation"]["event_counts"] = counts_fixture(
            min(spec["validation_episodes"], 4) * spec["steps"]
        )
        if snapshot["optimizer_updates"] == 0:
            snapshot["training"] = None
        else:
            snapshot["training"] = {
                "event_counts": counts_fixture(spec["batch_size"] * spec["steps"]),
                "losses": {"total": -0.5},
                "branches": {
                    "synthetic_complete_parameter_group": {
                        "parameters": PARAMETERS_BY_WIDTH[width],
                        "parameter_norm": 1.0,
                        "gradient_norm": 1.5,
                        "clipped_gradient_norm": 0.8,
                        "update_norm": 0.01,
                        "relative_update": 0.01,
                    },
                },
            }
    return evidence


def cell_records_fixture(spec, device="cuda"):
    """Complete synthetic cell records for pairing, packaging and summary tests."""
    records = {}
    for key in cells(spec):
        evidence = evidence_fixture(spec, key, device)
        manifest = evidence[0]
        records[cell_name(*key)] = {
            "status": "completed",
            "hidden_width": key[0],
            "initialization": key[1],
            "latent_state_norm": "layernorm",
            "learning_rate": LEARNING_RATE,
            "vision_loss_weight": VISION_WEIGHT,
            "parameters": PARAMETERS_BY_WIDTH[key[0]],
            "visual_encoder_parameters": VISUAL_PARAMETERS,
            "initial_parameters_sha256": manifest["initial_parameters_sha256"],
            "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
            "numerical_settings": numerical_fixture(device),
            "metrics": validate_training_evidence(*evidence, spec, key, device),
        }
    return records


def selected_checkpoint_fixture(manifest, spec, key):
    """Metadata fixture; real tensor checks are separate and require Torch."""
    return {
        "schema_version": 2,
        "model_config": registered_model(key[0]),
        "scene_config": spec["scene"],
        "seed": key[1],
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


class NormalizedCapacityStudyTests(unittest.TestCase):
    def test_six_cells_preserve_phase_b_budget_and_normalization(self):
        spec, reference = design("formal"), phase_b.design("formal")
        self.assertEqual(cells(spec), ((512, 7), (512, 11), (1024, 7), (1024, 11), (2048, 7), (2048, 11)))
        self.assertEqual(len({cell_name(*key) for key in cells(spec)}), 6)
        for field in (
            "training_episodes",
            "train_seed_base",
            "validation_episodes",
            "validation_seed_base",
            "batch_size",
            "optimizer_updates",
            "validation_every_updates",
            "scene",
            "visual_encoder_parameters",
            "initializations",
            "minibatch_order_seed",
            "diagnostic_updates",
        ):
            self.assertEqual(spec[field], reference[field], field)
        self.assertEqual(validation_updates(spec), [96, 192, 288, 384])
        self.assertEqual(epoch_budget(spec), 2)
        self.assertEqual(spec["learning_rate"], 3e-4)
        self.assertEqual(spec["vision_loss_weight"], 1.0)
        self.assertEqual(spec["normalization"]["eps"], NORM_EPS)
        self.assertFalse(spec["normalization"]["elementwise_affine"])
        self.assertNotIn("test_seed_base", spec)
        self.assertIn("no fresh test", spec["test_policy"])
        for width in WIDTHS:
            self.assertEqual(PARAMETERS_BY_WIDTH[width], 14 * width**2 + 410 * width + 3862)
            self.assertEqual(registered_model(width)["latent_state_norm"], "layernorm")
        self.assertEqual(spec["primary_contrasts"], ["1024_minus_512", "2048_minus_512"])

    def test_development_executes_every_width_with_disjoint_data_and_four_selections(self):
        development, formal = design("development"), design("formal")
        self.assertEqual(development["models_by_width"], formal["models_by_width"])
        self.assertEqual(cells(development), cells(formal))
        self.assertEqual(
            (development["training_episodes"], development["batch_size"], development["optimizer_updates"]),
            (8, 8, 4),
        )
        self.assertEqual(epoch_budget(development), 4)
        self.assertEqual(validation_updates(development), [1, 2, 3, 4])
        self.assertEqual(development["diagnostic_updates"], [0, 1, 2, 3, 4])
        development_seeds = set().union(*registered_seeds(development).values())
        for spec in (formal, phase_b.design("development")):
            self.assertFalse(development_seeds & set().union(*registered_seeds(spec).values()))
        for profile, device in (("formal", "cuda"), ("development", "cpu")):
            spec = design(profile)
            for key in cells(spec):
                metrics = validate_training_evidence(*evidence_fixture(spec, key, device), spec, key, device)
                self.assertEqual(metrics["best_update"], spec["optimizer_updates"])
                self.assertEqual(metrics["final_minus_best_validation_rmse_m"], 0)
                self.assertEqual(metrics["hidden_width"], key[0])
        with self.assertRaisesRegex(ValueError, "All registered"):
            cells(development | {"hidden_widths": [512]})

    def test_wrong_capacity_budget_or_resume_is_rejected(self):
        spec, key = design("formal"), (2048, 11)
        for field, value in (
            ("parameters", PARAMETERS_BY_WIDTH[512]),
            ("model", registered_model(1024)),
            ("visual_encoder_parameters", 1231),
            ("optimizer_updates", 383),
            ("planned_optimizer_updates", 385),
            ("validation_opportunities", 3),
            ("completed_epochs_this_call", 1),
            ("resume_source", "old.pt"),
            ("initialization", "pretrained"),
            ("validation_episode_seeds", [89140000]),
        ):
            evidence = evidence_fixture(spec, key)
            evidence[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_evidence(*evidence, spec, key, "cuda")

    def test_finite_hidden_gradient_and_cadence_checks_reject_corruption(self):
        spec, key = design("formal"), (1024, 7)
        for field, value in (
            ("hidden_pre_norm_rms", float("nan")),
            ("hidden_post_norm_rms", float("inf")),
            ("hidden_rms", 0.8),
        ):
            evidence = evidence_fixture(spec, key)
            evidence[3]["snapshots"][-1]["validation"]["network"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, key)
        evidence[3]["snapshots"][-1]["validation"]["network"].update(hidden_rms=1.1, hidden_post_norm_rms=1.1)
        with self.assertRaisesRegex(ValueError, "normalization bound"):
            validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, key)
        evidence[3]["snapshots"][-1]["training"]["branches"]["synthetic_complete_parameter_group"][
            "gradient_norm"
        ] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, key)
        evidence[3]["snapshots"][-1]["validation"]["event_counts"]["radio_new_valid"] += 1
        with self.assertRaisesRegex(ValueError, "cadence"):
            validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, key)
        evidence[3]["snapshots"][-1]["training"] = None
        with self.assertRaisesRegex(ValueError, "optimizer-step"):
            validate_training_evidence(*evidence, spec, key, "cuda")
        evidence = evidence_fixture(spec, key)
        evidence[3]["snapshots"].pop(-2)
        with self.assertRaisesRegex(ValueError, "diagnostic snapshot"):
            validate_training_evidence(*evidence, spec, key, "cuda")

    def test_resources_distinguish_allocator_memory_from_cpu_and_accept_negative_loss(self):
        spec, key = design("formal"), (512, 7)
        evidence = evidence_fixture(spec, key)
        metrics = validate_training_evidence(*evidence, spec, key, "cuda")
        self.assertGreater(metrics["resources"]["cuda_peak_reserved_bytes"], 0)
        for field, value in (
            ("cuda_peak_allocated_bytes", None),
            ("cuda_peak_reserved_bytes", 1),
            ("training_seconds", float("inf")),
        ):
            altered = copy.deepcopy(evidence)
            altered[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_evidence(*altered, spec, key, "cuda")
        development = design("development")
        cpu = evidence_fixture(development, key, "cpu")
        cpu[0]["cuda_peak_allocated_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "CPU development"):
            validate_training_evidence(*cpu, development, key, "cpu")

    def test_visual_hash_pairing_does_not_require_identical_full_models(self):
        spec = design("formal")
        records = cell_records_fixture(spec)
        validate_pairing(records, spec)
        self.assertEqual(len({row["initial_parameters_sha256"] for row in records.values()}), 6)
        for seed in INITIALIZATIONS:
            self.assertEqual(
                len(
                    {records[cell_name(width, seed)]["initial_visual_parameters_sha256"] for width in WIDTHS}
                ),
                1,
            )
        for field in ("initial_visual_parameters_sha256", "initial_parameters_sha256"):
            altered = copy.deepcopy(records)
            altered[cell_name(2048, 11)][field] = records[cell_name(512, 7)][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_pairing(altered, spec)
        altered = copy.deepcopy(records)
        altered[cell_name(1024, 7)]["metrics"]["event_counts_seen"]["radio_new_valid"] += 1
        with self.assertRaisesRegex(ValueError, "cadence"):
            validate_pairing(altered, spec)
        altered = copy.deepcopy(records)
        altered[cell_name(2048, 11)]["numerical_settings"]["cudnn_benchmark"] = True
        with self.assertRaisesRegex(ValueError, "Numerical settings"):
            validate_pairing(altered, spec)

    def test_capacity_contrasts_keep_seed_specific_late_regression(self):
        spec = design("formal")
        records = cell_records_fixture(spec)
        for key in cells(spec):
            records[cell_name(*key)]["metrics"].update(
                best_validation_rmse_m=0.3,
                last_validation_rmse_m=0.4,
                final_minus_best_validation_rmse_m=0.1,
                late_validation_change_m=-0.2,
            )
        records[cell_name(2048, 11)]["metrics"].update(
            last_validation_rmse_m=1.2,
            final_minus_best_validation_rmse_m=0.9,
            late_validation_change_m=0.6,
        )
        result = summarize(records, spec)
        contrast = result["contrasts"]["last_validation_rmse_m"]
        self.assertAlmostEqual(contrast["1024_minus_512_mean_over_inits"], 0)
        self.assertAlmostEqual(contrast["2048_minus_512_init_7"], 0)
        self.assertAlmostEqual(contrast["2048_minus_512_init_11"], 0.8)
        self.assertAlmostEqual(contrast["2048_minus_512_mean_over_inits"], 0.4)
        self.assertEqual(result["resources_by_width"]["2048"]["parameters"], PARAMETERS_BY_WIDTH[2048])
        records.pop(cell_name(1024, 11))
        with self.assertRaisesRegex(ValueError, "Every registered"):
            summarize(records, spec)

    def test_checkpoint_metadata_binds_selected_width_update_and_initialization(self):
        spec, key = design("formal"), (2048, 11)
        manifest = evidence_fixture(spec, key)[0]
        payload = selected_checkpoint_fixture(manifest, spec, key)
        validate_selected_checkpoint(payload, manifest, spec, key)
        for field, value in (
            ("model_config", registered_model(512)),
            ("seed", 7),
            ("best_update", 96),
            ("best_epoch", 1),
            ("state_dict", {}),
            ("vision_loss_weight", 0),
            ("validation_episode_seeds", [0]),
            ("initial_visual_parameters_sha256", "0" * 64),
        ):
            altered = copy.deepcopy(payload)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_selected_checkpoint(altered, manifest, spec, key)

    def test_numerical_settings_require_actual_fields_not_reproducibility_claim(self):
        for device in ("cpu", "cuda"):
            fixture = numerical_fixture(device)
            validate_numerical_settings(fixture)
            for field in tuple(fixture):
                invalid = fixture.copy()
                del invalid[field]
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_numerical_settings(invalid)
        with self.assertRaises(ValueError):
            validate_numerical_settings(numerical_fixture() | {"deterministic_algorithms": "true"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is required for real tensor validation")
class NormalizedCapacityTensorTests(unittest.TestCase):
    def test_training_failure_retains_failed_and_unstarted_cells_without_reusing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "failed-study"
            with patch("normalized_capacity_study.source_revision", return_value="9" * 40):
                with patch("isac_wm.training.train", side_effect=RuntimeError("injected training failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected training failure"):
                        run(out, "cpu", "development")
                saved = json.loads((out / "study.json").read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(len(saved["cells"]), 6)
                self.assertEqual(saved["cells"][cell_name(512, 7)]["status"], "failed")
                self.assertEqual(saved["cells"][cell_name(512, 7)]["error"]["type"], "RuntimeError")
                self.assertTrue(
                    all(
                        saved["cells"][cell_name(*key)]["status"] == "not_started"
                        for key in cells(design("development"))[1:]
                    )
                )
                self.assertFalse(saved["fresh_test_consumed"])
                with self.assertRaises(FileExistsError):
                    run(out, "cpu", "development")
                self.assertEqual(json.loads((out / "study.json").read_text()), saved)

    def test_all_registered_widths_have_exact_parameter_and_normalization_budget(self):
        import torch

        from isac_wm.structured import StructuredConfig, build_model

        for width in WIDTHS:
            with torch.device("meta"):
                model = build_model(StructuredConfig(**registered_model(width)))
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()), PARAMETERS_BY_WIDTH[width]
            )
            self.assertEqual(
                sum(parameter.numel() for parameter in model.visual.parameters()), VISUAL_PARAMETERS
            )
            self.assertFalse(model.latent_state_norm.elementwise_affine)
            self.assertEqual(model.latent_state_norm.eps, NORM_EPS)

    def test_real_checkpoint_tensors_reject_wrong_shape_dtype_and_nonfinite_values(self):
        import torch

        from isac_wm.structured import StructuredConfig, build_model

        model = build_model(StructuredConfig(**registered_model(512)))
        state = model.state_dict()
        payload = {"state_dict": state}
        validate_checkpoint_weights(payload, (512, 7))
        name = next(iter(state))
        original = state[name]
        for replacement in (original.flatten(), original.double(), torch.full_like(original, float("nan"))):
            state[name] = replacement
            with self.assertRaisesRegex(ValueError, "invalid shape, dtype or non-finite"):
                validate_checkpoint_weights(payload, (512, 7))
        state[name] = original
        state.pop(name)
        with self.assertRaisesRegex(ValueError, "weight keys"):
            validate_checkpoint_weights(payload, (512, 7))

    def test_numerical_settings_are_observations_without_mutation(self):
        import torch

        before = numerical_settings(torch)
        self.assertEqual(numerical_settings(torch), before)
        validate_numerical_settings(before)


if __name__ == "__main__":
    unittest.main()
