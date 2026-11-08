"""Runtime diagnosis stays bounded, uses development data and preserves model code."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_profile", ROOT / "scripts/profile_training_runtime.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class RuntimeProfileContractTests(unittest.TestCase):
    def test_default_budget_and_dedicated_development_seeds(self):
        spec = PROFILE.plan()
        self.assertEqual(spec["total_optimizer_updates_per_variant"], 13)
        seeds = spec["development_episode_seeds"]
        self.assertEqual(len(seeds), 104)
        self.assertEqual(len(set(seeds)), 104)
        self.assertEqual(seeds[0], 4_080_000_000)
        self.assertLess(seeds[-1], 2**32)
        self.assertEqual(spec["formal_train_validation_test_seeds_consumed"], [])

    def test_invalid_or_unbounded_work_rejected_before_torch_import(self):
        for kwargs in (
            {"measured_updates": 9},
            {"measured_updates": True},
            {"profile_updates": 4},
            {"warmup_updates": 0},
            {"batch_size": 9},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PROFILE.plan(**kwargs)
        with self.assertRaisesRegex(ValueError, "explicitly"):
            PROFILE.run_profile("unused", device_name="auto")

    def test_operator_candidates_do_not_sum_nested_calls(self):
        class Event:
            key = "aten::_local_scalar_dense"
            count = 8
            self_cpu_time_total = 100.0
            cpu_time_total = 100.0
            self_device_time_total = 0.0
            device_time_total = 0.0

        class Profiler:
            def key_averages(self):
                return [Event()]

        result = PROFILE.operator_statistics(Profiler())
        self.assertEqual(result["synchronization_candidates"][0]["calls"], 8)
        self.assertIn("not proof", result["interpretation"])

    def test_requested_cuda_rejects_trace_with_only_cpu_time(self):
        cpu_only = {"top_self_device": [{"self_device_us": 0.0}]}
        with self.assertRaisesRegex(RuntimeError, "no positive device time"):
            PROFILE.require_device_activity(cpu_only, "cuda")
        PROFILE.require_device_activity(cpu_only, "cpu")
        PROFILE.require_device_activity({"top_self_device": [{"self_device_us": 1.0}]}, "cuda")


@unittest.skipIf(importlib.util.find_spec("torch") is None, "Installed PyTorch is required")
class RuntimeProfileTorchTests(unittest.TestCase):
    def test_real_cpu_updates_share_corpus_and_have_operator_evidence(self):
        import torch

        from isac_wm.simulator import SceneConfig, recorded_episode

        previous_threads = torch.get_num_threads()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile"
            with patch("isac_wm.simulator.recorded_episode", wraps=recorded_episode) as generator:
                result = PROFILE.run_profile(
                    output,
                    device_name="cpu",
                    warmup_updates=1,
                    measured_updates=1,
                    profile_updates=1,
                    batch_size=1,
                    threads=1,
                    hidden_dims=(8, 16),
                    scene=SceneConfig(steps=8, image_size=16),
                )
            # Corpus created once, shared across two actual optimizing variants.
            self.assertEqual(generator.call_count, 3)
            self.assertEqual(
                [call.args[0] for call in generator.call_args_list],
                result["plan"]["development_episode_seeds"],
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["cpu_minibatch_sha256"]), 3)
            self.assertEqual(len(result["variants"]), 2)
            for row in result["variants"]:
                self.assertEqual(row["counts"]["optimizer_updates"], 3)
                self.assertEqual(row["counts"]["forward_calls"], 3)
                self.assertEqual(row["counts"]["episode_time_steps"], 24)
                self.assertEqual(len(row["measured"]["update_seconds"]), 1)
                self.assertGreater(row["measured"]["total_seconds"], 0)
                self.assertGreater(row["operators"]["operator_groups"], 0)
                scopes = {
                    event["operator"]: event["calls"] for event in row["operators"]["instrumented_scopes"]
                }
                self.assertEqual(scopes["diagnostic.backward"], 1)
                self.assertEqual(scopes["diagnostic.clip_and_step"], 1)
                self.assertIsNone(row["unprofiled_cuda_peak_allocated_bytes"])
            self.assertEqual(json.loads((output / "profile.json").read_text())["status"], "completed")
            self.assertFalse(list(output.glob("*.pt")))
            with self.assertRaises(FileExistsError):
                PROFILE.run_profile(output, device_name="cpu")
        self.assertEqual(torch.get_num_threads(), previous_threads)

    def test_cuda_request_never_falls_back_to_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unavailable"
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                    PROFILE.run_profile(output, device_name="cuda")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
