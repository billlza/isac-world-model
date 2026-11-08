"""The correction observer cannot change inference or confuse capture and delivery time."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from correction_diagnostics import (
    capture_state_for_evaluation,
    replay_with_diagnostics,
    summarize_events,
)
from recovery_challenge_core import observed_trace
from score_scale_recovery import apply_network

from isac_wm.simulator import SceneConfig


class CaptureTimeTests(unittest.TestCase):
    def test_capture_truth_is_not_later_delivery_truth(self):
        truth = np.zeros((16, 4))
        truth[:, 0] = np.arange(16)
        captured = capture_state_for_evaluation(truth, 0.4, 0.1)
        self.assertEqual(captured[0], 4)
        self.assertNotEqual(captured[0], truth[6, 0])
        captured[0] = -100
        self.assertEqual(truth[4, 0], 4)
        for capture in (-0.1, 0.45, 1.6, float("nan")):
            with self.subTest(capture=capture), self.assertRaises(ValueError):
                capture_state_for_evaluation(truth, capture, 0.1)

    def test_no_fresh_events_is_missing_evidence_not_zero_error(self):
        summary = summarize_events([])
        for panel in summary.values():
            for result in panel.values():
                self.assertEqual(result, {"event_count": 0, "metrics": None})


class CorrectionObserverTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Installed PyTorch required for real correction replay")
        self.torch = torch
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def make_model(self, width=96):
        from isac_wm.structured import StructuredConfig, StructuredEventModel

        torch = self.torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            return StructuredEventModel(StructuredConfig(hidden_dim=width)).eval()

    def test_observer_preserves_both_widths_and_detaches_without_aliases(self):
        from isac_wm.artifacts import model_fingerprint

        torch = self.torch
        for width in (96, 512):
            with self.subTest(width=width):
                model = self.make_model(width)
                fingerprint = model_fingerprint(model)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(23)
                    image = torch.rand(3, 3, 24, 24)
                # Unavailable/stale data may be non-finite; it must never be encoded
                # or applied as a new measurement, including on the observer path.
                image[2] = float("nan")
                radio = torch.tensor([[0.4, 0.6, 0.8, 0.1]]).expand(3, -1).clone()
                radio[1:] = float("nan")
                inputs = {
                    "image": image,
                    "radio": radio,
                    "age": torch.tensor([[0.2, 0.1], [0.4, 0.3], [0.0, 0.0]]),
                    "valid": torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
                    "new": torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
                    "dt": torch.full((3, 1), 0.1),
                }
                original = model.initial(3, "cpu")
                untouched = tuple(v.clone() for v in original)
                plain, plain_memory = model.step(inputs, original)
                observed = {}
                actual, actual_memory = model.step(inputs, original, diagnostics=observed)
                for key in plain:
                    torch.testing.assert_close(actual[key], plain[key], rtol=0, atol=0)
                for a, b in zip(actual_memory, plain_memory, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(original, untouched, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(observed["schema_version"], 1)
                for modality in ("vision", "radio"):
                    d = observed[modality]
                    for value in d.values():
                        self.assertFalse(value.requires_grad)
                        self.assertIsNone(value.grad_fn)
                    torch.testing.assert_close(
                        d["innovation_xy"], d["measurement_xy"] - d["predicted_measurement_xy"]
                    )
                    expected = (d["gain"] @ d["innovation_xy"].unsqueeze(-1)).squeeze(-1)
                    expected = torch.where(d["active"], expected, torch.zeros_like(expected))
                    torch.testing.assert_close(d["state_mean_after"], d["state_mean_before"] + expected)
                    torch.testing.assert_close(d["applied_state_correction"][1:], torch.zeros(2, 4))
                    self.assertEqual(d["active"][:, 0].tolist(), [True, False, False])
                vision = observed["vision"]
                torch.testing.assert_close(
                    vision["measurement_variance_xy"],
                    vision["learned_measurement_variance_xy"]
                    + 0.05 * vision["heatmap_entropy"].clamp(0, 1).pow(4),
                )
                # Mutating a diagnostic copy cannot mutate the caller's returned
                # mean, retained feature, memory or model parameters.
                before = actual["mean"].detach().clone()
                vision["state_mean_after"].zero_()
                vision["measurement_xy"].zero_()
                torch.testing.assert_close(actual["mean"], before, rtol=0, atol=0)
                torch.testing.assert_close(actual_memory[1], plain_memory[1], rtol=0, atol=0)
                self.assertEqual(fingerprint, model_fingerprint(model))
                with self.assertRaisesRegex(ValueError, "empty caller-owned"):
                    model.step(inputs, original, diagnostics=observed)

    def test_replay_is_identical_and_scores_measurement_at_actual_capture(self):
        from isac_wm.artifacts import model_fingerprint

        model = self.make_model()
        cfg = SceneConfig(steps=64)
        snapshots, deliveries, truth, meta = observed_trace(83_004_500, cfg, True)
        anchor = meta["first_post_blackout_camera_delivery_tick"]
        fingerprint = model_fingerprint(model)
        reference, reference_future = apply_network(model, cfg, "cpu", snapshots)
        xy, future, result = replay_with_diagnostics(model, cfg, "cpu", snapshots, deliveries, truth, anchor)
        np.testing.assert_array_equal(xy, reference)
        np.testing.assert_array_equal(future, reference_future)
        self.assertEqual(fingerprint, result["parameter_sha256_before_and_after"])
        self.assertEqual(fingerprint, model_fingerprint(model))
        events = result["events"]
        recovered = next(e for e in events if e["modality"] == "vision" and e["delivery_tick"] == anchor)
        self.assertAlmostEqual(recovered["capture_s"], 3.2)
        self.assertAlmostEqual(recovered["available_s"], 3.4)
        self.assertAlmostEqual(recovered["estimated_state_time_s"], 3.4)
        self.assertAlmostEqual(recovered["decision_time_s"], 3.4)
        self.assertAlmostEqual(recovered["age_s"], 0.2)
        np.testing.assert_array_equal(recovered["capture_truth_xy_m"], truth[32, :2])
        expected = np.linalg.norm(np.asarray(recovered["measurement_xy_m"]) - truth[32, :2])
        self.assertAlmostEqual(recovered["measurement_error_at_capture_m"], expected)
        for index, modality in enumerate(("vision", "radio")):
            counts = result["snapshot_counts"][modality]
            fresh = sum(bool(s["valid"][index] and s["new"][index]) for s in snapshots)
            self.assertEqual(counts["new_valid"], fresh)
            self.assertEqual(sum(counts.values()), cfg.steps)
            self.assertEqual(len([e for e in events if e["modality"] == modality]), fresh)
            self.assertEqual(result["summary"]["all_fresh"][modality]["event_count"], fresh)
        for event in events:
            innovation = np.asarray(event["innovation_xy_m"])
            np.testing.assert_allclose(
                np.asarray(event["position_gain"]) @ innovation,
                event["applied_position_correction_m"],
                rtol=1e-4,
                atol=5e-7,
            )
            np.testing.assert_allclose(
                np.asarray(event["velocity_gain_s_inv"]) @ innovation,
                event["applied_velocity_correction_mps"],
                rtol=1e-4,
                atol=5e-7,
            )
        # Different scoring labels cannot change even one prediction. Only the
        # post-replay diagnostic evaluation fields may change.
        changed_truth = truth.copy()
        changed_truth[:, :2] += 1.0
        changed_xy, changed_future, changed = replay_with_diagnostics(
            model, cfg, "cpu", snapshots, deliveries, changed_truth, anchor
        )
        np.testing.assert_array_equal(changed_xy, xy)
        np.testing.assert_array_equal(changed_future, future)
        self.assertNotEqual(
            changed["events"][0]["measurement_error_at_capture_m"],
            events[0]["measurement_error_at_capture_m"],
        )

    def test_snapshot_packet_mismatch_fails_before_replay(self):
        model = self.make_model()
        cfg = SceneConfig(steps=64)
        snapshots, deliveries, truth, meta = observed_trace(83_004_500, cfg, False)
        snapshots[0]["new"][0] = 1.0
        with self.assertRaisesRegex(ValueError, "reproduce"):
            replay_with_diagnostics(
                model,
                cfg,
                "cpu",
                snapshots,
                deliveries,
                truth,
                meta["first_post_blackout_camera_delivery_tick"],
            )
