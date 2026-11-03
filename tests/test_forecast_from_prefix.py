"""Information boundary, direct forecast equivalence, and exclusive output creation."""

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from isac_wm.simulator import SceneConfig, ToyScene

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("forecast_from_prefix", SCRIPTS / "forecast_from_prefix.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class PrefixTests(unittest.TestCase):
    def test_a_packet_released_after_cutoff_cannot_change_any_input(self):
        cfg = SceneConfig(steps=32, dropout=0)
        cutoff_tick = 7

        def traces(marker):
            class GuardedScene(ToyScene):
                def __init__(self, seed, config):
                    if config.steps != cutoff_tick + 1:
                        raise AssertionError("The simulation generated a future suffix")
                    super().__init__(seed, config)

                def camera(self, step):
                    if step >= cutoff_tick:
                        raise AssertionError("Acquisition requested at or beyond cutoff")
                    return super().camera(step)

                def radio(self, step, action):
                    if step >= cutoff_tick:
                        raise AssertionError("Acquisition requested at or beyond cutoff")
                    packet = super().radio(step, action)
                    if step == cutoff_tick - 1:
                        return replace(packet, available_s=1.0, payload=np.full(4, marker))
                    return packet

                def truth_for_evaluation(self):
                    raise AssertionError("Prediction cannot access truth")

            with patch.object(demo, "ToyScene", GuardedScene):
                return list(demo.synthetic_prefix(cfg, 55, cutoff_tick + 1))

        a, b = traces(-1000), traces(1000)
        self.assertEqual(len(a), 8)
        self.assertAlmostEqual(a[-1][0], 0.7)
        self.assertEqual(a[-1][1]["new"][1], 0)
        self.assertGreater(sum(s["new"][1] for _, s in a), 0)
        for (_, left), (_, right) in zip(a, b, strict=True):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])

    def test_output_never_replaces_a_file_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "demo.json"
            with self.assertRaises(ValueError):
                demo.write_new(p, {"x": float("nan")})
            self.assertFalse(p.exists())
            demo.write_new(p, {"x": 1})
            with self.assertRaises(FileExistsError):
                demo.write_new(p, {"x": 2})
            self.assertEqual(json.loads(p.read_text()), {"x": 1})

    def test_limits_and_training_horizon(self):
        for bad in (7, 257):
            with self.assertRaises(ValueError):
                list(demo.synthetic_prefix(SceneConfig(), 0, bad))
        self.assertEqual(demo.horizon_scope(0.3, 3 * 0.1), "direct training horizon")
        self.assertIn("extrapolation", demo.horizon_scope(1.0, 3 * 0.1))
        for bad in (float("nan"), float("inf"), -1, 0):
            with self.assertRaises(ValueError):
                demo.horizon_scope(bad, 0.3)


class CheckpointTests(unittest.TestCase):
    def test_report_matches_direct_calls_from_same_cutoff_belief(self):
        import torch

        from isac_wm.artifacts import save_checkpoint
        from isac_wm.structured import StructuredConfig, StructuredEventModel
        from isac_wm.training import INPUT_KEYS

        # A small random checkpoint tests plumbing and units, not forecast accuracy.
        cfg = SceneConfig(steps=16, image_size=16, dropout=0)
        model = StructuredEventModel(StructuredConfig(8, 8, "structured-attention", 4, 2)).eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            save_checkpoint(path, {
                "state_dict": model.state_dict(),
                "model_config": model.config.to_dict(),
                "scene_config": cfg.to_dict(),
            })
            report = demo.run(path, prefix_steps=8, seed=55, radio_queries=True)
        self.assertEqual(report["runtime"]["device"], "cpu")
        with torch.inference_mode():
            memory = model.initial(1, "cpu")
            for _, snapshot in demo.synthetic_prefix(cfg, 55, 8):
                x = {k: torch.from_numpy(snapshot[k])[None] for k in INPUT_KEYS}
                pred, memory = model.step(x, memory)
            hidden = pred["hidden"]
            for row, seconds in zip(report["forecasts"], (0.1, 0.3, 1.0), strict=True):
                self.assertEqual(row["horizon_s"], seconds)
                ref = model.forecast(hidden, seconds)
                np.testing.assert_allclose(row["mean"], ref["mean"][0].numpy() * cfg.state_scale, rtol=1e-5)
                cov = ref["covariance"][0].numpy() * np.outer(cfg.state_scale, cfg.state_scale)
                np.testing.assert_allclose(row["covariance"], cov, rtol=1e-5)
                np.testing.assert_allclose(row["variance"], cov.diagonal(), rtol=1e-5)
                self.assertTrue((np.array(row["variance"]) > 0).all())
            for code, name in ((1, "quick"), (2, "precise")):
                ref = model.predict_observation(hidden, torch.tensor([code]))["mean"][0].numpy()
                expected = ref * [cfg.room_m * 2**0.5, 1, 1, 2]
                np.testing.assert_allclose(report["hypothetical_rf_at_cutoff"][name], expected, rtol=1e-5)
