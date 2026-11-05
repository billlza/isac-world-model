"""Capture identity, masked supervision and the unchanged online input contract."""

import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from isac_wm.events import EventQueue, Observation, SnapshotBuffer
from isac_wm.simulator import SceneConfig, ToyScene, recorded_episode


class CaptureTimeLabelTests(unittest.TestCase):
    def test_targets_follow_capture_identity_not_delivery_time_or_visibility(self):
        config = SceneConfig(
            steps=48,
            camera_period=2,
            camera_delay_s=0.35,
            delay_jitter_s=0.13,
            camera_dropout=0,
            occlusion_fraction=0.25,
        )
        scene = ToyScene(512, config)
        episode = recorded_episode(512, config)
        truth = scene.truth_for_evaluation()
        fresh = (episode["new"][:, 0] * episode["valid"][:, 0]).astype(bool)
        masks = episode["vision_target_mask"].astype(bool)
        self.assertGreater(fresh.sum(), masks.sum())
        self.assertGreater(masks.sum(), 0)
        shifted_visibility = 0
        for tick in np.flatnonzero(fresh):
            uid = int(episode["vision_event_uid"][tick])
            capture_tick = uid // 3
            captured = capture_tick * config.dt
            event = scene.camera(capture_tick)
            self.assertEqual(uid, event.uid)
            self.assertEqual(episode["vision_capture_s"][tick], captured)
            np.testing.assert_array_equal(episode["image"][tick], event.payload)
            self.assertLessEqual(event.available_s, tick * config.dt + 1e-10)
            self.assertAlmostEqual(episode["age"][tick, 0], tick * config.dt - captured, places=6)
            visible = (
                not scene.occlusion_start <= capture_tick < scene.occlusion_start + scene.occlusion_length
            )
            current_visible = (
                not scene.occlusion_start <= tick < scene.occlusion_start + scene.occlusion_length
            )
            shifted_visibility += int(visible != current_visible)
            self.assertEqual(masks[tick], visible)
            if visible:
                np.testing.assert_array_equal(
                    episode["vision_capture_target_xy"][tick], truth[capture_tick, :2] / config.room_m
                )
                self.assertFalse(
                    np.array_equal(episode["vision_capture_target_xy"][tick], truth[tick, :2] / config.room_m)
                )
            else:
                self.assertEqual(episode["valid"][tick, 0], 1)
        self.assertGreater(shifted_visibility, 0)
        self.assertTrue(np.all(episode["vision_event_uid"][~fresh] == -1))
        self.assertTrue(np.all(episode["vision_capture_s"][~fresh] == -1))
        self.assertTrue(np.all(episode["vision_capture_target_xy"][~masks] == 0))
        self.assertFalse(masks[~fresh].any())
        self.assertEqual(len(np.unique(episode["vision_event_uid"][fresh])), int(fresh.sum()))

    def test_older_out_of_order_packet_is_not_relabelled_as_new(self):
        config = SceneConfig(steps=16, camera_period=2, camera_dropout=0, occlusion_fraction=0)
        original = ToyScene.camera

        def reordered(scene, step):
            event = original(scene, step)
            if event is None:
                return None
            available = 0.95 if step == 0 else event.capture_s + 0.05
            return Observation(event.uid, event.modality, event.capture_s, available, event.payload)

        with patch.object(ToyScene, "camera", reordered):
            episode = recorded_episode(913, config)
        accepted = episode["vision_event_uid"][episode["vision_event_uid"] >= 0]
        self.assertNotIn(0, accepted)
        self.assertTrue(np.all(np.diff(accepted) > 0))
        self.assertEqual(int(episode["vision_target_mask"].sum()), len(accepted))

    def test_missing_camera_has_no_localization_label(self):
        episode = recorded_episode(21, SceneConfig(steps=16, camera_dropout=1))
        self.assertEqual(episode["vision_target_mask"].sum(), 0)
        self.assertTrue(np.all(episode["vision_event_uid"] == -1))
        self.assertTrue(np.all(episode["vision_capture_target_xy"] == 0))

    def test_legacy_trace_and_six_online_input_keys_are_unchanged(self):
        config, seed = SceneConfig(steps=24), 65
        scene, queue, buffer = ToyScene(seed, config), EventQueue(), SnapshotBuffer(config.image_size)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 33]))
        snapshots, actions = [], []
        for tick in range(config.steps):
            snapshots.append(buffer.update(queue.release(tick * config.dt), tick * config.dt, config.dt))
            action = int(rng.choice([0, 1, 2], p=[0.30, 0.45, 0.25]))
            actions.append(action)
            for event in (scene.camera(tick), scene.radio(tick, action)):
                if event is not None:
                    queue.push(event)
        episode = recorded_episode(seed, config)
        for key in snapshots[0]:
            np.testing.assert_array_equal(episode[key], np.stack([row[key] for row in snapshots]))
        np.testing.assert_array_equal(episode["action_target"], actions)
        path = Path(__file__).resolve().parents[1] / "src/isac_wm/training.py"
        tree = ast.parse(path.read_text())
        online = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "INPUT_KEYS" for target in node.targets)
        )
        self.assertEqual(online, ("image", "radio", "age", "valid", "new", "dt"))
        self.assertFalse(any(name.startswith("vision_") for name in online))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "Installed PyTorch is required")
class VisualLossTests(unittest.TestCase):
    def setUp(self):
        import torch

        from isac_wm.model import training_loss, visual_supervision_loss
        from isac_wm.structured import StructuredConfig, build_model
        from isac_wm.training import batchify

        self.torch, self.training_loss, self.visual_loss = torch, training_loss, visual_supervision_loss
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = build_model(StructuredConfig(hidden_dim=8))
        self.batch = batchify(
            [
                recorded_episode(
                    100,
                    SceneConfig(
                        steps=8, image_size=16, camera_period=1, camera_dropout=0, occlusion_fraction=0
                    ),
                )
            ],
            torch.device("cpu"),
        )

    def tearDown(self):
        self.torch.set_num_threads(self.threads)

    def test_auxiliary_outputs_preserve_model_values_and_ignore_all_truth_inputs(self):
        torch = self.torch
        ordinary = self.model(self.batch, batched_encoding=True)
        observed = self.model(self.batch, batched_encoding=True, return_visual=True)
        self.assertEqual(sum(p.numel() for p in self.model.visual.parameters()), 1230)
        for key, value in ordinary.items():
            torch.testing.assert_close(observed[key], value, rtol=0, atol=0)
        changed = {key: value.clone() for key, value in self.batch.items()}
        for key in (
            "state_target",
            "radio_target",
            "vision_capture_target_xy",
            "vision_target_mask",
            "vision_capture_s",
            "vision_event_uid",
        ):
            changed[key].fill_(123)
        replay = self.model(changed, batched_encoding=True, return_visual=True)
        for key in observed:
            torch.testing.assert_close(replay[key], observed[key], rtol=0, atol=0)

    def test_default_and_zero_weight_are_exactly_original_objective(self):
        torch = self.torch
        from isac_wm.model import gaussian_nll

        output = self.model(self.batch, batched_encoding=True, return_visual=True)
        target = self.batch["state_target"]
        current = gaussian_nll(output["mean"], output["logvar"], target).mean()
        future = gaussian_nll(
            output["future_mean"][:, :-3], output["future_logvar"][:, :-3], target[:, 3:]
        ).mean()
        mask = self.batch["radio_target_valid"]
        radio = gaussian_nll(output["obs_mean"], output["obs_logvar"], self.batch["radio_target"])
        observation = (radio * mask).sum() / mask.sum().clamp_min(1)
        reference = current + 0.5 * future + 0.15 * observation
        default, details = self.training_loss(output, self.batch)
        zero, zero_details = self.training_loss(output, self.batch, vision_loss_weight=0.0)
        torch.testing.assert_close(default, reference, rtol=0, atol=0)
        torch.testing.assert_close(zero, default, rtol=0, atol=0)
        self.assertEqual(details, zero_details)
        parameters = tuple(self.model.parameters())
        gradients = torch.autograd.grad(default, parameters, retain_graph=True, allow_unused=True)
        reference_gradients = torch.autograd.grad(reference, parameters, retain_graph=True, allow_unused=True)
        for a, b in zip(gradients, reference_gradients, strict=True):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_direct_localization_gradient_reaches_spatial_encoder(self):
        output = self.model(self.batch, batched_encoding=True, return_visual=True)
        direct, count = self.visual_loss(output, self.batch)
        self.assertGreater(count, 0)
        direct.backward()
        self.assertGreater(float(self.model.visual.heat.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.visual.trunk[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.fusion.weight_ih.grad)
        self.assertIsNone(self.model.noise.weight.grad)

    def test_normalized_loss_empty_support_and_bad_input_rejection(self):
        torch = self.torch
        predicted = torch.tensor([[[0.2, 0.4], [0.1, 0.1]]], requires_grad=True)
        sample = {
            "vision_capture_target_xy": torch.tensor([[[0.4, 0.0], [0.9, 0.9]]]),
            "vision_target_mask": torch.tensor([[1.0, 0.0]]),
            "new": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            "valid": torch.ones(1, 2, 2),
        }
        loss, count = self.visual_loss({"visual_xy": predicted}, sample)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(loss), 0.1, places=6)
        sample["vision_target_mask"].zero_()
        empty, count = self.visual_loss({"visual_xy": predicted}, sample)
        self.assertEqual(count, 0)
        self.assertEqual(float(empty), 0)
        empty.backward()
        self.assertEqual(float(predicted.grad.abs().sum()), 0)
        sample["vision_target_mask"][0, 1] = 1
        with self.assertRaisesRegex(ValueError, "newly delivered"):
            self.visual_loss({"visual_xy": predicted}, sample)
        for value in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.training_loss({}, {}, vision_loss_weight=value)


if __name__ == "__main__":
    unittest.main()
