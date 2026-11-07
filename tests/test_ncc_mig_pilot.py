"""NCC development evidence must retain the formal working set and real updates."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ncc_mig_pilot as pilot
from vision_learning_study import design, registered_seeds

ROOT = Path(__file__).resolve().parents[1]
PIN = "d" * 40


def fixture(directory, weight=1.0):
    directory.mkdir()
    manifest = {
        "parameters": pilot.PARAMETERS,
        "visual_encoder_parameters": pilot.VISUAL_PARAMETERS,
        "model": pilot.MODEL,
        "scene": pilot.SCENE,
        "epochs": 2,
        "completed_epochs": 2,
        "episodes": 16,
        "batch_size": 8,
        "seed": 97,
        "learning_rate": 0.001,
        "early_stopping_patience": 0,
        "optimizer_updates": 4,
        "optimizer_updates_this_call": 4,
        "validation_updates": [2, 4],
        "validation_opportunities": 2,
        "validation_episodes": 8,
        "vision_loss_weight": weight,
        "resume_source": None,
        "model_selection_status": "selected",
        "diagnostic_observed_updates": [0, 1, 2, 4],
        "train_episode_seeds": list(range(pilot.DEVELOPMENT_TRAIN_BASE, pilot.DEVELOPMENT_TRAIN_BASE + 16)),
        "validation_episode_seeds": list(
            range(pilot.DEVELOPMENT_VALIDATION_BASE, pilot.DEVELOPMENT_VALIDATION_BASE + 8)
        ),
        "runtime": {"device": "cuda"},
        "best_validation_rmse_m": 0.4,
        "training_seconds": 3.0,
        "event_counts_seen": {"snapshots": 2048},
        "initial_parameters_sha256": "a" * 64,
        "initial_visual_parameters_sha256": "b" * 64,
        "cuda_peak_allocated_bytes": 1024,
        "cuda_peak_reserved_bytes": 2048,
        "cuda_memory_scope": "training/validation/checkpoint loop",
    }
    history = [
        {"optimizer_updates": n, "train_loss": -0.2, "validation_position_rmse_m": 0.4} for n in (2, 4)
    ]
    (directory / "history.json").write_text(json.dumps(history))
    branch = {"gradient_norm": 1.0, "clipped_gradient_norm": 0.5, "update_norm": 0.001}
    snapshots = [{"optimizer_updates": 0, "training": None}] + [
        {"optimizer_updates": n, "training": {"branches": {"visual": branch}}} for n in (1, 2, 4)
    ]
    (directory / "visual_diagnostics.json").write_text(json.dumps({"snapshots": snapshots}))
    for name in ("model.pt", "resume.pt"):
        # Only checksum handling is exercised here; the compatibility run writes real torch archives.
        (directory / name).write_bytes(b"fixture checkpoint")
        (directory / (name + ".sha256")).write_text(hashlib.sha256(b"fixture checkpoint").hexdigest())
    return manifest


class TestNccMigPilot(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)

    def test_working_set_matches_formal_but_seeds_do_not(self):
        formal = design("formal")
        args = pilot.train_arguments(1.0)
        self.assertEqual(pilot.MODEL, formal["model"])
        self.assertEqual(pilot.SCENE, formal["scene"])
        self.assertEqual(args["batch_size"], formal["batch_size"])
        development = set(range(args["train_seed_base"], args["train_seed_base"] + args["episodes"]))
        development |= set(range(args["validation_seed_base"], args["validation_seed_base"] + 8))
        for seeds in registered_seeds(formal).values():
            self.assertFalse(development & seeds)
        self.assertEqual(args["epochs"] * args["episodes"] // args["batch_size"], 4)

    def test_negative_finite_loss_and_real_updates_are_valid(self):
        directory = self.tmp / "cell"
        manifest = fixture(directory)
        result = pilot.validate_cell(directory, manifest, 1.0)
        self.assertEqual(result["optimizer_updates"], 4)
        self.assertEqual(len(result["checkpoint_sha256"]), 2)

    def test_reduced_batch_and_cpu_cannot_pass_cuda_pilot(self):
        directory = self.tmp / "cell"
        manifest = fixture(directory)
        changed = copy.deepcopy(manifest)
        changed["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "batch_size"):
            pilot.validate_cell(directory, changed, 1.0)
        manifest["runtime"]["device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "device"):
            pilot.validate_cell(directory, manifest, 1.0)

    def test_fake_optimization_and_changed_checkpoint_are_rejected(self):
        directory = self.tmp / "cell"
        manifest = fixture(directory)
        path = directory / "visual_diagnostics.json"
        diagnostics = json.loads(path.read_text())
        diagnostics["snapshots"][1]["training"]["branches"]["visual"]["update_norm"] = 0
        path.write_text(json.dumps(diagnostics))
        with self.assertRaisesRegex(ValueError, "No parameters"):
            pilot.validate_cell(directory, manifest, 1.0)
        diagnostics["snapshots"][1]["training"]["branches"]["visual"]["update_norm"] = 0.01
        path.write_text(json.dumps(diagnostics))
        (directory / "model.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
            pilot.validate_cell(directory, manifest, 1.0)

    def test_wrong_job_preflight_fails_with_evidence_and_no_training(self):
        preflight = self.tmp / "preflight.json"
        preflight.write_text(json.dumps({"status": "passed", "git_commit": PIN, "slurm": {"job_id": "12"}}))
        out = self.tmp / "pilot"
        with patch.dict(os.environ, {"SLURM_JOB_ID": "13"}):
            with self.assertRaisesRegex(ValueError, "another job"):
                pilot.run(out, preflight, PIN)
        report = json.loads((out / "pilot.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cells"], [])
        with self.assertRaises(FileExistsError):
            pilot.run(out, preflight, PIN)

    def test_gpu_wrapper_preserves_failed_stage_and_refuses_overwrite(self):
        root = self.tmp / "checkout"
        root.mkdir()
        runtime = self.tmp / "runtime"
        (runtime / "venv/bin").mkdir(parents=True)
        fake_python = runtime / "venv/bin/python"
        fake_python.write_text("#!/bin/bash\nif [[ $1 == -I ]]; then cat >/dev/null; exit 0; fi\nexit 17\n")
        fake_python.chmod(0o700)
        (runtime / "runtime.json").write_text("{}")
        env = os.environ | {
            "SLURM_JOB_ID": "14",
            "ISAC_NCC_SETUP_JOB_ID": "13",
            "ISAC_NCC_PIN": PIN,
            "ISAC_PROJECT_ROOT": str(root),
            "ISAC_RUNS_DIR": str(root / "runs"),
            "ISAC_NCC_RUNTIME_ROOT": str(runtime),
        }
        env.pop("SLURM_ARRAY_JOB_ID", None)
        script = ROOT / "scripts/ncc_mig_pilot.sh"
        first = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        self.assertEqual(first.returncode, 17, first.stderr)
        status_path = root / "runs/ncc-mig-14/status.json"
        evidence = status_path.read_bytes()
        self.assertEqual(json.loads(evidence)["stage"], "preflight")
        self.assertEqual(json.loads(evidence)["status"], "failed")
        second = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(status_path.read_bytes(), evidence)


if __name__ == "__main__":
    unittest.main()
