"""Standard-library tests of rejection/evidence logic; no GPU validation claim."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hamilton_preflight", ROOT / "scripts/hamilton_preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)
GPU_UUID = "GPU-12345678-1234-5678-1234-567812345678"


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name) / "preflight.json"
        self.env = {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "cuda", "CUDA_VISIBLE_DEVICES": "3"}
        self.slurm = (
            "JobId=123 JobState=RUNNING Partition=cuda "
            "AllocTRES=cpu=16,gres/gpu=1,gres/gpu:h200_nvl=1 TresPerNode=gres/gpu:h200_nvl:1"
        )
        self.mig_mode = "Disabled"
        self.torch = SimpleNamespace(
            __version__="test-build",
            version=SimpleNamespace(cuda="test-cuda"),
            backends=SimpleNamespace(cuda=SimpleNamespace(is_built=Mock(return_value=True))),
            cuda=SimpleNamespace(
                is_available=Mock(return_value=True),
                device_count=Mock(return_value=1),
                get_device_properties=Mock(
                    return_value=SimpleNamespace(
                        name="NVIDIA H200 NVL", total_memory=140 * 1024**3, uuid=GPU_UUID
                    )
                ),
                get_device_capability=Mock(return_value=(9, 0)),
            ),
        )
        self.loader = Mock(return_value=self.torch)

    def runner(self, command, **kwargs):
        if command[0] == "scontrol":
            output = self.slurm
        elif command[0] == "git":
            output = "a" * 40
        elif command[0] == "nvidia-smi":
            output = f"{GPU_UUID}, NVIDIA H200 NVL, test-driver, 143360, {self.mig_mode}"
        else:
            self.fail(f"Unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def run_check(self):
        return preflight.run_preflight(self.out, env=self.env, runner=self.runner, torch_loader=self.loader)

    def assert_failed(self, message):
        with self.assertRaisesRegex(preflight.PreflightError, message):
            self.run_check()
        record = json.loads(self.out.read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_type"], "PreflightError")
        self.assertNotIn("cuda_smoke", record)

    def test_login_node_rejected_before_importing_torch(self):
        self.env.pop("SLURM_JOB_ID")
        self.assert_failed("Slurm job")
        self.loader.assert_not_called()

    def test_non_cuda_partition_rejected(self):
        self.env["SLURM_JOB_PARTITION"] = "shared"
        self.assert_failed("cuda partition")

    def test_missing_gpu_visibility_rejected(self):
        self.env.pop("CUDA_VISIBLE_DEVICES")
        self.assert_failed("CUDA_VISIBLE_DEVICES")

    def test_default_mig_request_rejected_before_importing_torch(self):
        self.slurm = (
            "JobId=123 JobState=RUNNING Partition=cuda "
            "AllocTRES=gres/gpu=1,gres/gpu:h200_nvl_1g.18gb=1 "
            "TresPerNode=gres/gpu:h200_nvl_1g.18gb:1"
        )
        self.assert_failed("Whole-H200 allocation")
        self.loader.assert_not_called()

    def test_cpu_wheel_rejected_even_if_device_availability_were_true(self):
        self.torch.version.cuda = None
        self.torch.backends.cuda.is_built.return_value = False
        self.assert_failed("CPU-only PyTorch")
        self.torch.cuda.get_device_properties.assert_not_called()

    def test_cuda_unavailable_rejected(self):
        self.torch.cuda.is_available.return_value = False
        self.assert_failed("CUDA is unavailable")

    def test_multiple_visible_devices_rejected(self):
        self.torch.cuda.device_count.return_value = 2
        self.assert_failed("Exactly one")

    def test_small_device_rejected_despite_full_card_request(self):
        self.torch.cuda.get_device_properties.return_value.total_memory = 18 * 1024**3
        self.assert_failed("Expected a whole H200")

    def test_missing_uuid_rejected(self):
        del self.torch.cuda.get_device_properties.return_value.uuid
        self.assert_failed("device UUID")

    def test_mig_mode_enabled_rejected(self):
        self.mig_mode = "Enabled"
        self.assert_failed("MIG disabled")

    def test_cuda_kernel_failure_is_recorded_and_reraised(self):
        with patch.object(preflight, "_cuda_smoke", side_effect=RuntimeError("CUDA kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "CUDA kernel failed"):
                self.run_check()
        record = json.loads(self.out.read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_type"], "RuntimeError")
        self.assertEqual(record["gpu"]["uuid"], GPU_UUID)

    def test_success_records_actual_observations_with_smoke_explicitly_mocked(self):
        smoke = {"optimizer_steps": 2, "scope": "mocked in a unit test, not GPU evidence"}
        with patch.object(preflight, "_cuda_smoke", return_value=smoke) as check:
            record = self.run_check()
        check.assert_called_once_with(self.torch)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["runtime"]["torch_cuda"], "test-cuda")
        self.assertEqual(record["slurm"]["cuda_visible_devices"], "3")
        self.assertEqual(record["gpu"]["total_memory_bytes"], 140 * 1024**3)
        self.assertEqual(record["gpu"]["uuid"], GPU_UUID)
        self.assertEqual(json.loads(self.out.read_text()), record)

    def test_existing_evidence_is_never_overwritten(self):
        self.out.write_text("previous evidence\n")
        with self.assertRaises(FileExistsError):
            self.run_check()
        self.assertEqual(self.out.read_text(), "previous evidence\n")
        self.loader.assert_not_called()

    def test_slurm_command_failure_is_not_swallowed(self):
        runner = Mock(side_effect=subprocess.CalledProcessError(1, ["scontrol"]))
        with self.assertRaises(subprocess.CalledProcessError):
            preflight.run_preflight(self.out, env=self.env, runner=runner, torch_loader=self.loader)
        self.assertEqual(json.loads(self.out.read_text())["status"], "failed")
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
