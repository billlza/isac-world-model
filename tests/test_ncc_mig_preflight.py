"""Allocation, identity, and failure-evidence checks with GPU execution mocked."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ncc_mig_preflight", ROOT / "scripts/ncc_mig_preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)
PIN = "a" * 40
PARENT = "GPU-11111111-2222-3333-4444-555555555555"
MIG = "MIG-12345678-1234-5678-1234-567812345678"
SIBLING = "MIG-87654321-4321-8765-4321-876543218765"
OTHER_PARENT = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class NccMigPreflightTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.out = self.directory / "preflight.json"
        self.env = {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_PARTITION": "res-gpu-small",
            "SLURM_JOB_ACCOUNT": "ncc_users",
            "SLURMD_NODENAME": "gpu10",
            # Preserve Slurm's value; don't require UUID spelling in the env.
            "CUDA_VISIBLE_DEVICES": "0",
        }
        self.slurm = (
            f"JobId=123 UserId=bhmq03({os.getuid()}) "
            "JobState=RUNNING Partition=res-gpu-small Account=ncc_users QOS=debug "
            "NumNodes=1 NodeList=gpu10 "
            "ReqTRES=cpu=2,mem=8G,gres/gpu=1,gres/gpu:1g.10gb=1 "
            "AllocTRES=cpu=2,mem=8G,gres/gpu=1,gres/gpu:1g.10gb=1 "
            "TresPerNode=gres/gpu:1g.10gb:1"
        )
        self.nodes = "gpu10"
        self.commit, self.untracked_source, self.tracked_diff_status = PIN, "", 0
        self.listing = (
            f"GPU 0: NVIDIA A100-SXM4-80GB (UUID: {PARENT})\n"
            f"  MIG 1g.10gb Device 0: (UUID: {MIG})\n"
            f"  MIG 1g.10gb Device 1: (UUID: {SIBLING})\n"
            f"GPU 1: NVIDIA A100-SXM4-80GB (UUID: {OTHER_PARENT})\n"
        )
        self.parent_query = f"{PARENT}, NVIDIA A100-SXM4-80GB, 570.124.06, Enabled"
        self.identity = {
            "uuid": MIG.removeprefix("MIG-"),
            "device_count": 1,
            "total_memory_bytes": 10 * 1024**3,
            "driver_api_version": 12080,
            "uuid_api": "cuDeviceGetUuid_v2",
        }
        self.identity_probe = Mock(side_effect=lambda: dict(self.identity))
        self.torch = SimpleNamespace(
            __version__="2.9.0+cu128",
            version=SimpleNamespace(cuda="12.8"),
            backends=SimpleNamespace(cuda=SimpleNamespace(is_built=Mock(return_value=True))),
            cuda=SimpleNamespace(
                is_available=Mock(return_value=True),
                device_count=Mock(return_value=1),
                get_device_properties=Mock(
                    return_value=SimpleNamespace(
                        name="NVIDIA A100-SXM4-80GB MIG 1g.10gb",
                        total_memory=10 * 1024**3,
                        # Deliberately a parent UUID: only Driver API v2 is
                        # authoritative for the independently checked MIG ID.
                        uuid=PARENT,
                    )
                ),
                get_device_capability=Mock(return_value=(8, 0)),
            ),
        )
        modules = {"numpy": SimpleNamespace(__version__="2.3.4"), "torch": self.torch}
        self.loader = Mock(side_effect=modules.__getitem__)
        self.python_version = patch.object(preflight.platform, "python_version", return_value="3.12.3")
        self.python_version.start()
        self.addCleanup(self.python_version.stop)

    def runner(self, command, **kwargs):
        if command[:3] == ["scontrol", "show", "job"]:
            output = self.slurm
        elif command[:3] == ["scontrol", "show", "hostnames"]:
            output = self.nodes
        elif command == ["git", "rev-parse", "--show-toplevel"]:
            output = str(ROOT)
        elif command == ["git", "rev-parse", "HEAD"]:
            output = self.commit
        elif command == ["git", "diff", "--quiet", "HEAD", "--"]:
            return subprocess.CompletedProcess(command, self.tracked_diff_status, stdout="", stderr="")
        elif command == ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "*.py"]:
            output = self.untracked_source
        elif command == ["nvidia-smi", "-L"]:
            output = self.listing
        elif command[0] == "nvidia-smi" and command[1] == "--id=" + PARENT:
            output = self.parent_query
        else:
            self.fail(f"Unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def run_check(self, **kwargs):
        return preflight.run_preflight(
            self.out,
            PIN,
            env=self.env,
            runner=kwargs.pop("runner", self.runner),
            module_loader=self.loader,
            identity_probe=self.identity_probe,
            hostname=kwargs.pop("hostname", "gpu10.clients.dur.ac.uk"),
            **kwargs,
        )

    def assert_failed(self, message, **kwargs):
        with self.assertRaisesRegex(preflight.PreflightError, message):
            self.run_check(**kwargs)
        record = json.loads(self.out.read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"]["type"], "PreflightError")
        self.assertNotIn("cuda_smoke", record)
        return record

    def test_login_host_rejected_before_torch_import(self):
        self.assert_failed("compute node", hostname="ncc1.clients.dur.ac.uk")
        self.loader.assert_not_called()

    def test_missing_slurm_job_rejected(self):
        self.env.pop("SLURM_JOB_ID")
        self.assert_failed("Slurm job")
        self.loader.assert_not_called()

    def test_missing_visibility_rejected_without_setting_a_gpu(self):
        self.env.pop("CUDA_VISIBLE_DEVICES")
        self.assert_failed("CUDA_VISIBLE_DEVICES")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.env)

    def test_pending_job_rejected(self):
        self.slurm = self.slurm.replace("RUNNING", "PENDING")
        self.assert_failed("JobState must be RUNNING")

    def test_wrong_slurm_job_rejected(self):
        self.slurm = self.slurm.replace("JobId=123", "JobId=124")
        self.assert_failed("different job ID")

    def test_other_partition_rejected(self):
        self.slurm = self.slurm.replace("Partition=res-gpu-small", "Partition=gpu-bigmem")
        self.assert_failed("Partition must be res-gpu-small")

    def test_other_account_rejected(self):
        self.slurm = self.slurm.replace("Account=ncc_users", "Account=another_group")
        self.assert_failed("Account must be ncc_users")

    def test_other_users_allocation_cannot_be_borrowed(self):
        self.slurm = self.slurm.replace(f"UserId=bhmq03({os.getuid()})", f"UserId=another({os.getuid() + 1})")
        self.assert_failed("UserId must match")
        self.loader.assert_not_called()

    def test_requested_gpu_must_match_not_only_allocated(self):
        self.slurm = self.slurm.replace(
            "ReqTRES=cpu=2,mem=8G,gres/gpu=1,gres/gpu:1g.10gb=1",
            "ReqTRES=cpu=2,mem=8G,gres/gpu=1,gres/gpu:hopper=1",
        )
        self.assert_failed("Requested TRES")

    def test_allocated_gpu_must_match_not_only_requested(self):
        self.slurm = self.slurm.replace(
            "AllocTRES=cpu=2,mem=8G,gres/gpu=1,gres/gpu:1g.10gb=1",
            "AllocTRES=cpu=2,mem=8G,gres/gpu=1,gres/gpu:ampere=1",
        )
        self.assert_failed("Allocated TRES")

    def test_an_additional_gpu_type_is_rejected(self):
        self.slurm = self.slurm.replace(
            "AllocTRES=cpu=2,mem=8G,gres/gpu=1,gres/gpu:1g.10gb=1",
            "AllocTRES=cpu=2,mem=8G,gres/gpu=2,gres/gpu:1g.10gb=1,gres/gpu:ampere=1",
        )
        self.assert_failed("Allocated TRES")

    def test_node_must_match_allocation(self):
        self.nodes = "gpu12"
        self.assert_failed("sole node")

    def test_array_identity_is_checked(self):
        self.env.update(SLURM_ARRAY_JOB_ID="120", SLURM_ARRAY_TASK_ID="1")
        self.slurm += " ArrayJobId=120 ArrayTaskId=2"
        self.assert_failed("different Slurm array element")

    def test_source_commit_must_match(self):
        self.commit = "b" * 40
        self.assert_failed("Git commit mismatch")
        self.loader.assert_not_called()

    def test_untracked_source_changes_are_not_ignored(self):
        self.untracked_source = "scripts/unreviewed.py"
        self.assert_failed("not clean")
        self.loader.assert_not_called()

    def test_tracked_changes_are_rejected(self):
        self.tracked_diff_status = 1
        self.assert_failed("Tracked source tree is not clean")
        self.loader.assert_not_called()

    def test_receipts_and_scheduler_logs_do_not_make_source_dirty(self):
        repository = self.directory / "source-fixture"
        repository.mkdir()

        def git(*args):
            return subprocess.run(
                ["git", *args], cwd=repository, capture_output=True, text=True, check=True
            ).stdout.strip()

        git("init", "-q")
        (repository / "tracked.py").write_text("MODEL = 'fixture'\n")
        git("add", "tracked.py")
        git(
            "-c",
            "user.name=Unit Test",
            "-c",
            "user.email=unit-test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "source fixture",
        )
        observed_pin = git("rev-parse", "HEAD")
        receipt_directory = repository / ".ncc-mig-submission"
        receipt_directory.mkdir()
        (receipt_directory / "request.json").write_text('{"job": 123}\n')
        (receipt_directory / "receipt").write_text("123\n")
        (repository / "slurm-123.out").write_text("job started\n")
        with patch.object(preflight, "ROOT", repository):
            record = preflight._source(observed_pin, subprocess.run, [])
            self.assertTrue(record["source_tree_clean"])
            self.assertIn("All tracked files", record["source_check_scope"])
            (repository / "untracked.py").write_text("raise RuntimeError('not reviewed')\n")
            with self.assertRaisesRegex(preflight.PreflightError, "untracked source"):
                preflight._source(observed_pin, subprocess.run, [])

    def test_python_version_is_exact(self):
        with patch.object(preflight.platform, "python_version", return_value="3.13.9"):
            self.assert_failed("Python 3.12.3")

    def test_cuda_build_versions_are_exact(self):
        self.torch.__version__ = "2.9.0+cpu"
        self.torch.version.cuda = None
        self.assert_failed("Runtime version mismatch")

    def test_cuda_unavailable_does_not_fall_back(self):
        self.torch.cuda.is_available.return_value = False
        self.assert_failed("CUDA is unavailable")
        self.identity_probe.assert_not_called()

    def test_multiple_visible_cuda_devices_rejected(self):
        self.torch.cuda.device_count.return_value = 2
        self.assert_failed("Exactly one PyTorch CUDA device")

    def test_uuid_v1_is_not_accepted_as_mig_identity(self):
        self.identity["uuid_api"] = "cuDeviceGetUuid"
        self.assert_failed("cuDeviceGetUuid_v2 evidence")

    def test_physical_parent_selected_as_device_is_rejected(self):
        self.identity["uuid"] = PARENT.removeprefix("GPU-")
        self.assert_failed("whole physical GPU")

    def test_unlisted_mig_uuid_is_rejected(self):
        self.identity["uuid"] = "55555555-1111-2222-3333-444444444444"
        self.assert_failed("does not match a MIG child")

    def test_another_mig_profile_is_rejected(self):
        self.listing = self.listing.replace(
            f"MIG 1g.10gb Device 0: (UUID: {MIG})", f"MIG 2g.20gb Device 0: (UUID: {MIG})"
        )
        self.assert_failed("Expected MIG profile 1g.10gb")

    def test_orphan_mig_line_is_rejected(self):
        self.listing = f"  MIG 1g.10gb Device 0: (UUID: {MIG})\n"
        self.assert_failed("parent hierarchy")

    def test_duplicate_instance_uuid_is_rejected(self):
        self.listing = self.listing.replace(SIBLING, MIG)
        self.assert_failed("Duplicate device UUID")

    def test_parent_metadata_must_match_selected_child_parent(self):
        self.parent_query = self.parent_query.replace(PARENT, OTHER_PARENT)
        self.assert_failed("parent identity/mode")

    def test_full_parent_memory_cannot_pass_as_slice_memory(self):
        self.torch.cuda.get_device_properties.return_value.total_memory = 80 * 1024**3
        self.assert_failed("instance memory is inconsistent")

    def test_ready_manifest_versions_pin_and_interpreter_are_checked(self):
        manifest = {
            "schema_version": 1,
            "status": "ready",
            "profile": preflight.RUNTIME_PROFILE,
            "git_commit": PIN,
            "runtime": preflight.VERSIONS
            | {
                "interpreter": sys.executable,
                "prefix": sys.prefix,
                "cuda_build": True,
            },
        }
        path = self.directory / "runtime.json"
        path.write_text(json.dumps(manifest))
        checked = preflight._runtime_manifest(path, PIN)
        self.assertEqual(checked["status"], "ready")
        self.assertEqual(len(checked["sha256"]), 64)
        for key, value, message in (
            ("status", "failed", "status=ready"),
            ("git_commit", "b" * 40, "Git pin"),
            ("profile", "another-profile", "schema/profile"),
        ):
            with self.subTest(field=key):
                changed = dict(manifest, **{key: value})
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(preflight.PreflightError, message):
                    preflight._runtime_manifest(path, PIN)
        manifest["runtime"]["interpreter"] = "/another/venv/bin/python"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(preflight.PreflightError, "interpreter/prefix"):
            preflight._runtime_manifest(path, PIN)

    def test_kernel_failure_keeps_hardware_and_raw_inventory_evidence(self):
        with patch.object(preflight, "_cuda_smoke", side_effect=RuntimeError("CUDA kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "CUDA kernel failed"):
                self.run_check()
        record = json.loads(self.out.read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"]["type"], "RuntimeError")
        self.assertEqual(record["hardware"]["mig_uuid"], MIG)
        self.assertTrue(any(row.get("stdout") == self.listing for row in record["raw_inventory"]))

    def test_cuda_driver_probe_uses_v2_instance_uuid_bytes(self):
        def fill_scalar(pointer, value):
            pointer._obj.value = value
            return 0

        def fill_uuid(pointer, device):
            pointer._obj.bytes[:] = uuid.UUID(MIG.removeprefix("MIG-")).bytes
            return 0

        library = SimpleNamespace(
            cuInit=Mock(return_value=0),
            cuDeviceGetCount=Mock(side_effect=lambda pointer: fill_scalar(pointer, 1)),
            cuDeviceGet=Mock(side_effect=lambda pointer, ordinal: fill_scalar(pointer, 0)),
            cuDeviceGetUuid_v2=Mock(side_effect=fill_uuid),
            cuDeviceTotalMem_v2=Mock(side_effect=lambda pointer, device: fill_scalar(pointer, 10 * 1024**3)),
            cuDriverGetVersion=Mock(side_effect=lambda pointer: fill_scalar(pointer, 12080)),
        )
        with patch.object(preflight.ctypes, "CDLL", return_value=library):
            observed = preflight._cuda_identity()
        self.assertEqual(observed["uuid"], MIG.removeprefix("MIG-"))
        self.assertEqual(observed["uuid_api"], "cuDeviceGetUuid_v2")
        self.assertEqual(observed["total_memory_bytes"], 10 * 1024**3)
        library.cuDeviceGetUuid_v2.assert_called_once()

    def test_cuda_driver_error_is_not_replaced_by_a_fallback(self):
        library = SimpleNamespace(cuInit=Mock(return_value=35))
        with patch.object(preflight.ctypes, "CDLL", return_value=library):
            with self.assertRaisesRegex(preflight.PreflightError, "cuInit failed with code 35"):
                preflight._cuda_identity()

    def test_command_failure_is_retained_without_importing_torch(self):
        runner = Mock(return_value=subprocess.CompletedProcess(["scontrol"], 1, "", "permission denied"))
        record = self.assert_failed("Command failed", runner=runner)
        self.assertEqual(record["raw_inventory"][0]["stderr"], "permission denied")
        self.loader.assert_not_called()

    def test_success_accepts_shared_physical_inventory_and_uses_instance_memory(self):
        smoke = {"optimizer_steps": 2, "scope": "mock only; no GPU execution claimed"}
        with patch.object(preflight, "_cuda_smoke", return_value=smoke) as check:
            record = self.run_check()
        check.assert_called_once_with(self.torch)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["git_commit"], PIN)
        self.assertEqual(record["slurm"]["job_id"], "123")
        self.assertEqual(record["slurm"]["cuda_visible_devices"], "0")
        self.assertEqual(record["hardware"]["mig_uuid"], MIG)
        self.assertEqual(record["hardware"]["parent_gpu_uuid"], PARENT)
        self.assertEqual(record["hardware"]["total_memory_bytes"], 10 * 1024**3)
        self.assertEqual(record["hardware"]["parent_mig_mode"], "Enabled")
        self.assertEqual(json.loads(self.out.read_text()), record)
        self.assertFalse(list(self.directory.glob(".*.tmp")))

    def test_existing_evidence_is_never_replaced(self):
        self.out.write_text("previous evidence\n")
        with self.assertRaises(FileExistsError):
            self.run_check()
        self.assertEqual(self.out.read_text(), "previous evidence\n")
        self.loader.assert_not_called()
        self.assertFalse(list(self.directory.glob(".*.tmp")))

    def test_cli_returns_failure_json_when_output_already_exists(self):
        self.out.write_text("previous evidence\n")
        with patch("builtins.print") as printer:
            code = preflight.main(["--out", str(self.out), "--expected-commit", PIN])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(printer.call_args_list[0].args[0])["status"], "failed")
        self.assertEqual(self.out.read_text(), "previous evidence\n")


if __name__ == "__main__":
    unittest.main()
