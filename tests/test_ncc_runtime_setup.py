"""Exercise the NCC setup job with mocked commands; no packages or CUDA installed."""

from __future__ import annotations

import json
import os
import pwd
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ncc_runtime_setup.sh"
SOURCE = SCRIPT.read_text().split("<<'PY_NCC_RUNTIME'\n", 1)[1].rsplit("\nPY_NCC_RUNTIME", 1)[0]
runtime = types.ModuleType("ncc_runtime_setup_under_test")
exec(compile(SOURCE, str(SCRIPT), "exec"), runtime.__dict__)


class RuntimeSetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.project = self.root / "checkout"
        self.project.mkdir()
        self.runtime_root = self.root / "ncc-runtime"
        self.pin = "a" * 40
        self.checkout_pin = self.pin
        self.dirty = False
        self.env = {
            "ISAC_PROJECT_ROOT": str(self.project),
            "ISAC_RUNS_DIR": str(self.project / "runs"),
            "ISAC_NCC_RUNTIME_ROOT": str(self.runtime_root),
            "ISAC_NCC_PIN": self.pin,
            "SLURM_JOB_ID": "555",
            "SLURM_JOB_PARTITION": "cpu",
            "PIP_EXTRA_INDEX_URL": "https://credential:secret@example.invalid/simple",
            "PIP_INDEX_URL": "https://another.invalid",
            "PYTHONPATH": "/unrelated/python",
        }
        self.system = {
            "python": "3.12.3",
            "interpreter": "/usr/bin/python3",
            "platform": "test Linux",
            "system": "Linux",
            "machine": "x86_64",
            "libc": ["glibc", "2.39"],
        }
        user = pwd.getpwuid(os.getuid())
        self.slurm = (
            f"JobId=555 JobState=RUNNING Partition=cpu Account=ncc_users QOS=debug "
            f"UserId={user.pw_name}({user.pw_uid}) NumNodes=1 NumCPUs=2 "
            "NodeList=cpu0 AllocTRES=cpu=2,mem=8G,node=1"
        )
        self.fail_install = None
        self.import_version = "2.9.0+cu128"
        self.runner = Mock(side_effect=self.fake_runner)

    def fake_runner(self, command, **kwargs):
        output = ""
        code = 0
        if command[:3] == ["scontrol", "show", "job"]:
            output = self.slurm
        elif command[:3] == ["scontrol", "show", "hostnames"]:
            output = "cpu0\n"
        elif command[0] == "git":
            if command[3:] == ["rev-parse", "--show-toplevel"]:
                output = str(self.project)
            elif command[3:] == ["rev-parse", "HEAD"]:
                output = self.checkout_pin
            elif command[3:] == ["diff", "--quiet", "HEAD", "--"]:
                code = int(self.dirty)
        elif "venv" in command:
            venv = Path(command[-1])
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").touch()
        elif "pip" in command:
            env = kwargs["env"]
            self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
            self.assertNotIn("PIP_INDEX_URL", env)
            self.assertNotIn("PIP_EXTRA_INDEX_URL", env)
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("--extra-index-url", command)
            if "install" in command:
                name = command[-1].split("==", 1)[0]
                if self.fail_install == name:
                    code = 1
                else:
                    report = Path(command[command.index("--report") + 1])
                    report.write_text(json.dumps({"install": [{"metadata": {"name": name}}]}))
                kwargs["stdout"].write("Mock installation only; no package downloaded.\n")
            elif "freeze" in command:
                output = "numpy==2.3.4\npip==24.0\ntorch==2.9.0+cu128\n"
            elif "check" in command:
                kwargs["stdout"].write("No broken requirements found.\n")
            elif "--version" in command:
                output = "pip 24.0 (mocked for unit tests)"
        elif "-c" in command:
            # Match the interpreter actually invoked, including canonical paths
            # when the temporary directory has a symlinked parent on macOS.
            interpreter = Path(command[0])
            venv = interpreter.parent.parent
            output = json.dumps(
                {
                    "python": "3.12.3",
                    "numpy": "2.3.4",
                    "torch": self.import_version,
                    "torch_cuda": "12.8",
                    "cuda_build": True,
                    "interpreter": str(interpreter),
                    "prefix": str(venv),
                    "base_prefix": "/usr",
                    "numpy_file": str(venv / "lib/numpy/__init__.py"),
                    "torch_file": str(venv / "lib/torch/__init__.py"),
                }
            )
        else:
            self.fail(f"Unexpected subprocess: {command}")
        return subprocess.CompletedProcess(command, code, stdout=output, stderr="")

    def setup_runtime(self):
        return runtime.run_setup(
            env=self.env,
            runner=self.runner,
            system_probe=lambda: self.system,
            hostname=lambda: "cpu0.example.test",
        )

    def manifest(self):
        return json.loads((self.runtime_root / "runtime.json").read_text())

    def test_success_records_runtime_and_provenance_without_a_gpu_claim(self):
        result = self.setup_runtime()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["runtime"]["torch"], "2.9.0+cu128")
        self.assertEqual(result["slurm_setup_job_id"], "555")
        self.assertEqual(result["git_commit"], self.pin)
        self.assertFalse(result["cuda_execution_validated"])
        self.assertEqual(len(result["installation_reports"]), 2)
        self.assertIn("torch==2.9.0+cu128", result["dependencies"])
        self.assertEqual(len(result["freeze"]["sha256"]), 64)
        self.assertEqual(self.manifest(), result)
        commands = [call.args[0] for call in self.runner.call_args_list]
        installs = [command for command in commands if "install" in command]
        self.assertEqual(len(installs), 2)
        self.assertEqual(installs[0][installs[0].index("--index-url") + 1], "https://pypi.org/simple")
        self.assertEqual(
            installs[1][installs[1].index("--index-url") + 1], "https://download.pytorch.org/whl/cu128"
        )
        self.assertTrue(all("--only-binary=:all:" in command for command in installs))
        self.assertNotIn("nvidia-smi", [command[0] for command in commands])
        self.assertNotIn("secret", (self.runtime_root / "runtime.json").read_text())

    def test_install_failure_is_failed_and_preserves_logs(self):
        self.fail_install = "torch"
        with self.assertRaisesRegex(runtime.SetupError, "Command failed"):
            self.setup_runtime()
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.manifest()["stage"], "install-torch")
        self.assertTrue((self.runtime_root / "torch-install.log").exists())
        self.assertNotIn("runtime", self.manifest())
        self.assertNotIn("ready", (self.runtime_root / "runtime.json").read_text())

    def test_success_with_a_symlinked_parent_directory(self):
        alias = self.root / "linked-root"
        alias.symlink_to(self.root, target_is_directory=True)
        self.project = alias / "checkout"
        self.runtime_root = alias / "ncc-runtime"
        self.env.update(
            ISAC_PROJECT_ROOT=str(self.project),
            ISAC_RUNS_DIR=str(self.project / "runs"),
            ISAC_NCC_RUNTIME_ROOT=str(self.runtime_root),
        )
        self.assertNotEqual(self.runtime_root, self.runtime_root.resolve())
        result = self.setup_runtime()
        expected_venv = self.runtime_root.resolve() / "venv"
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["runtime"]["interpreter"], str(expected_venv / "bin" / "python"))
        self.assertEqual(result["runtime"]["prefix"], str(expected_venv))
        self.assertEqual(self.manifest(), result)

    def test_existing_directory_is_not_reused_even_without_manifest(self):
        self.runtime_root.mkdir()
        sentinel = self.runtime_root / "original"
        sentinel.write_text("preserve me\n")
        with self.assertRaises(FileExistsError):
            self.setup_runtime()
        self.assertEqual(sentinel.read_text(), "preserve me\n")
        self.assertFalse((self.runtime_root / "runtime.json").exists())
        self.assertFalse((self.runtime_root / "venv").exists())

    def test_ready_environment_is_not_overwritten(self):
        self.setup_runtime()
        before = (self.runtime_root / "runtime.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.setup_runtime()
        self.assertEqual((self.runtime_root / "runtime.json").read_bytes(), before)

    def test_existing_dangling_symlink_is_not_repurposed(self):
        destination = self.root / "missing-target"
        self.runtime_root.symlink_to(destination, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.setup_runtime()
        self.assertTrue(self.runtime_root.is_symlink())
        self.assertFalse(destination.exists())

    def test_system_version_mismatch_fails_before_venv_or_download(self):
        self.system["python"] = "3.13.9"
        with self.assertRaisesRegex(runtime.SetupError, "version 3.12.3"):
            self.setup_runtime()
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.manifest()["system_runtime"]["python"], "3.13.9")
        self.assertFalse((self.runtime_root / "venv").exists())
        self.assertFalse(any("install" in call.args[0] for call in self.runner.call_args_list))

    def test_imported_version_mismatch_never_marks_ready(self):
        self.import_version = "2.9.0+cpu"
        with self.assertRaisesRegex(runtime.SetupError, "Installed runtime differs"):
            self.setup_runtime()
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.manifest()["runtime"]["torch"], "2.9.0+cpu")

    def test_login_execution_is_rejected_before_files_or_subprocesses(self):
        self.env.pop("SLURM_JOB_ID")
        with self.assertRaisesRegex(runtime.SetupError, "single Slurm CPU"):
            self.setup_runtime()
        self.runner.assert_not_called()
        self.assertFalse(self.runtime_root.exists())

    def test_pending_job_is_rejected(self):
        self.slurm = self.slurm.replace("JobState=RUNNING", "JobState=PENDING")
        with self.assertRaisesRegex(runtime.SetupError, "controller"):
            self.setup_runtime()
        self.assertFalse(self.runtime_root.exists())

    def test_another_users_running_job_is_rejected(self):
        current = pwd.getpwuid(os.getuid())
        self.slurm = self.slurm.replace(f"UserId={current.pw_name}({current.pw_uid})", "UserId=other(99999)")
        with self.assertRaisesRegex(runtime.SetupError, "controller"):
            self.setup_runtime()
        self.assertFalse(self.runtime_root.exists())

    def test_source_pin_mismatch_prevents_environment_creation(self):
        self.checkout_pin = "b" * 40
        with self.assertRaisesRegex(runtime.SetupError, "differs from ISAC_NCC_PIN"):
            self.setup_runtime()
        self.assertFalse(self.runtime_root.exists())

    def test_tracked_source_changes_prevent_environment_creation(self):
        self.dirty = True
        with self.assertRaisesRegex(runtime.SetupError, "git command failed"):
            self.setup_runtime()
        self.assertFalse(self.runtime_root.exists())

    def test_login_host_with_forged_slurm_environment_is_rejected(self):
        with self.assertRaisesRegex(runtime.SetupError, "allocated compute node"):
            runtime.run_setup(
                env=self.env, runner=self.runner, system_probe=lambda: self.system, hostname=lambda: "ncc1"
            )
        self.assertFalse(self.runtime_root.exists())

    def test_runtime_inside_checkout_is_rejected(self):
        self.env["ISAC_NCC_RUNTIME_ROOT"] = str(self.project / "venv")
        with self.assertRaisesRegex(runtime.SetupError, "outside"):
            self.setup_runtime()
        self.runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
