"""Mock the transport and test archive integrity; never submit a real Slurm job.

Opaque checkpoint bytes and formal-looking metadata are protocol fixtures, not
model performance evidence. These tests require only the Python standard library.
"""

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
PIN = "d" * 40
SPEC = importlib.util.spec_from_file_location(
    "complex_pretraining_package", PROJECT / "scripts/package_complex_pretraining.py"
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def tool(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)


def process_output(completed):
    return f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


class TestComplexPretrainingJobs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        # Exercise real quoting with a home path containing spaces; never alter HOME.
        self.remote_root = self.tmp / "hamilton home"
        self.remote_root.mkdir()
        self.root = self.remote_root / f"isac-pretrain-{PIN[:12]}"
        self.fakepy = self.bin / "school-python"
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "TEST_REMOTE_ROOT": str(self.remote_root),
            "TEST_PIN": PIN,
            "TEST_MAC_REPO": str(PROJECT),
            "TEST_HISTORY": str(self.tmp / "history"),
            "TEST_BUNDLE_ARGS": str(self.tmp / "bundle-args"),
        }
        tool(
            self.bin / "git",
            """
if [[ ${1:-} == -C ]]; then cd "$2"; shift 2; fi
case "$1/${2:-}" in
 rev-parse/--show-toplevel) echo "$TEST_MAC_REPO";;
 rev-parse/HEAD) echo "${MOCK_CHECKOUT_PIN:-$TEST_PIN}";;
 rev-parse/FETCH_HEAD) echo "${MOCK_FETCH_PIN:-$TEST_PIN}";;
 rev-parse/refs/remotes/origin/research/complex-scene-pretraining) echo "${MOCK_REF_PIN:-$TEST_PIN}";;
 bundle/create)
   [[ $# == 4 && "$4" == refs/remotes/origin/research/complex-scene-pretraining ]]
   printf '%s\n' "$@" > "$TEST_BUNDLE_ARGS"
   echo bundle > "$3";;
 init/--quiet|bundle/verify|fetch/*|checkout/--detach) ;;
 diff/--quiet) [[ -z "${MOCK_DIRTY:-}" && ! -f "$TEST_REMOTE_ROOT/source-mutated" ]];;
 ls-files/--others) [[ -z "${MOCK_UNTRACKED:-}" ]] || echo shadow.py;;
 *) echo "Unexpected Git command: $*" >&2; exit 97;;
esac""",
        )
        tool(self.bin / "uname", 'echo "${MOCK_UNAME:-Darwin}"')
        tool(self.bin / "hostname", 'echo "${MOCK_HOST:-login2.ham8.dur.ac.uk}"')
        tool(self.bin / "id", '[[ "$1" == -un ]]; echo "${MOCK_ACCOUNT:-testuser}"')
        tool(self.bin / "scp", 'cp "$1" "$TEST_REMOTE_ROOT/${2#*:}"')
        tool(self.bin / "ssh", 'shift; read -r -a args <<< "${1#*-- }"; exec bash -s -- "${args[@]}"')
        tool(self.bin / "module", '[[ "$1 $2" == "load python/3.13.9cuda" ]]')
        tool(
            self.bin / "sbatch",
            """echo submit >> "$TEST_HISTORY"
[[ "$1 $2 $3" == "--parsable --export=ALL scripts/hamilton_complex_pretraining.sh" ]]
[[ "$ISAC_PRETRAIN_PIN" == "$TEST_PIN" && "$ISAC_PROJECT_ROOT" == "$PWD" ]]
[[ "$ISAC_RUNS_DIR" == "$PWD/runs" ]]
[[ -z "$(compgen -v SBATCH_ || true)" ]]
[[ -z "${MOCK_SBATCH_FAIL:-}" ]] || exit 72
echo "${MOCK_RECEIPT:-391;ham}" """,
        )
        tool(
            self.fakepy,
            """
if [[ "$1" == '-' ]]; then
    cat >/dev/null
    [[ -z "${MOCK_RUNTIME_FAIL:-}" ]] || exit 19
    exit 0
fi
name=${1##*/}
shift
if [[ "$name" == package_complex_pretraining.py ]]; then name="$name:$1"; shift; fi
echo "$name" >> "$TEST_HISTORY"
[[ "$name" != "${MOCK_FAIL_AT:-}" ]] || exit 17
while [[ "$1" != --out ]]; do shift; done
dest=$2
if [[ "$name" == complex_scene_pretraining.py ]]; then
    [[ "$3 $4 $5 $6 $7 $8" == "--device cuda --profile formal --threads 2" ]]
    mkdir "$dest"; dest="$dest/study.json"
fi
echo '{"status":"completed"}' > "$dest"
if [[ "$name" == "${MOCK_MUTATE_AT:-}" ]]; then
    touch "$TEST_REMOTE_ROOT/source-mutated"
fi""",
        )

    def submit(self, **overrides):
        shell = (PROJECT / "scripts/submit_complex_pretraining_from_mac.sh").read_text()
        placeholder = "/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python"
        self.assertEqual(shell.count(placeholder), 1)
        shell = shell.replace(placeholder, str(self.fakepy))
        for variable in ("root", "bundle"):
            fragment = f'{variable}="$HOME/'
            self.assertEqual(shell.count(fragment), 1)
            shell = shell.replace(fragment, f'{variable}="{self.remote_root}/')
        launcher = self.tmp / "launcher.sh"
        launcher.write_text(shell)
        return subprocess.run(
            ["bash", str(launcher), PIN, "testuser"], env=self.env | overrides, text=True, capture_output=True
        )

    def compute(self, job, **overrides):
        self.root.mkdir(exist_ok=True)
        env = self.env | {
            "SLURM_JOB_ID": job,
            "ISAC_PRETRAIN_PIN": PIN,
            "ISAC_PROJECT_ROOT": str(self.root),
            "ISAC_RUNS_DIR": str(self.root / "runs"),
            "ISAC_PYTHON": str(self.fakepy),
        }
        env.pop("SLURM_ARRAY_JOB_ID", None)
        return subprocess.run(
            ["bash", str(PROJECT / "scripts/hamilton_complex_pretraining.sh")],
            env=env | overrides,
            text=True,
            capture_output=True,
        )

    def test_full_bundle_submits_once_without_previous_checkouts_and_clears_overrides(self):
        first = self.submit(
            SBATCH_PARTITION="wrong",
            SBATCH_GRES="gpu:1g.10gb:1",
            SBATCH_TIME="1",
            SBATCH_CPUS_PER_TASK="16",
            SBATCH_MEM_PER_NODE="64G",
        )
        self.assertEqual(first.returncode, 0, process_output(first))
        self.assertIn("Slurm job: 391", first.stdout)
        self.assertIn('pretrain_backup="$HOME/Downloads/isac-pretrain-391"', first.stdout)
        self.assertIn("status.json", first.stdout)
        self.assertIn("hamilton\\ home", first.stdout)
        bundle_args = (self.tmp / "bundle-args").read_text().splitlines()
        self.assertEqual(len(bundle_args), 4)
        self.assertFalse(any(argument.startswith("^") for argument in bundle_args))
        second = self.submit()
        self.assertEqual(second.returncode, 0, process_output(second))
        self.assertIn("Already submitted", second.stdout)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])
        claim = self.root / ".pretrain-submission"
        self.assertEqual((claim / "job-id").read_text().strip(), "391")
        self.assertEqual((claim / "git-commit").read_text().strip(), PIN)
        self.assertEqual((claim / "stage").read_text().strip(), "submitted")
        self.assertEqual(json.loads((claim / "request.json").read_text())["profile"], "formal")

    def test_scheduler_error_is_preserved_without_duplicate_submission(self):
        failed = self.submit(MOCK_SBATCH_FAIL="1")
        self.assertNotEqual(failed.returncode, 0, process_output(failed))
        claim = self.root / ".pretrain-submission"
        self.assertTrue((claim / "receipt.pending").is_file())
        self.assertTrue((claim / "submission.stderr").is_file())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertIn("claim exists without a confirmed receipt", retried.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_malformed_receipt_is_retained_and_not_retried(self):
        malformed = self.submit(MOCK_RECEIPT="warning\n391;ham")
        self.assertNotEqual(malformed.returncode, 0, process_output(malformed))
        pending = self.root / ".pretrain-submission/receipt.pending"
        self.assertEqual(pending.read_text().strip(), "warning\n391;ham")
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_claim_revision_mismatch_does_not_reuse_receipt(self):
        self.assertEqual(self.submit().returncode, 0)
        (self.root / ".pretrain-submission/git-commit").write_text("e" * 40)
        rejected = self.submit()
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        self.assertIn("different revision", rejected.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def assert_submit_rejected(self, **overrides):
        rejected = self.submit(**overrides)
        diagnostic = process_output(rejected)
        self.assertNotEqual(rejected.returncode, 0, diagnostic)
        self.assertFalse((self.tmp / "history").exists(), diagnostic)
        self.assertFalse((self.root / ".pretrain-submission").exists(), diagnostic)

    # Independent fixtures prevent a previous receipt masking later guard failures.
    def test_non_mac_cannot_submit(self):
        self.assert_submit_rejected(MOCK_UNAME="Linux")

    def test_wrong_local_ref_cannot_submit(self):
        self.assert_submit_rejected(MOCK_REF_PIN="e" * 40)

    def test_wrong_remote_host_cannot_submit(self):
        self.assert_submit_rejected(MOCK_HOST="unexpected.example")

    def test_wrong_account_cannot_submit(self):
        self.assert_submit_rejected(MOCK_ACCOUNT="anotheruser")

    def test_wrong_bundle_pin_cannot_submit(self):
        self.assert_submit_rejected(MOCK_FETCH_PIN="e" * 40)

    def test_wrong_checkout_pin_cannot_submit(self):
        self.assert_submit_rejected(MOCK_CHECKOUT_PIN="e" * 40)

    def test_dirty_source_cannot_submit(self):
        self.assert_submit_rejected(MOCK_DIRTY="1")

    def test_untracked_source_cannot_submit(self):
        self.assert_submit_rejected(MOCK_UNTRACKED="1")

    def test_invalid_runtime_cannot_submit(self):
        self.assert_submit_rejected(MOCK_RUNTIME_FAIL="1")

    def test_symlink_checkout_cannot_submit(self):
        other = self.tmp / "other-checkout"
        other.mkdir()
        self.root.symlink_to(other, target_is_directory=True)
        self.assert_submit_rejected()
        self.assertTrue(self.root.is_symlink())

    def test_study_failure_preserves_outputs_and_cannot_overwrite_them(self):
        run = self.compute("392", MOCK_FAIL_AT="complex_scene_pretraining.py")
        self.assertEqual(run.returncode, 17, process_output(run))
        out = self.root / "runs/pretrain-392"
        before = (out / "status.json").read_bytes()
        status = json.loads(before)
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("failed", "study", 17))
        self.assertTrue((out / "preflight.json").is_file())
        self.assertFalse((out / "isac-pretrain-392.tar.gz").exists())
        retried = self.compute("392")
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((out / "status.json").read_bytes(), before)

    def test_complete_job_and_packaging_failure_have_distinct_status(self):
        completed = self.compute("393")
        self.assertEqual(completed.returncode, 0, process_output(completed))
        status = json.loads((self.root / "runs/pretrain-393/status.json").read_text())
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("completed", "done", 0))
        self.assertEqual(
            (self.tmp / "history").read_text().splitlines(),
            [
                "package_complex_pretraining.py:source",
                "hamilton_preflight.py",
                "complex_scene_pretraining.py",
                "package_complex_pretraining.py:package",
            ],
        )
        failed = self.compute("394", MOCK_FAIL_AT="package_complex_pretraining.py:package")
        self.assertEqual(failed.returncode, 17, process_output(failed))
        out = self.root / "runs/pretrain-394"
        self.assertTrue((out / "study/study.json").is_file())
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "package"))

    def test_compute_source_guard_runs_before_training(self):
        rejected = self.compute("395", MOCK_CHECKOUT_PIN="e" * 40)
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        status = json.loads((self.root / "runs/pretrain-395/status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "preflight"))
        self.assertFalse((self.tmp / "history").exists())

    def test_source_change_after_training_blocks_package(self):
        rejected = self.compute("396", MOCK_MUTATE_AT="complex_scene_pretraining.py")
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        out = self.root / "runs/pretrain-396"
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "verify-source"))
        self.assertTrue((out / "study/study.json").is_file())
        self.assertFalse((out / "isac-pretrain-396.tar.gz").exists())

    def test_source_change_during_packaging_never_marks_job_completed(self):
        rejected = self.compute("397", MOCK_MUTATE_AT="package_complex_pretraining.py:package")
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        out = self.root / "runs/pretrain-397"
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "verify-source-final"))
        self.assertTrue((out / "isac-pretrain-397.tar.gz").is_file())

    def test_array_job_cannot_start_formal_study(self):
        rejected = self.compute("398", SLURM_ARRAY_JOB_ID="398")
        self.assertEqual(rejected.returncode, 2, process_output(rejected))
        self.assertFalse((self.root / "runs/pretrain-398").exists())

    def test_formal_allocation_is_explicit_and_bounded(self):
        script = (PROJECT / "scripts/hamilton_complex_pretraining.sh").read_text()
        directives = {line for line in script.splitlines() if line.startswith("#SBATCH ")}
        self.assertIn("#SBATCH --partition=cuda", directives)
        self.assertIn("#SBATCH --gres=gpu:h200_nvl:1", directives)
        self.assertIn("#SBATCH --time=04:00:00", directives)
        self.assertIn("--expected-profile formal", script)

    def test_gpu_partition_controls_cpu_memory_and_temporary_disk(self):
        # Hamilton's GPU policy prohibits separate CPU/RAM/TMP requests. This
        # guards the observed sbatch rejection, including short-option aliases.
        script = (PROJECT / "scripts/hamilton_complex_pretraining.sh").read_text()
        for line in script.splitlines():
            line = line.lstrip()
            if not line.startswith("#SBATCH ") and not line.startswith("#SBATCH\t"):
                continue
            for argument in shlex.split(line[len("#SBATCH") :], comments=True):
                option = argument.partition("=")[0].replace("_", "-")
                with self.subTest(directive=line, argument=argument):
                    self.assertFalse(
                        option.startswith(("-c", "-n", "--cpus-", "--ntasks", "--mem", "--tmp")),
                        "CPU and host-memory allocations are controlled by Hamilton's cuda partition",
                    )
                    self.assertNotIn("tmp:", argument, "Temporary disk is allocated from the GPU request")

    def test_preflight_or_source_snapshot_failure_stops_before_training(self):
        for job, failure in (
            ("399", "hamilton_preflight.py"),
            ("400", "package_complex_pretraining.py:source"),
        ):
            with self.subTest(failure=failure):
                result = self.compute(job, MOCK_FAIL_AT=failure)
                self.assertEqual(result.returncode, 17, process_output(result))
                out = self.root / f"runs/pretrain-{job}"
                status = json.loads((out / "status.json").read_text())
                self.assertEqual((status["status"], status["stage"]), ("failed", "preflight"))
                self.assertFalse((out / "study").exists())

    def test_saved_receipt_symlink_is_not_followed(self):
        self.assertEqual(self.submit().returncode, 0)
        receipt = self.root / ".pretrain-submission/receipt"
        receipt.unlink()
        target = self.tmp / "other-receipt"
        target.write_text("777\n")
        receipt.symlink_to(target)
        result = self.submit()
        self.assertNotEqual(result.returncode, 0, process_output(result))
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])


class TestComplexPretrainingPackage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.source = self.tmp / "source"
        self.source.mkdir()
        (self.source / "src").mkdir()
        (self.source / "src/train.py").write_text(
            "# Synthetic source fixture, not a training implementation.\n"
        )
        self.git("init", "--quiet")
        self.git("add", "src/train.py")
        self.git(
            "-c",
            "user.name=Protocol Tests",
            "-c",
            "user.email=protocol@localhost",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Synthetic source fixture",
        )
        self.pin = self.git("rev-parse", "HEAD").strip()
        self.root = self.tmp / "run"
        self.root.mkdir()
        self.out = self.root / "isac-pretrain-391.tar.gz"
        packager.record_source(self.source, self.pin, self.root / "source_manifest.json")
        self.write(
            "preflight.json",
            {
                "status": "passed",
                "git_commit": self.pin,
                "slurm": {"job_id": "391"},
                "runtime": {"python": "3.13.9", "torch": "2.9.0+cu128", "torch_cuda": "12.8"},
                "gpu": {
                    "name": "NVIDIA H200 NVL",
                    "mig_mode": "Disabled",
                    "total_memory_bytes": 140 * 1024**3,
                },
            },
        )
        self.write(
            "status.json",
            {
                "status": "running",
                "stage": "package",
                "git_commit": self.pin,
                "slurm_job_id": "391",
                "exit_code": None,
            },
        )
        names = ["mixed-enhanced-seed7", "clean-baseline-seed7"]
        self.study = {
            "schema_version": 1,
            "status": "completed",
            "profile": "formal",
            "git_commit": self.pin,
            "slurm_job_id": "391",
            "runtime": dict(packager.SCHOOL_RUNTIME),
            "planned_cells": names,
            "cell_order": names,
            "cells": {},
            "artifacts_sha256": {},
        }
        for name in names:
            cell = {
                "status": "completed",
                "path": name,
                "recipe": name.rsplit("-seed", 1)[0],
                "seed": 7,
                "artifacts_sha256": {},
            }
            for kind, value in (
                ("manifest", {"status": "completed", "protocol": {"initialization": "random"}}),
                ("history", [{"step": 1, "loss": 0.5}]),
                ("metrics", {"synthetic_fixture": True, "rmse_m": 0.3}),
            ):
                relative = f"{name}/{kind}.json"
                cell[kind] = relative
                cell["artifacts_sha256"][relative] = self.write("study/" + relative, value)
            for checkpoint in ("model.pt", "final.pt", "resume.pt"):
                relative = f"{name}/{checkpoint}"
                path = self.root / "study" / relative
                # Opaque bytes test integrity only. They are not a learned checkpoint.
                raw = f"synthetic-checkpoint:{relative}".encode()
                path.write_bytes(raw)
                checksum = hashlib.sha256(raw).hexdigest()
                path.with_suffix(".pt.sha256").write_text(checksum + "\n")
                cell["artifacts_sha256"][relative] = checksum
                cell["artifacts_sha256"][relative + ".sha256"] = packager.digest_file(
                    path.with_suffix(".pt.sha256")
                )
            self.study["cells"][name] = cell
            self.study["artifacts_sha256"].update(cell["artifacts_sha256"])
        self.write_study()

    def git(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.source), *args], text=True, capture_output=True, check=True
        )
        return result.stdout

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = packager.json_bytes(value)
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def write_study(self):
        self.write("study/study.json", self.study)

    def update_digest(self, relative):
        checksum = packager.digest_file(self.root / "study" / relative)
        self.study["artifacts_sha256"][relative] = checksum
        name = relative.split("/", 1)[0]
        self.study["cells"][name]["artifacts_sha256"][relative] = checksum
        self.write_study()

    def package(self, expected_profile="formal"):
        return packager.package(self.root, self.out, self.source, expected_profile)

    def test_archive_contains_all_weights_and_verified_digests(self):
        result = self.package()
        self.assertEqual(result["status"], "packaged")
        self.assertTrue(result["weights_included"])
        self.assertEqual(result["sha256"], packager.digest_file(self.out))
        self.assertEqual(
            self.out.with_suffix(".gz.sha256").read_text(), f"{result['sha256']}  {self.out.name}\n"
        )
        with tarfile.open(self.out) as archive:
            names = archive.getnames()
            self.assertEqual(len([name for name in names if name.endswith(".pt")]), 6)
            bundle = json.load(archive.extractfile("bundle_manifest.json"))
            self.assertTrue(bundle["weights_included"])
            self.assertEqual(len(bundle["checkpoints"]), 6)
            self.assertEqual(bundle["git_commit"], self.pin)
            for path, checkpoint in bundle["checkpoints"].items():
                self.assertIn(path, names)
                self.assertEqual(checkpoint["sha256"], bundle["artifacts"][path]["sha256"])
            for name, expected in bundle["artifacts"].items():
                raw = archive.extractfile(name).read()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected["sha256"])
                self.assertEqual(len(raw), expected["bytes"])
            status = json.load(archive.extractfile("job_status_at_packaging.json"))
            self.assertEqual((status["status"], status["stage"]), ("running", "package"))
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.package()
        self.assertEqual(self.out.read_bytes(), before)

    def test_development_archive_is_labelled_and_cannot_satisfy_formal_launch(self):
        self.study.update(
            profile="development",
            slurm_job_id=None,
            runtime={"python": "3.12.14", "torch": "2.14.1+cpu", "numpy": "2.3.4", "device": "cpu"},
        )
        self.write_study()
        (self.root / "preflight.json").unlink()
        (self.root / "status.json").unlink()
        with self.assertRaisesRegex(ValueError, "profile"):
            self.package()
        result = self.package(expected_profile="development")
        self.assertTrue(result["weights_included"])
        with tarfile.open(self.out) as archive:
            bundle = json.load(archive.extractfile("bundle_manifest.json"))
            self.assertEqual(bundle["profile"], "development")
            self.assertIsNone(bundle["slurm_job_id"])
            self.assertNotIn("preflight.json", archive.getnames())

    def test_incomplete_study_or_omitted_planned_cell_cannot_package(self):
        self.study["status"] = "running"
        self.write_study()
        with self.assertRaisesRegex(ValueError, "completed"):
            self.package()
        self.study["status"] = "completed"
        self.study["cells"].pop("clean-baseline-seed7")
        self.write_study()
        with self.assertRaisesRegex(ValueError, "planned"):
            self.package()
        self.assertFalse(self.out.exists())

    def test_smoke_archive_is_explicit_and_cannot_satisfy_formal_launch(self):
        self.study.update(profile="smoke", slurm_job_id=None, runtime={"device": "cpu"})
        self.write_study()
        (self.root / "preflight.json").unlink()
        (self.root / "status.json").unlink()
        with self.assertRaisesRegex(ValueError, "profile"):
            self.package()
        result = self.package(expected_profile="smoke")
        self.assertTrue(result["weights_included"])
        with tarfile.open(self.out) as archive:
            bundle = json.load(archive.extractfile("bundle_manifest.json"))
            self.assertEqual(bundle["profile"], "smoke")
            self.assertIn("Smoke training evidence", bundle["completion_scope"])

    def test_changed_or_missing_evidence_is_detected(self):
        for relative in ("mixed-enhanced-seed7/model.pt", "clean-baseline-seed7/history.json"):
            with self.subTest(relative=relative):
                path = self.root / "study" / relative
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "Changed result evidence"):
                    self.package()
                path.unlink()
                with self.assertRaisesRegex(ValueError, "inventory"):
                    self.package()
                path.write_bytes(original)
        self.assertFalse(self.out.exists())

    def test_checkpoint_digest_is_verified_independently_of_inventory(self):
        relative = "mixed-enhanced-seed7/model.pt.sha256"
        (self.root / "study" / relative).write_text("e" * 64 + "\n")
        self.update_digest(relative)
        with self.assertRaisesRegex(ValueError, "Checkpoint checksum"):
            self.package()

    def test_undigested_file_and_unregistered_checkpoint_are_rejected(self):
        path = self.root / "study/mixed-enhanced-seed7/extra.pt"
        path.write_bytes(b"unexpected weights")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.package()
        self.update_digest("mixed-enhanced-seed7/extra.pt")
        with self.assertRaisesRegex(ValueError, "Unexpected checkpoint"):
            self.package()

    def test_nan_infinity_and_duplicate_keys_are_rejected(self):
        relative = "mixed-enhanced-seed7/metrics.json"
        path = self.root / "study" / relative
        for raw in (b'{"x":NaN}', b'{"x":1e1000}', b'{"x":1,"x":2}'):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                self.update_digest(relative)
                with self.assertRaises(ValueError):
                    self.package()

    def test_result_symlinks_and_path_traversal_are_rejected(self):
        path = self.root / "study/mixed-enhanced-seed7/model.pt"
        path.rename(self.tmp / "outside.pt")
        path.symlink_to(self.tmp / "outside.pt")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.package()
        path.unlink()
        (self.tmp / "outside.pt").rename(path)
        self.study["artifacts_sha256"]["../outside.pt"] = "a" * 64
        self.write_study()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.package()

    def test_directory_symlink_is_rejected(self):
        (self.root / "study/linked-data").symlink_to(self.tmp, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink result directory"):
            self.package()

    def test_failed_job_runtime_drift_or_identity_mismatch_cannot_package(self):
        status_path = self.root / "status.json"
        original_status = status_path.read_bytes()
        status = json.loads(original_status)
        status.update(status="failed", stage="study", exit_code=9)
        self.write("status.json", status)
        with self.assertRaisesRegex(ValueError, "successful study"):
            self.package()
        status_path.write_bytes(original_status)
        self.study["runtime"]["torch"] = "another-version"
        self.write_study()
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.package()
        self.study["runtime"] = dict(packager.SCHOOL_RUNTIME)
        self.study["slurm_job_id"] = "392"
        self.write_study()
        with self.assertRaisesRegex(ValueError, "identities"):
            self.package()

    def test_changed_source_and_assume_unchanged_bypass_are_detected(self):
        path = self.source / "src/train.py"
        original = path.read_bytes()
        path.write_bytes(original + b"# source drift\n")
        with self.assertRaises(subprocess.CalledProcessError):
            self.package()
        self.git("update-index", "--assume-unchanged", "src/train.py")
        self.git("diff", "--quiet", "HEAD", "--")
        with self.assertRaisesRegex(ValueError, "pinned Git blob"):
            self.package()

    def test_formal_package_rejects_wrong_gpu_mig_and_preflight_runtime(self):
        path = self.root / "preflight.json"
        original = path.read_text()
        for field, value in (
            ("name", "NVIDIA A100"),
            ("mig_mode", "Enabled"),
            ("total_memory_bytes", 40 * 1024**3),
        ):
            with self.subTest(field=field):
                preflight = json.loads(original)
                preflight["gpu"][field] = value
                self.write("preflight.json", preflight)
                with self.assertRaisesRegex(ValueError, "Whole H200"):
                    self.package()
        preflight = json.loads(original)
        preflight["runtime"]["torch_cuda"] = "12.9"
        self.write("preflight.json", preflight)
        with self.assertRaisesRegex(ValueError, "preflight runtime"):
            self.package()

    def test_incomplete_cell_or_changed_per_cell_manifest_is_rejected(self):
        cell = self.study["cells"]["mixed-enhanced-seed7"]
        cell["status"] = "running"
        self.write_study()
        with self.assertRaisesRegex(ValueError, "incomplete training cell"):
            self.package()
        cell["status"] = "completed"
        cell["artifacts_sha256"]["mixed-enhanced-seed7/history.json"] = "e" * 64
        self.write_study()
        with self.assertRaisesRegex(ValueError, "Cell artifact inventory"):
            self.package()

    def test_untracked_source_or_changed_source_manifest_is_detected(self):
        (self.source / "src/shadow.py").write_text("# untracked import shadow\n")
        with self.assertRaisesRegex(ValueError, "Untracked source"):
            self.package()
        (self.source / "src/shadow.py").unlink()
        manifest = json.loads((self.root / "source_manifest.json").read_text())
        manifest["files"]["src/train.py"]["sha256"] = "e" * 64
        self.write("source_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.package()

    def test_evidence_change_during_packaging_prevents_archive_publication(self):
        original = packager.verify_evidence
        calls = 0

        def mutate_before_final_check(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                (self.root / "study/mixed-enhanced-seed7/final.pt").write_bytes(b"changed during packaging")
            return original(*args, **kwargs)

        with mock.patch.object(packager, "verify_evidence", side_effect=mutate_before_final_check):
            with self.assertRaisesRegex(ValueError, "Changed result evidence"):
                self.package()
        self.assertFalse(self.out.exists())
        self.assertTrue(self.out.with_suffix(".gz.writing").is_file())

    def test_existing_archive_symlink_is_never_overwritten(self):
        target = self.tmp / "existing"
        target.write_bytes(b"retain me")
        self.out.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.package()
        self.assertEqual(target.read_bytes(), b"retain me")


if __name__ == "__main__":
    unittest.main()
