"""Exercise submission, failure preservation and complete evidence packaging without Slurm/GPU."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_vision_learning_study import evidence_fixture
from vision_learning_study import cell_name, cells, design, summarize, validate_training_evidence

PROJECT = Path(__file__).resolve().parents[1]
BASE = "3225ce7435c41e1695b7d8f3f1740586b6edbd00"
PIN = "d" * 40
SPEC = importlib.util.spec_from_file_location(
    "vision_package", PROJECT / "scripts/package_vision_learning_study.py"
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def tool(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)


def process_output(completed):
    return f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


class TestVisionLearningJobs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.remote_root = self.tmp / "hamilton"
        self.remote_root.mkdir()
        self.prior = self.remote_root / "isac-capacity-3225ce7435c4"
        self.prior.mkdir()
        self.root = self.remote_root / f"isac-vision-{PIN[:12]}"
        self.fakepy = self.bin / "school-python"
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "TEST_REMOTE_ROOT": str(self.remote_root),
            "TEST_BASE": BASE,
            "TEST_PIN": PIN,
            "TEST_MAC_REPO": str(PROJECT),
            "TEST_HISTORY": str(self.tmp / "history"),
        }
        tool(
            self.bin / "git",
            """
if [[ ${1:-} == -C ]]; then cd "$2"; shift 2; fi
case "$1/$2" in
 rev-parse/--show-toplevel) echo "$TEST_MAC_REPO";;
 rev-parse/HEAD)
   [[ "$PWD" == "$TEST_REMOTE_ROOT/isac-capacity-3225ce7435c4" ]] && echo "$TEST_BASE" || echo "${MOCK_CHECKOUT_PIN:-$TEST_PIN}";;
 rev-parse/FETCH_HEAD) echo "$TEST_PIN";;
 rev-parse/refs/remotes/origin/research/vision-supervision-h200) echo "${MOCK_REF_PIN:-$TEST_PIN}";;
 merge-base/--is-ancestor) [[ "$3" == "$TEST_BASE" && "$4" == "$TEST_PIN" ]];;
 bundle/create) echo bundle > "$3";;
 bundle/verify|fetch/*) ;;
 diff/--quiet) [[ -z "${MOCK_DIRTY:-}" ]];;
 ls-files/--others) [[ -z "${MOCK_UNTRACKED:-}" ]] || echo shadow.py;;
 worktree/add) mkdir "$4";;
 *) echo "Unexpected Git command: $*" >&2; exit 97;;
esac""",
        )
        tool(self.bin / "uname", 'echo "${MOCK_UNAME:-Darwin}"')
        tool(self.bin / "hostname", 'echo "${MOCK_HOST:-login2.ham8.dur.ac.uk}"')
        tool(self.bin / "scp", 'cp "$1" "$TEST_REMOTE_ROOT/${2#*:}"')
        tool(self.bin / "ssh", 'shift; read -r -a args <<< "${1#*-- }"; exec bash -s -- "${args[@]}"')
        tool(self.bin / "module", '[[ "$1 $2" == "load python/3.13.9cuda" ]]')
        tool(
            self.bin / "sbatch",
            """echo submit >> "$TEST_HISTORY"
[[ "$1 $2 $3" == "--parsable --export=ALL scripts/hamilton_vision_learning.sh" ]]
[[ "$ISAC_VISION_PIN" == "$TEST_PIN" && "$ISAC_PROJECT_ROOT" == "$PWD" ]]
[[ "$ISAC_RUNS_DIR" == "$PWD/runs" ]]
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
echo "$name" >> "$TEST_HISTORY"
[[ "$name" != "${MOCK_FAIL_AT:-}" ]] || exit 17
while [[ "$1" != --out ]]; do shift; done
dest=$2
if [[ "$name" == vision_learning_study.py ]]; then
    [[ "$3 $4 $5 $6" == "--device cuda --profile formal" ]]
    mkdir "$dest"; dest="$dest/study.json"
fi
echo '{"status":"completed"}' > "$dest" """,
        )

    def submit(self, **overrides):
        shell = (PROJECT / "scripts/submit_vision_learning_from_mac.sh").read_text()
        placeholder = "/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python"
        self.assertEqual(shell.count(placeholder), 1)
        shell = shell.replace(placeholder, str(self.fakepy))
        # Redirect only the remote test fixture paths; do not alter the process's real HOME.
        for variable in ("prior_root", "root", "bundle"):
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
            "ISAC_VISION_PIN": PIN,
            "ISAC_PROJECT_ROOT": str(self.root),
            "ISAC_RUNS_DIR": str(self.root / "runs"),
            "ISAC_PYTHON": str(self.fakepy),
        }
        env.pop("SLURM_ARRAY_JOB_ID", None)
        return subprocess.run(
            ["bash", str(PROJECT / "scripts/hamilton_vision_learning.sh")],
            env=env | overrides,
            text=True,
            capture_output=True,
        )

    def test_submit_once_and_report_saved_receipt_without_prior_results(self):
        first = self.submit()
        self.assertEqual(first.returncode, 0, process_output(first))
        self.assertIn("Slurm job: 391", first.stdout)
        self.assertIn('vision_backup="$HOME/Downloads/isac-vision-391"', first.stdout)
        self.assertIn("status.json", first.stdout)
        self.prior.rmdir()  # Existing receipt lookup does not require old checkpoint/result directories.
        second = self.submit()
        self.assertEqual(second.returncode, 0, process_output(second))
        self.assertIn("Already submitted", second.stdout)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])
        request = json.loads((self.root / ".vision-submission/request.json").read_text())
        self.assertEqual(request["git_commit"], PIN)

    def test_unknown_receipt_and_scheduler_error_are_not_retried(self):
        failed = self.submit(MOCK_SBATCH_FAIL="1")
        self.assertNotEqual(failed.returncode, 0, process_output(failed))
        self.assertTrue((self.root / ".vision-submission/receipt.pending").exists())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertIn("claim exists without a confirmed receipt", retried.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_malformed_success_receipt_is_retained_and_not_retried(self):
        malformed = self.submit(MOCK_RECEIPT="warning\n391;ham")
        self.assertNotEqual(malformed.returncode, 0, process_output(malformed))
        self.assertIn("Unexpected scheduler receipt", malformed.stderr)
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def assert_submit_rejected(self, **overrides):
        rejected = self.submit(**overrides)
        diagnostic = process_output(rejected)
        self.assertNotEqual(rejected.returncode, 0, diagnostic)
        self.assertFalse((self.tmp / "history").exists(), diagnostic)
        self.assertFalse((self.root / ".vision-submission").exists(), diagnostic)

    # Each rejection has its own setUp, so a faulty submission cannot leave a
    # saved receipt that bypasses another case's guard.
    def test_non_mac_host_cannot_submit(self):
        self.assert_submit_rejected(MOCK_UNAME="Linux")

    def test_wrong_local_ref_cannot_submit(self):
        self.assert_submit_rejected(MOCK_REF_PIN="e" * 40)

    def test_wrong_remote_host_cannot_submit(self):
        self.assert_submit_rejected(MOCK_HOST="unexpected.example")

    def test_wrong_checkout_pin_cannot_submit(self):
        self.assert_submit_rejected(MOCK_CHECKOUT_PIN="e" * 40)

    def test_dirty_source_cannot_submit(self):
        self.assert_submit_rejected(MOCK_DIRTY="1")

    def test_untracked_source_cannot_submit(self):
        self.assert_submit_rejected(MOCK_UNTRACKED="1")

    def test_invalid_runtime_cannot_submit(self):
        self.assert_submit_rejected(MOCK_RUNTIME_FAIL="1")

    def test_study_failure_preserves_outputs_and_cannot_overwrite_them(self):
        run = self.compute("392", MOCK_FAIL_AT="vision_learning_study.py")
        self.assertEqual(run.returncode, 17, process_output(run))
        out = self.root / "runs/vision-392"
        before = (out / "status.json").read_bytes()
        status = json.loads(before)
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("failed", "study", 17))
        self.assertTrue((out / "preflight.json").is_file())
        self.assertFalse((out / "isac-vision-392.tar.gz").exists())
        retried = self.compute("392")
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((out / "status.json").read_bytes(), before)

    def test_complete_job_and_packaging_failure_have_distinct_status(self):
        completed = self.compute("393")
        self.assertEqual(completed.returncode, 0, process_output(completed))
        status = json.loads((self.root / "runs/vision-393/status.json").read_text())
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("completed", "done", 0))
        self.assertEqual(
            (self.tmp / "history").read_text().splitlines(),
            ["hamilton_preflight.py", "vision_learning_study.py", "package_vision_learning_study.py"],
        )
        failed = self.compute("394", MOCK_FAIL_AT="package_vision_learning_study.py")
        self.assertEqual(failed.returncode, 17, process_output(failed))
        out = self.root / "runs/vision-394"
        self.assertTrue((out / "study/study.json").is_file())
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "package"))

    def test_compute_source_guard_failure_is_recorded_before_training(self):
        rejected = self.compute("395", MOCK_CHECKOUT_PIN="e" * 40)
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        status = json.loads((self.root / "runs/vision-395/status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "preflight"))
        self.assertFalse((self.tmp / "history").exists())


class TestVisionLearningPackage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "isac-vision-391.tar.gz"
        self.spec = design("formal")
        self.study = {
            "status": "completed",
            "profile": "formal",
            "git_commit": PIN,
            "slurm_job_id": "391",
            "runtime": {"device": "cuda"},
            "design": self.spec,
            "fresh_test_consumed": False,
            "test_episode_seeds": [],
            "cell_order": [cell_name(*k) for k in cells(self.spec)],
            "cells": {},
        }
        self.write(
            "preflight.json",
            {
                "status": "passed",
                "git_commit": PIN,
                "slurm": {"job_id": "391"},
                "gpu": {
                    "name": "NVIDIA H200 NVL",
                    "mig_mode": "Disabled",
                    "total_memory_bytes": 140 * 1024**3,
                },
            },
        )
        self.write(
            "status.json", {"status": "running", "stage": "package", "slurm_job_id": "391", "git_commit": PIN}
        )
        for key in cells(self.spec):
            name = cell_name(*key)
            prefix = f"study/cells/{name}"
            evidence = evidence_fixture(self.spec, key)
            hashes = {
                f"{item}_sha256": self.write(f"{prefix}/{item}.json", value)
                for item, value in zip(
                    ("manifest", "history", "epoch_history", "visual_diagnostics"), evidence, strict=True
                )
            }
            weights = f"fixture-{name}".encode()
            checkpoint = self.root / prefix / "model.pt"
            checkpoint.write_bytes(weights)
            checksum = hashlib.sha256(weights).hexdigest()
            checkpoint.with_suffix(".pt.sha256").write_text(checksum + "\n")
            manifest = evidence[0]
            self.study["cells"][name] = {
                "status": "completed",
                "learning_rate": key[0],
                "vision_loss_weight": key[1],
                "initialization": key[2],
                "parameters": manifest["parameters"],
                "visual_encoder_parameters": manifest["visual_encoder_parameters"],
                "initial_parameters_sha256": manifest["initial_parameters_sha256"],
                "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
                "checkpoint": f"cells/{name}/model.pt",
                "checkpoint_sha256": checksum,
                "metrics": validate_training_evidence(*evidence, self.spec, key, "cuda"),
                **hashes,
            }
        self.study["summary"] = summarize(self.study["cells"], self.spec)
        self.write("study/study.json", self.study)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, indent=2) + "\n").encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_complete_bundle_has_all_diagnostics_hashes_and_no_weights_or_fresh_test(self):
        result = packager.package(self.root, self.out)
        self.assertEqual(result["status"], "packaged")
        self.assertFalse(result["weights_included"])
        self.assertEqual(result["sha256"], hashlib.sha256(self.out.read_bytes()).hexdigest())
        with tarfile.open(self.out) as archive:
            names = archive.getnames()
            self.assertEqual(len([n for n in names if n.endswith("/visual_diagnostics.json")]), 8)
            self.assertEqual(len([n for n in names if n.endswith("/history.json")]), 8)
            self.assertFalse(any(n.endswith(".pt") or "fresh_test" in n for n in names))
            self.assertTrue(all(n.endswith((".json", ".sha256")) for n in names))
            bundle = json.load(archive.extractfile("bundle_manifest.json"))
            self.assertFalse(bundle["weights_included"])
            self.assertFalse(bundle["fresh_test_consumed"])
            self.assertEqual(len(bundle["checkpoints"]), 8)
            for name, meta in bundle["artifacts"].items():
                raw = archive.extractfile(name).read()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"])
                self.assertEqual(len(raw), meta["bytes"])
            status = json.load(archive.extractfile("job_status_at_packaging.json"))
            self.assertEqual(status["status"], "running")
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            packager.package(self.root, self.out)
        self.assertEqual(before, self.out.read_bytes())

    def test_missing_cell_wrong_identity_or_changed_design_cannot_package(self):
        name = cell_name(0.0003, 1.0, 11)
        cell = self.study["cells"].pop(name)
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "formal visual-learning cell"):
            packager.package(self.root, self.out)
        self.study["cells"][name] = cell
        self.study["slurm_job_id"] = "392"
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "identities"):
            packager.package(self.root, self.out)
        self.study["slurm_job_id"] = "391"
        self.study["design"]["optimizer_updates"] = 768
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "formal CUDA"):
            packager.package(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_changed_weights_history_or_diagnostics_are_detected(self):
        prefix = self.root / "study/cells" / cell_name(0.001, 0.0, 7)
        for filename, message in (
            ("model.pt", "Checkpoint changed"),
            ("history.json", "Changed history"),
            ("visual_diagnostics.json", "Changed visual_diagnostics"),
        ):
            with self.subTest(filename=filename):
                path = prefix / filename
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                with self.assertRaisesRegex(ValueError, message):
                    packager.package(self.root, self.out)
                path.write_bytes(before)
        self.assertFalse(self.out.exists())

    def test_missing_diagnostic_nonfinite_and_symlink_evidence_are_rejected(self):
        relative = f"study/cells/{cell_name(0.001, 0.0, 7)}/visual_diagnostics.json"
        path = self.root / relative
        before = path.read_bytes()
        data = json.loads(before)
        data["snapshots"][0]["validation"]["visual_position_rmse_m"] = float("nan")
        self.write(relative, data)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            packager.package(self.root, self.out)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or non-regular"):
            packager.package(self.root, self.out)
        original = self.root / "original.json"
        original.write_bytes(before)
        path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "non-regular"):
            packager.package(self.root, self.out)

    def test_changed_initialization_or_cadence_is_not_hidden_by_recomputed_hash(self):
        name = cell_name(0.001, 1.0, 7)
        relative = f"study/cells/{name}/manifest.json"
        manifest = json.loads((self.root / relative).read_text())
        manifest["initial_visual_parameters_sha256"] = "e" * 64
        cell = self.study["cells"][name]
        cell["manifest_sha256"] = self.write(relative, manifest)
        cell["initial_visual_parameters_sha256"] = "e" * 64
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "Initial weights differ"):
            packager.package(self.root, self.out)

    def test_partial_publication_is_preserved(self):
        partial = self.out.with_suffix(self.out.suffix + ".writing")
        partial.write_bytes(b"interrupted")
        with self.assertRaisesRegex(ValueError, "staging exists"):
            packager.package(self.root, self.out)
        self.assertEqual(partial.read_bytes(), b"interrupted")


if __name__ == "__main__":
    unittest.main()
