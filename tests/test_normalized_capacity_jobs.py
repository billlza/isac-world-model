"""Exercise Phase C transport and packaging with mocks and synthetic evidence.

No job is submitted and no GPU is needed. Formal-looking JSON below is test
data for protocol rejection paths, not evidence that a scientific run occurred.
"""

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
from normalized_capacity_study import (
    LEARNING_RATE,
    PARAMETERS_BY_WIDTH,
    VISION_WEIGHT,
    cell_name,
    cells,
    design,
    summarize,
    validate_training_evidence,
)
from test_normalized_capacity_study import evidence_fixture, numerical_fixture

PROJECT = Path(__file__).resolve().parents[1]
PIN = "d" * 40
SPEC = importlib.util.spec_from_file_location(
    "normalized_capacity_package", PROJECT / "scripts/package_normalized_capacity_study.py"
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def tool(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)


def process_output(completed):
    return f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


class TestNormalizedCapacityJobs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        # Exercise real quoting with a home path containing spaces; never alter HOME.
        self.remote_root = self.tmp / "hamilton home"
        self.remote_root.mkdir()
        self.root = self.remote_root / f"isac-scale-{PIN[:12]}"
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
 rev-parse/refs/remotes/origin/research/normalized-capacity-h200) echo "${MOCK_REF_PIN:-$TEST_PIN}";;
 bundle/create)
   [[ $# == 4 && "$4" == refs/remotes/origin/research/normalized-capacity-h200 ]]
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
[[ "$1 $2 $3" == "--parsable --export=ALL scripts/hamilton_normalized_capacity.sh" ]]
[[ "$ISAC_SCALE_PIN" == "$TEST_PIN" && "$ISAC_PROJECT_ROOT" == "$PWD" ]]
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
echo "$name" >> "$TEST_HISTORY"
[[ "$name" != "${MOCK_FAIL_AT:-}" ]] || exit 17
while [[ "$1" != --out ]]; do shift; done
dest=$2
if [[ "$name" == normalized_capacity_study.py ]]; then
    [[ "$3 $4 $5 $6" == "--device cuda --profile formal" ]]
    mkdir "$dest"; dest="$dest/study.json"
fi
echo '{"status":"completed"}' > "$dest"
if [[ "$name" == "${MOCK_MUTATE_AT:-}" ]]; then
    touch "$TEST_REMOTE_ROOT/source-mutated"
fi""",
        )

    def submit(self, **overrides):
        shell = (PROJECT / "scripts/submit_normalized_capacity_from_mac.sh").read_text()
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
            "ISAC_SCALE_PIN": PIN,
            "ISAC_PROJECT_ROOT": str(self.root),
            "ISAC_RUNS_DIR": str(self.root / "runs"),
            "ISAC_PYTHON": str(self.fakepy),
        }
        env.pop("SLURM_ARRAY_JOB_ID", None)
        return subprocess.run(
            ["bash", str(PROJECT / "scripts/hamilton_normalized_capacity.sh")],
            env=env | overrides,
            text=True,
            capture_output=True,
        )

    def test_full_bundle_submits_once_without_previous_checkouts_and_clears_overrides(self):
        first = self.submit(SBATCH_PARTITION="wrong", SBATCH_GRES="gpu:1g.10gb:1", SBATCH_TIME="1")
        self.assertEqual(first.returncode, 0, process_output(first))
        self.assertIn("Slurm job: 391", first.stdout)
        self.assertIn('scale_backup="$HOME/Downloads/isac-scale-391"', first.stdout)
        self.assertIn("status.json", first.stdout)
        self.assertIn("hamilton\\ home", first.stdout)
        bundle_args = (self.tmp / "bundle-args").read_text().splitlines()
        self.assertEqual(len(bundle_args), 4)
        self.assertFalse(any(argument.startswith("^") for argument in bundle_args))
        second = self.submit()
        self.assertEqual(second.returncode, 0, process_output(second))
        self.assertIn("Already submitted", second.stdout)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])
        claim = self.root / ".scale-submission"
        self.assertEqual((claim / "job-id").read_text().strip(), "391")
        self.assertEqual((claim / "git-commit").read_text().strip(), PIN)
        self.assertEqual((claim / "stage").read_text().strip(), "submitted")
        self.assertEqual(json.loads((claim / "request.json").read_text())["profile"], "formal")
        self.assertEqual(
            json.loads((claim / "request.json").read_text())["study"], "normalized-capacity-phase-c"
        )

    def test_scheduler_error_is_preserved_without_duplicate_submission(self):
        failed = self.submit(MOCK_SBATCH_FAIL="1")
        self.assertNotEqual(failed.returncode, 0, process_output(failed))
        claim = self.root / ".scale-submission"
        self.assertTrue((claim / "receipt.pending").is_file())
        self.assertTrue((claim / "submission.stderr").is_file())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertIn("claim exists without a confirmed receipt", retried.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_malformed_receipt_is_retained_and_not_retried(self):
        malformed = self.submit(MOCK_RECEIPT="warning\n391;ham")
        self.assertNotEqual(malformed.returncode, 0, process_output(malformed))
        pending = self.root / ".scale-submission/receipt.pending"
        self.assertEqual(pending.read_text().strip(), "warning\n391;ham")
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_claim_revision_mismatch_does_not_reuse_receipt(self):
        self.assertEqual(self.submit().returncode, 0)
        (self.root / ".scale-submission/git-commit").write_text("e" * 40)
        rejected = self.submit()
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        self.assertIn("different revision", rejected.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def assert_submit_rejected(self, **overrides):
        rejected = self.submit(**overrides)
        diagnostic = process_output(rejected)
        self.assertNotEqual(rejected.returncode, 0, diagnostic)
        self.assertFalse((self.tmp / "history").exists(), diagnostic)
        self.assertFalse((self.root / ".scale-submission").exists(), diagnostic)

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
        run = self.compute("392", MOCK_FAIL_AT="normalized_capacity_study.py")
        self.assertEqual(run.returncode, 17, process_output(run))
        out = self.root / "runs/scale-392"
        before = (out / "status.json").read_bytes()
        status = json.loads(before)
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("failed", "study", 17))
        self.assertTrue((out / "preflight.json").is_file())
        self.assertFalse((out / "isac-scale-392.tar.gz").exists())
        retried = self.compute("392")
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((out / "status.json").read_bytes(), before)

    def test_complete_job_and_packaging_failure_have_distinct_status(self):
        completed = self.compute("393")
        self.assertEqual(completed.returncode, 0, process_output(completed))
        status = json.loads((self.root / "runs/scale-393/status.json").read_text())
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("completed", "done", 0))
        self.assertEqual(
            (self.tmp / "history").read_text().splitlines(),
            ["hamilton_preflight.py", "normalized_capacity_study.py", "package_normalized_capacity_study.py"],
        )
        failed = self.compute("394", MOCK_FAIL_AT="package_normalized_capacity_study.py")
        self.assertEqual(failed.returncode, 17, process_output(failed))
        out = self.root / "runs/scale-394"
        self.assertTrue((out / "study/study.json").is_file())
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "package"))

    def test_compute_source_guard_runs_before_training(self):
        rejected = self.compute("395", MOCK_CHECKOUT_PIN="e" * 40)
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        status = json.loads((self.root / "runs/scale-395/status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "preflight"))
        self.assertFalse((self.tmp / "history").exists())

    def test_source_change_after_training_blocks_package(self):
        rejected = self.compute("396", MOCK_MUTATE_AT="normalized_capacity_study.py")
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        out = self.root / "runs/scale-396"
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "verify-source"))
        self.assertTrue((out / "study/study.json").is_file())
        self.assertFalse((out / "isac-scale-396.tar.gz").exists())

    def test_source_change_during_packaging_never_marks_job_completed(self):
        rejected = self.compute("397", MOCK_MUTATE_AT="package_normalized_capacity_study.py")
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        out = self.root / "runs/scale-397"
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "verify-source-final"))
        self.assertTrue((out / "isac-scale-397.tar.gz").is_file())

    def test_array_job_cannot_start_formal_study(self):
        rejected = self.compute("398", SLURM_ARRAY_JOB_ID="398")
        self.assertEqual(rejected.returncode, 2, process_output(rejected))
        self.assertFalse((self.root / "runs/scale-398").exists())

    def test_formal_allocation_retains_one_whole_h200_and_four_hour_limit(self):
        script = (PROJECT / "scripts/hamilton_normalized_capacity.sh").read_text()
        options = {
            line.split("=", 1)[0].removeprefix("#SBATCH --"): line.split("=", 1)[1]
            for line in script.splitlines()
            if line.startswith("#SBATCH --")
        }
        self.assertEqual(options["partition"], "cuda")
        self.assertEqual(options["gres"], "gpu:h200_nvl:1")
        self.assertEqual(options["time"], "04:00:00")
        self.assertEqual(options["job-name"], "isac-scale")


class TestNormalizedCapacityPackage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "isac-scale-391.tar.gz"
        self.spec = design("formal")
        self.study = {
            "status": "completed",
            "profile": "formal",
            "git_commit": PIN,
            "slurm_job_id": "391",
            "runtime": dict(packager.SCHOOL_RUNTIME),
            "numerical_settings": numerical_fixture(),
            "design": self.spec,
            "scene": self.spec["scene"].copy(),
            "fresh_test_consumed": False,
            "test_episode_seeds": [],
            "cell_order": [cell_name(*key) for key in cells(self.spec)],
            "cells": {},
        }
        self.write(
            "preflight.json",
            {
                "status": "passed",
                "git_commit": PIN,
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
                "slurm_job_id": "391",
                "git_commit": PIN,
                "exit_code": None,
            },
        )
        for key in cells(self.spec):
            name = cell_name(*key)
            prefix = f"study/cells/{name}"
            evidence = evidence_fixture(self.spec, key)
            evidence[0]["runtime"] = dict(packager.SCHOOL_RUNTIME)
            hashes = {
                f"{item}_sha256": self.write(f"{prefix}/{item}.json", value)
                for item, value in zip(
                    ("manifest", "history", "epoch_history", "visual_diagnostics"), evidence, strict=True
                )
            }
            # Opaque bytes test integrity only; these are not learned checkpoint tensors.
            weights = f"synthetic-protocol-fixture-{name}".encode()
            checkpoint = self.root / prefix / "model.pt"
            checkpoint.write_bytes(weights)
            checksum = hashlib.sha256(weights).hexdigest()
            checkpoint.with_suffix(".pt.sha256").write_text(checksum + "\n")
            manifest = evidence[0]
            self.study["cells"][name] = {
                "status": "completed",
                "hidden_width": key[0],
                "latent_state_norm": "layernorm",
                "learning_rate": LEARNING_RATE,
                "vision_loss_weight": VISION_WEIGHT,
                "initialization": key[1],
                "parameters": manifest["parameters"],
                "visual_encoder_parameters": manifest["visual_encoder_parameters"],
                "initial_parameters_sha256": manifest["initial_parameters_sha256"],
                "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
                "numerical_settings": numerical_fixture(),
                "checkpoint": f"cells/{name}/model.pt",
                "checkpoint_sha256": checksum,
                "checkpoint_metadata_validated": True,
                "checkpoint_weights_validated": True,
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

    def update_evidence(self, name, item, value, *, metrics=False, summary=True):
        self.study["cells"][name][f"{item}_sha256"] = self.write(f"study/cells/{name}/{item}.json", value)
        if metrics:
            key = next(key for key in cells(self.spec) if cell_name(*key) == name)
            evidence = [
                json.loads((self.root / f"study/cells/{name}/{item}.json").read_text())
                for item in ("manifest", "history", "epoch_history", "visual_diagnostics")
            ]
            self.study["cells"][name]["metrics"] = validate_training_evidence(
                *evidence, self.spec, key, "cuda"
            )
            if summary:
                self.study["summary"] = summarize(self.study["cells"], self.spec)
        self.write("study/study.json", self.study)

    def test_complete_bundle_has_six_cells_hashes_and_checkpoint_identities(self):
        result = packager.package(self.root, self.out)
        self.assertEqual(result["status"], "packaged")
        self.assertFalse(result["weights_included"])
        self.assertEqual(result["sha256"], hashlib.sha256(self.out.read_bytes()).hexdigest())
        with tarfile.open(self.out) as archive:
            names = archive.getnames()
            self.assertEqual(len([name for name in names if name.endswith("/visual_diagnostics.json")]), 6)
            self.assertEqual(len([name for name in names if name.endswith("/history.json")]), 6)
            self.assertFalse(any(name.endswith(".pt") or "fresh_test" in name for name in names))
            self.assertTrue(all(name.endswith((".json", ".sha256")) for name in names))
            bundle = json.load(archive.extractfile("bundle_manifest.json"))
            self.assertFalse(bundle["weights_included"])
            self.assertFalse(bundle["fresh_test_consumed"])
            self.assertEqual(len(bundle["checkpoints"]), 6)
            self.assertEqual(bundle["kind"], "isac-normalized-capacity-phase-c")
            for name, checkpoint in bundle["checkpoints"].items():
                self.assertEqual(
                    checkpoint["best_update"], self.study["cells"][name]["metrics"]["best_update"]
                )
                self.assertEqual(
                    checkpoint["latent_state_norm"], self.study["cells"][name]["latent_state_norm"]
                )
                self.assertEqual(checkpoint["hidden_width"], self.study["cells"][name]["hidden_width"])
                self.assertEqual(checkpoint["parameters"], PARAMETERS_BY_WIDTH[checkpoint["hidden_width"]])
                self.assertTrue(checkpoint["checkpoint_weights_validated"])
            for name, meta in bundle["artifacts"].items():
                raw = archive.extractfile(name).read()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"])
                self.assertEqual(len(raw), meta["bytes"])
            status = json.load(archive.extractfile("job_status_at_packaging.json"))
            self.assertEqual((status["status"], status["stage"]), ("running", "package"))
            readout = json.load(archive.extractfile("readout.json"))
            self.assertEqual(readout["phase"], "C")
            self.assertEqual(readout["numerical_settings"], numerical_fixture())
            self.assertEqual(set(readout["summary"]["resources_by_width"]), {"512", "1024", "2048"})
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            packager.package(self.root, self.out)
        self.assertEqual(before, self.out.read_bytes())

    def test_missing_cell_wrong_identity_or_changed_design_cannot_package(self):
        name = cell_name(2048, 11)
        cell = self.study["cells"].pop(name)
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "formal normalized-capacity cell"):
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

    def test_development_or_fresh_test_consumption_cannot_masquerade_as_formal(self):
        self.study["profile"] = "development"
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "formal CUDA"):
            packager.package(self.root, self.out)
        self.study["profile"] = "formal"
        self.study["fresh_test_consumed"] = True
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "without test consumption"):
            packager.package(self.root, self.out)

    def test_changed_weights_history_or_diagnostics_are_detected(self):
        prefix = self.root / "study/cells" / cell_name(512, 7)
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

    def test_missing_nonfinite_and_symlink_evidence_are_rejected(self):
        relative = f"study/cells/{cell_name(512, 7)}/visual_diagnostics.json"
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

    def test_changed_initial_weights_are_not_hidden_by_recomputed_hash(self):
        name = cell_name(2048, 7)
        relative = f"study/cells/{name}/manifest.json"
        manifest = json.loads((self.root / relative).read_text())
        manifest["initial_visual_parameters_sha256"] = "e" * 64
        self.study["cells"][name]["initial_visual_parameters_sha256"] = "e" * 64
        self.update_evidence(name, "manifest", manifest)
        with self.assertRaisesRegex(ValueError, "visual initialization differs across widths"):
            packager.package(self.root, self.out)

    def test_runtime_drift_is_rejected_even_with_updated_manifest_hash(self):
        name = cell_name(512, 7)
        manifest = json.loads((self.root / f"study/cells/{name}/manifest.json").read_text())
        manifest["runtime"]["torch"] = "different"
        self.update_evidence(name, "manifest", manifest)
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            packager.package(self.root, self.out)

    def test_numerical_execution_drift_is_rejected(self):
        name = cell_name(1024, 7)
        self.study["cells"][name]["numerical_settings"]["float32_matmul_precision"] = "medium"
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "numerical settings differ"):
            packager.package(self.root, self.out)

    def test_recorded_common_numeric_policy_is_not_replaced_by_a_new_default(self):
        self.study["numerical_settings"]["deterministic_algorithms"] = True
        for cell in self.study["cells"].values():
            cell["numerical_settings"]["deterministic_algorithms"] = True
        self.write("study/study.json", self.study)
        result = packager.package(self.root, self.out)
        self.assertEqual(result["status"], "packaged")
        with tarfile.open(self.out) as archive:
            readout = json.load(archive.extractfile("readout.json"))
            self.assertTrue(readout["numerical_settings"]["deterministic_algorithms"])

    def test_different_widths_cannot_share_a_full_model_fingerprint(self):
        name = cell_name(2048, 7)
        manifest = json.loads((self.root / f"study/cells/{name}/manifest.json").read_text())
        fingerprint = self.study["cells"][cell_name(512, 7)]["initial_parameters_sha256"]
        manifest["initial_parameters_sha256"] = fingerprint
        self.study["cells"][name]["initial_parameters_sha256"] = fingerprint
        self.update_evidence(name, "manifest", manifest)
        with self.assertRaisesRegex(ValueError, "identical full-model fingerprints"):
            packager.package(self.root, self.out)

    def test_wider_cell_cannot_report_smaller_parameter_count(self):
        name = cell_name(2048, 11)
        manifest = json.loads((self.root / f"study/cells/{name}/manifest.json").read_text())
        manifest["parameters"] = PARAMETERS_BY_WIDTH[512]
        self.update_evidence(name, "manifest", manifest)
        with self.assertRaisesRegex(ValueError, "parameters"):
            packager.package(self.root, self.out)

    def test_unvalidated_checkpoint_tensors_cannot_be_packaged(self):
        self.study["cells"][cell_name(2048, 11)]["checkpoint_weights_validated"] = False
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "Cell protocol differs"):
            packager.package(self.root, self.out)

    def test_normalization_mislabel_and_best_checkpoint_mismatch_are_rejected(self):
        name = cell_name(2048, 7)
        self.study["cells"][name]["checkpoint_metadata_validated"] = False
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "Cell protocol differs"):
            packager.package(self.root, self.out)
        self.study["cells"][name]["checkpoint_metadata_validated"] = True
        self.study["cells"][name]["latent_state_norm"] = "none"
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "Cell protocol differs"):
            packager.package(self.root, self.out)
        self.study["cells"][name]["latent_state_norm"] = "layernorm"
        manifest = json.loads((self.root / f"study/cells/{name}/manifest.json").read_text())
        manifest["best_update"] = 1
        self.update_evidence(name, "manifest", manifest)
        with self.assertRaisesRegex(ValueError, "checkpoint|selection"):
            packager.package(self.root, self.out)

    def test_changed_diagnostic_cadence_is_rejected_after_hash_and_summary_update(self):
        name = cell_name(2048, 7)
        diagnostics = json.loads((self.root / f"study/cells/{name}/visual_diagnostics.json").read_text())
        for snapshot in diagnostics["snapshots"]:
            snapshot["validation"]["event_counts"]["vision_new_valid"] += 1
        self.update_evidence(name, "visual_diagnostics", diagnostics, metrics=True, summary=False)
        with self.assertRaisesRegex(ValueError, "Diagnostic event cadence differs across paired cells"):
            packager.package(self.root, self.out)

    def test_failed_terminal_status_and_mig_gpu_cannot_package(self):
        self.write(
            "status.json",
            {
                "status": "failed",
                "stage": "package",
                "slurm_job_id": "391",
                "git_commit": PIN,
                "exit_code": 1,
            },
        )
        with self.assertRaisesRegex(ValueError, "successful study packaging"):
            packager.package(self.root, self.out)
        self.write(
            "status.json",
            {
                "status": "completed",
                "stage": "done",
                "slurm_job_id": "391",
                "git_commit": PIN,
                "exit_code": 0,
            },
        )
        preflight = json.loads((self.root / "preflight.json").read_text())
        preflight["gpu"]["mig_mode"] = "Enabled"
        self.write("preflight.json", preflight)
        with self.assertRaisesRegex(ValueError, "Whole H200"):
            packager.package(self.root, self.out)

    def test_partial_publication_is_preserved(self):
        partial = self.out.with_suffix(self.out.suffix + ".writing")
        partial.write_bytes(b"interrupted")
        with self.assertRaisesRegex(ValueError, "staging exists"):
            packager.package(self.root, self.out)
        self.assertEqual(partial.read_bytes(), b"interrupted")


if __name__ == "__main__":
    unittest.main()
