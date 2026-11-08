"""Exercise Phase D job transport using mocks; never submit a real scheduler job."""

import copy
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
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
PIN = "d" * 40
HAS_TORCH = importlib.util.find_spec("torch") is not None


def tool(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)


def process_output(completed):
    return f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


class TestWidthOptimizationJobs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        # Exercise real quoting with a home path containing spaces; never alter HOME.
        self.remote_root = self.tmp / "hamilton home"
        self.remote_root.mkdir()
        self.root = self.remote_root / f"isac-optim-{PIN[:12]}"
        self.historical = self.remote_root / "isac-scale-249407dbbe40/runs/scale-19404810"
        self.historical.mkdir(parents=True)
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
 rev-parse/refs/remotes/origin/research/width-optimization-h200) echo "${MOCK_REF_PIN:-$TEST_PIN}";;
 bundle/create)
   [[ $# == 4 && "$4" == refs/remotes/origin/research/width-optimization-h200 ]]
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
        tool(
            self.bin / "scp",
            """
destination="$TEST_REMOTE_ROOT/${2#*:}"
cp "$1" "$destination"
if [[ -n "${MOCK_TRANSFER_CORRUPT:-}" ]]; then echo corrupt >> "$destination"; fi
""",
        )
        tool(self.bin / "shasum", '[[ "$1 $2" == "-a 256" ]]; exec sha256sum "$3"')
        tool(self.bin / "ssh", 'shift; read -r -a args <<< "${1#*-- }"; exec bash -s -- "${args[@]}"')
        tool(self.bin / "module", '[[ "$1 $2" == "load python/3.13.9cuda" ]]')
        tool(
            self.bin / "sbatch",
            """echo submit >> "$TEST_HISTORY"
[[ "$1 $2 $3" == "--parsable --export=ALL scripts/hamilton_width_optimization.sh" ]]
[[ "$ISAC_OPTIM_PIN" == "$TEST_PIN" && "$ISAC_PROJECT_ROOT" == "$PWD" ]]
[[ "$ISAC_RUNS_DIR" == "$PWD/runs" ]]
[[ "$ISAC_OPTIM_HISTORICAL_RUN" == "$TEST_REMOTE_ROOT/isac-scale-249407dbbe40/runs/scale-19404810" ]]
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
    if [[ "${2:-}" == */isac-scale-249407dbbe40/runs/scale-19404810 ]]; then
        [[ -z "${MOCK_HISTORY_FAIL:-}" ]] || exit 20
    fi
    exit 0
fi
name=${1##*/}
shift
echo "$name" >> "$TEST_HISTORY"
[[ "$name" != "${MOCK_FAIL_AT:-}" ]] || exit 17
while [[ "$1" != --out ]]; do shift; done
dest=$2
if [[ "$name" == width_optimization_study.py ]]; then
    [[ "$3 $4 $5 $6" == "--device cuda --profile formal" ]]
    [[ "$7" == --historical-run && "$8" == "$ISAC_OPTIM_HISTORICAL_RUN" ]]
    mkdir "$dest"; dest="$dest/study.json"
fi
echo '{"status":"completed"}' > "$dest"
if [[ "$name" == "${MOCK_MUTATE_AT:-}" ]]; then
    touch "$TEST_REMOTE_ROOT/source-mutated"
fi""",
        )

    def submit(self, **overrides):
        shell = (PROJECT / "scripts/submit_width_optimization_from_mac.sh").read_text()
        placeholder = "/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python"
        self.assertEqual(shell.count(placeholder), 1)
        shell = shell.replace(placeholder, str(self.fakepy))
        for variable in ("root", "bundle", "historical_run"):
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
            "ISAC_OPTIM_PIN": PIN,
            "ISAC_PROJECT_ROOT": str(self.root),
            "ISAC_RUNS_DIR": str(self.root / "runs"),
            "ISAC_PYTHON": str(self.fakepy),
            "ISAC_OPTIM_HISTORICAL_RUN": str(self.historical),
        }
        env.pop("SLURM_ARRAY_JOB_ID", None)
        return subprocess.run(
            ["bash", str(PROJECT / "scripts/hamilton_width_optimization.sh")],
            env=env | overrides,
            text=True,
            capture_output=True,
        )

    def test_full_bundle_submits_once_without_previous_checkouts_and_clears_overrides(self):
        first = self.submit(SBATCH_PARTITION="wrong", SBATCH_GRES="gpu:1g.10gb:1", SBATCH_TIME="1")
        self.assertEqual(first.returncode, 0, process_output(first))
        self.assertIn("Slurm job: 391", first.stdout)
        self.assertIn('optim_backup="$HOME/Downloads/isac-optim-391"', first.stdout)
        self.assertIn("status.json", first.stdout)
        self.assertIn("hamilton\\ home", first.stdout)
        bundle_args = (self.tmp / "bundle-args").read_text().splitlines()
        self.assertEqual(len(bundle_args), 4)
        self.assertFalse(any(argument.startswith("^") for argument in bundle_args))
        second = self.submit()
        self.assertEqual(second.returncode, 0, process_output(second))
        self.assertIn("Already submitted", second.stdout)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])
        claim = self.root / ".optim-submission"
        self.assertEqual((claim / "job-id").read_text().strip(), "391")
        self.assertEqual((claim / "git-commit").read_text().strip(), PIN)
        self.assertEqual((claim / "stage").read_text().strip(), "submitted")
        self.assertEqual(
            (claim / "receipt.sha256").read_text().strip(),
            hashlib.sha256((claim / "receipt").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            (claim / "bundle.sha256").read_text().strip(), hashlib.sha256(b"bundle\n").hexdigest()
        )
        self.assertEqual(json.loads((claim / "request.json").read_text())["profile"], "formal")
        self.assertEqual(
            json.loads((claim / "request.json").read_text())["study"], "width-optimization-phase-d"
        )

    def test_scheduler_error_is_preserved_without_duplicate_submission(self):
        failed = self.submit(MOCK_SBATCH_FAIL="1")
        self.assertNotEqual(failed.returncode, 0, process_output(failed))
        claim = self.root / ".optim-submission"
        self.assertTrue((claim / "receipt.pending").is_file())
        self.assertTrue((claim / "submission.stderr").is_file())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertIn("claim exists without a confirmed receipt", retried.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_malformed_receipt_is_retained_and_not_retried(self):
        malformed = self.submit(MOCK_RECEIPT="warning\n391;ham")
        self.assertNotEqual(malformed.returncode, 0, process_output(malformed))
        pending = self.root / ".optim-submission/receipt.pending"
        self.assertEqual(pending.read_text().strip(), "warning\n391;ham")
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_claim_revision_mismatch_does_not_reuse_receipt(self):
        self.assertEqual(self.submit().returncode, 0)
        (self.root / ".optim-submission/git-commit").write_text("e" * 40)
        rejected = self.submit()
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        self.assertIn("different revision", rejected.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def assert_submit_rejected(self, **overrides):
        rejected = self.submit(**overrides)
        diagnostic = process_output(rejected)
        self.assertNotEqual(rejected.returncode, 0, diagnostic)
        self.assertFalse((self.tmp / "history").exists(), diagnostic)
        self.assertFalse((self.root / ".optim-submission").exists(), diagnostic)

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

    def test_invalid_historical_source_cannot_submit(self):
        self.assert_submit_rejected(MOCK_HISTORY_FAIL="1")

    def test_changed_transferred_bundle_cannot_submit(self):
        self.assert_submit_rejected(MOCK_TRANSFER_CORRUPT="1")

    def test_receipt_tampering_does_not_submit_or_reuse_it(self):
        self.assertEqual(self.submit().returncode, 0)
        (self.root / ".optim-submission/receipt").write_text("777;ham\n")
        rejected = self.submit()
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        self.assertIn("receipt checksum differs", rejected.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_incomplete_checksum_publication_keeps_claim(self):
        self.assertEqual(self.submit().returncode, 0)
        (self.root / ".optim-submission/receipt.sha256").unlink()
        rejected = self.submit()
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        self.assertIn("checksum is missing", rejected.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_symlink_checkout_cannot_submit(self):
        other = self.tmp / "other-checkout"
        other.mkdir()
        self.root.symlink_to(other, target_is_directory=True)
        self.assert_submit_rejected()
        self.assertTrue(self.root.is_symlink())

    def test_study_failure_preserves_outputs_and_cannot_overwrite_them(self):
        run = self.compute("392", MOCK_FAIL_AT="width_optimization_study.py")
        self.assertEqual(run.returncode, 17, process_output(run))
        out = self.root / "runs/optim-392"
        before = (out / "status.json").read_bytes()
        status = json.loads(before)
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("failed", "study", 17))
        self.assertTrue((out / "preflight.json").is_file())
        self.assertFalse((out / "isac-optim-392.tar.gz").exists())
        retried = self.compute("392")
        self.assertNotEqual(retried.returncode, 0, process_output(retried))
        self.assertEqual((out / "status.json").read_bytes(), before)

    def test_complete_job_and_packaging_failure_have_distinct_status(self):
        completed = self.compute("393")
        self.assertEqual(completed.returncode, 0, process_output(completed))
        status = json.loads((self.root / "runs/optim-393/status.json").read_text())
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("completed", "done", 0))
        self.assertEqual(
            (self.tmp / "history").read_text().splitlines(),
            ["hamilton_preflight.py", "width_optimization_study.py", "package_width_optimization_study.py"],
        )
        failed = self.compute("394", MOCK_FAIL_AT="package_width_optimization_study.py")
        self.assertEqual(failed.returncode, 17, process_output(failed))
        out = self.root / "runs/optim-394"
        self.assertTrue((out / "study/study.json").is_file())
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "package"))

    def test_compute_source_guard_runs_before_training(self):
        rejected = self.compute("395", MOCK_CHECKOUT_PIN="e" * 40)
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        status = json.loads((self.root / "runs/optim-395/status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "preflight"))
        self.assertFalse((self.tmp / "history").exists())

    def test_source_change_after_training_blocks_package(self):
        rejected = self.compute("396", MOCK_MUTATE_AT="width_optimization_study.py")
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        out = self.root / "runs/optim-396"
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "verify-source"))
        self.assertTrue((out / "study/study.json").is_file())
        self.assertFalse((out / "isac-optim-396.tar.gz").exists())

    def test_source_change_during_packaging_never_marks_job_completed(self):
        rejected = self.compute("397", MOCK_MUTATE_AT="package_width_optimization_study.py")
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        out = self.root / "runs/optim-397"
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "verify-source-final"))
        self.assertTrue((out / "isac-optim-397.tar.gz").is_file())

    def test_array_job_cannot_start_formal_study(self):
        rejected = self.compute("398", SLURM_ARRAY_JOB_ID="398")
        self.assertEqual(rejected.returncode, 2, process_output(rejected))
        self.assertFalse((self.root / "runs/optim-398").exists())

    def test_missing_historical_run_preserves_failed_status_without_training(self):
        self.historical.rmdir()
        rejected = self.compute("399")
        self.assertNotEqual(rejected.returncode, 0, process_output(rejected))
        status = json.loads((self.root / "runs/optim-399/status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "preflight"))
        self.assertFalse((self.tmp / "history").exists())

    def test_formal_allocation_retains_one_whole_h200_and_four_hour_limit(self):
        script = (PROJECT / "scripts/hamilton_width_optimization.sh").read_text()
        options = {
            line.split("=", 1)[0].removeprefix("#SBATCH --"): line.split("=", 1)[1]
            for line in script.splitlines()
            if line.startswith("#SBATCH --")
        }
        self.assertEqual(options["partition"], "cuda")
        self.assertEqual(options["gres"], "gpu:h200_nvl:1")
        self.assertEqual(options["time"], "04:00:00")
        self.assertEqual(options["job-name"], "isac-optim")


@unittest.skipUnless(HAS_TORCH, "Bounded checkpoint artifact tests require PyTorch")
class TestWidthOptimizationPackage(unittest.TestCase):
    """Synthetic protocol fixtures exercise real hashes, bounded readers and states.

    Only the registered large tensor shape validator is mocked. Small real Torch
    tensors make best/current substitution tests inexpensive; these values are
    deliberately not presented as research training or GPU execution evidence.
    """

    def setUp(self):
        sys.path.insert(0, str(PROJECT / "scripts"))
        import package_width_optimization_study as packager
        import width_optimization_study as study
        from test_checkpoint_replay import replay_fixture
        from test_normalized_capacity_study import (
            evidence_fixture,
            numerical_fixture,
            selected_checkpoint_fixture,
        )

        from isac_wm.artifacts import save_checkpoint, save_resume_checkpoint
        from isac_wm.checkpoint_replay import state_dict_fingerprint

        self.packager, self.study = packager, study
        self.save_checkpoint, self.save_resume_checkpoint = save_checkpoint, save_resume_checkpoint
        self.fingerprint, self.replay_fixture = state_dict_fingerprint, replay_fixture
        self.evidence_fixture = evidence_fixture
        self.selected_fixture = selected_checkpoint_fixture
        self.numerical_fixture = numerical_fixture
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.root, self.old = self.tmp / "new-run", self.tmp / "old-run"
        self.root.mkdir()
        self.old.mkdir()
        self.out = self.root / "isac-optim-391.tar.gz"
        guard = patch.object(study.phase_c, "validate_checkpoint_weights", return_value=None)
        guard.start()
        self.addCleanup(guard.stop)
        historical = self.historical_fixture()
        self.spec = study.design("formal")
        self.report = {
            "schema_version": 1,
            "status": "completed",
            "profile": "formal",
            "git_commit": PIN,
            "slurm_job_id": "391",
            "runtime": dict(packager.SCHOOL_RUNTIME),
            "numerical_settings": numerical_fixture(),
            "design": self.spec,
            "scene": self.spec["scene"],
            "fresh_test_consumed": False,
            "test_episode_seeds": [],
            "deployment_validated": False,
            "historical_replay": historical,
            "cell_order": [study.cell_name(*key) for key in study.cells(self.spec)],
            "cells": {},
        }
        self.write(
            self.root / "preflight.json",
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
            self.root / "status.json",
            {
                "status": "running",
                "stage": "package",
                "slurm_job_id": "391",
                "git_commit": PIN,
                "exit_code": None,
            },
        )
        self.replay_reports = {}
        for key in study.cells(self.spec):
            width, scale, seed = key
            name = study.cell_name(*key)
            folder = self.root / "study/cells" / name
            evidence = self.evidence_fixture(self.spec, (width, seed))
            manifest = evidence[0]
            manifest.update(
                width_lr_scale=scale, optimizer_policy=study.expected_optimizer_policy(width, scale)
            )
            manifest["runtime"] = dict(packager.SCHOOL_RUNTIME)
            self.choose_midpoint(evidence)
            digests = {
                f"{kind}_sha256": self.write(folder / f"{kind}.json", value)
                for kind, value in zip(study.EVIDENCE_FILES, evidence, strict=True)
            }
            weights = self.checkpoints(folder, evidence, self.spec, (width, seed), policy=True)
            links, replays = self.make_replays(folder, evidence, weights, PIN)
            self.replay_reports[name] = replays
            self.report["cells"][name] = {
                "status": "completed",
                "hidden_width": width,
                "width_lr_scale": scale,
                "initialization": seed,
                "latent_state_norm": "layernorm",
                "learning_rate": study.LEARNING_RATE,
                "vision_loss_weight": study.VISION_WEIGHT,
                "parameters": manifest["parameters"],
                "visual_encoder_parameters": study.VISUAL_PARAMETERS,
                "initial_parameters_sha256": manifest["initial_parameters_sha256"],
                "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
                "numerical_settings": numerical_fixture(),
                "optimizer_policy": manifest["optimizer_policy"],
                "checkpoint": f"cells/{name}/model.pt",
                "checkpoint_sha256": weights["best"]["sha256"],
                "resume_checkpoint": f"cells/{name}/resume.pt",
                "resume_checkpoint_sha256": weights["final"]["sha256"],
                "checkpoint_metadata_validated": True,
                "checkpoint_weights_validated": True,
                "final_checkpoint_metadata_validated": True,
                "final_checkpoint_weights_validated": True,
                "metrics": study.validate_training_evidence(*evidence, self.spec, key, "cuda"),
                "replays": links,
                **digests,
            }
        self.report["summary"] = study.summarize(self.report["cells"], self.spec)
        self.report["replay_summary"] = study.summarize_replays(
            self.report["cells"], self.replay_reports, self.spec
        )
        self.write(self.root / "study/study.json", self.report)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def choose_midpoint(self, evidence):
        manifest, history, _, _ = evidence
        history[1]["validation_position_rmse_m"] = 0.001
        manifest["best_update"] = history[1]["optimizer_updates"]
        manifest["best_validation_rmse_m"] = 0.001

    def checkpoints(self, folder, evidence, spec, key, *, policy=False):
        import torch

        manifest, history, epoch_history, diagnostics = evidence
        selected = self.selected_fixture(manifest, spec, key)
        selected["state_dict"] = {"synthetic_fixture.weight": torch.tensor([1.0, 2.0])}
        if policy:
            selected.update(
                width_lr_scale=manifest["width_lr_scale"], optimizer_policy=manifest["optimizer_policy"]
            )
        final = {
            **copy.deepcopy(selected),
            "state_dict": {"synthetic_fixture.weight": torch.tensor([3.0, 4.0])},
            "optimizer": {"state": {}, "param_groups": []},
            "numpy_rng_json": "{}",
            "torch_rng": torch.tensor([0, 1], dtype=torch.uint8),
            "completed_epochs": manifest["completed_epochs"],
            "optimizer_updates": manifest["optimizer_updates"],
            "best_checkpoint": copy.deepcopy(selected),
            "best_rmse": manifest["best_validation_rmse_m"],
            "history": history,
            "epoch_history": epoch_history,
            "episodes": manifest["episodes"],
            "batch_size": manifest["batch_size"],
            "learning_rate": manifest["learning_rate"],
            "event_counts_seen": manifest["event_counts_seen"],
            "visual_diagnostics_snapshots": diagnostics["snapshots"],
        }
        folder.mkdir(parents=True, exist_ok=True)
        result = {}
        for kind, payload, name, save in (
            ("best", selected, "model.pt", self.save_checkpoint),
            ("final", final, "resume.pt", self.save_resume_checkpoint),
        ):
            path = folder / name
            save(path, payload)
            result[kind] = {
                "path": path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "state_fingerprint": self.fingerprint(payload["state_dict"]),
            }
        return result

    def make_replays(self, folder, evidence, weights, source_commit, *, historical=False):
        manifest, history = evidence[:2]
        manifest_sha = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
        links, replays = {}, {}
        for kind in ("best", "final"):
            report = self.replay_fixture(
                manifest,
                kind=kind,
                source_commit=source_commit,
                checkpoint_sha=weights[kind]["sha256"],
                manifest_sha=manifest_sha,
                state_fingerprint=weights[kind]["state_fingerprint"],
                final_score=history[-1]["validation_position_rmse_m"],
            )
            report.update(profile="formal", device="cuda", runtime={"device": "cuda"})
            if historical:
                path = self.root / "study/historical-replay" / f"{kind}.json"
            else:
                path = folder / f"replay-{kind}.json"
            sha = self.write(path, report)
            links[kind] = {"path": path.relative_to(self.root / "study").as_posix(), "sha256": sha}
            replays[kind] = report
        return links, replays

    def historical_fixture(self):
        study = self.study
        old_spec = study.phase_c.design("formal")
        evidence = self.evidence_fixture(old_spec, (512, 11))
        evidence[0]["runtime"] = dict(self.packager.SCHOOL_RUNTIME)
        self.choose_midpoint(evidence)
        prefix = self.old / "study/cells" / study.HISTORICAL_CELL
        hashes = {
            f"{name}_sha256": self.write(prefix / f"{name}.json", value)
            for name, value in zip(study.EVIDENCE_FILES, evidence, strict=True)
        }
        weights = self.checkpoints(prefix, evidence, old_spec, (512, 11))
        old_study = {
            "status": "completed",
            "profile": "formal",
            "git_commit": study.HISTORICAL_COMMIT,
            "slurm_job_id": study.HISTORICAL_JOB,
            "design": old_spec,
            "fresh_test_consumed": False,
            "test_episode_seeds": [],
            "cells": {
                study.HISTORICAL_CELL: {
                    "checkpoint_sha256": weights["best"]["sha256"],
                    **hashes,
                    "metrics": study.phase_c.validate_training_evidence(
                        *evidence, old_spec, (512, 11), "cuda"
                    ),
                }
            },
        }
        study_sha = self.write(self.old / "study/study.json", old_study)
        self.write(
            self.old / "status.json",
            {
                "status": "completed",
                "stage": "done",
                "exit_code": 0,
                "git_commit": study.HISTORICAL_COMMIT,
                "slurm_job_id": study.HISTORICAL_JOB,
            },
        )
        for field, value in {
            "HISTORICAL_STUDY_SHA256": study_sha,
            "HISTORICAL_MANIFEST_SHA256": hashes["manifest_sha256"],
            "HISTORICAL_CHECKPOINT_SHA256": weights["best"]["sha256"],
        }.items():
            substitution = patch.object(study, field, value)
            substitution.start()
            self.addCleanup(substitution.stop)
        source_files = {}
        for name in self.packager.HISTORICAL_FILES:
            original = self.old / (
                "study/study.json"
                if name == "study"
                else "status.json"
                if name == "status"
                else f"study/cells/{study.HISTORICAL_CELL}/{name}.json"
            )
            dest = self.root / "study/historical-replay" / f"source-{name}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = original.read_bytes()
            dest.write_bytes(raw)
            source_files[name] = {
                "path": dest.relative_to(self.root / "study").as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        links, _ = self.make_replays(prefix, evidence, weights, study.HISTORICAL_COMMIT, historical=True)
        return {
            "status": "completed",
            "source_commit": study.HISTORICAL_COMMIT,
            "source_job_id": study.HISTORICAL_JOB,
            "source_cell": study.HISTORICAL_CELL,
            "source_run": str(self.old),
            "evaluator_git_commit": PIN,
            "source_files": source_files,
            "reports": links,
            "source_evidence": {
                "study_sha256": study_sha,
                "manifest_sha256": hashes["manifest_sha256"],
                "checkpoint_sha256": weights["best"]["sha256"],
                "resume_sha256": weights["final"]["sha256"],
            },
            "source_verification": {
                "best_digest_origin": "previously audited Phase C archive",
                "final_digest_origin": "retained Hamilton sidecar first recorded by Phase D; no resume digest in Phase C archive",
                "final_protocol_and_weights_validated": True,
                "source_mutated": False,
                "optimizer_steps": 0,
                "fresh_test_consumed": False,
            },
        }

    def changed_replay(self, cell_name, kind, edit):
        link = self.report["cells"][cell_name]["replays"][kind]
        path = self.root / "study" / link["path"]
        value = json.loads(path.read_text())
        edit(value)
        link["sha256"] = self.write(path, value)
        self.write(self.root / "study/study.json", self.report)

    def test_complete_package_preserves_reports_and_sidecars_without_weights(self):
        extra = self.root / "study/additional-diagnostic.json"
        self.write(extra, {"scope": "synthetic extra diagnostic", "count": 1})
        old_hashes = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.old.rglob("*")
            if path.is_file()
        }
        result = self.packager.package(self.root, self.out)
        self.assertFalse(result["weights_included"])
        self.assertEqual(result["sha256"], hashlib.sha256(self.out.read_bytes()).hexdigest())
        self.assertEqual(
            old_hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_hashes}
        )
        with tarfile.open(self.out) as archive:
            names = archive.getnames()
            self.assertFalse(any(name.endswith(".pt") for name in names))
            self.assertIn("study/additional-diagnostic.json", names)
            bundle = json.load(archive.extractfile("bundle_manifest.json"))
            self.assertEqual(bundle["kind"], "isac-width-optimization-phase-d")
            self.assertEqual(len(bundle["checkpoints"]), 6)
            for cell in bundle["checkpoints"].values():
                self.assertEqual(set(cell), {"best", "final"})
                self.assertEqual(cell["best"]["weights_update"], 192)
                self.assertEqual(cell["final"]["weights_update"], 384)
                self.assertNotEqual(
                    cell["best"]["state_dict_fingerprint"], cell["final"]["state_dict_fingerprint"]
                )
            for name, meta in bundle["artifacts"].items():
                raw = archive.extractfile(name).read()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"])
                self.assertEqual(len(raw), meta["bytes"])
            readout = json.load(archive.extractfile("readout.json"))
            self.assertEqual(readout["phase"], "D")
            self.assertFalse(readout["fresh_test_consumed"])
            self.assertFalse(readout["deployment_validated"])
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.packager.package(self.root, self.out)
        self.assertEqual(self.out.read_bytes(), before)

    def test_final_replay_cannot_substitute_selected_weights_even_after_rehash(self):
        name = self.study.cell_name(512, 1.0, 7)
        selected = self.replay_reports[name]["best"]["state_dict_fingerprint"]
        self.changed_replay(name, "final", lambda value: value.update(state_dict_fingerprint=selected))
        with self.assertRaisesRegex(ValueError, "different final weights"):
            self.packager.package(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_final_score_and_checkpoint_update_cannot_be_relabelled(self):
        name = self.study.cell_name(512, 1.0, 7)
        self.changed_replay(name, "final", lambda value: value.update(weights_update=192))
        with self.assertRaisesRegex(ValueError, "weights_update"):
            self.packager.package(self.root, self.out)

        def relabel_score(value):
            value["weights_update"] = 384
            value["recorded_score"].update(
                expected_rmse_m=0.123,
                delta_m=-0.123,
                within_registered_tolerance=False,
            )

        self.changed_replay(name, "final", relabel_score)
        with self.assertRaisesRegex(ValueError, "recorded endpoint"):
            self.packager.package(self.root, self.out)

    def test_changed_checkpoint_bytes_and_replay_bytes_are_detected(self):
        name = self.study.cell_name(512, 1.0, 7)
        for relative, message in (
            (f"study/cells/{name}/resume.pt", "Checkpoint changed"),
            (f"study/cells/{name}/replay-final.json", "replay evidence changed"),
        ):
            with self.subTest(relative=relative):
                path = self.root / relative
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                with self.assertRaisesRegex(ValueError, message):
                    self.packager.package(self.root, self.out)
                path.write_bytes(before)

    def test_historical_source_change_or_mutation_claim_is_rejected(self):
        path = self.old / "study/study.json"
        before = path.read_bytes()
        path.write_bytes(before + b" ")
        with self.assertRaisesRegex(ValueError, "Historical source changed"):
            self.packager.package(self.root, self.out)
        path.write_bytes(before)
        self.report["historical_replay"]["source_verification"]["source_mutated"] = True
        self.write(self.root / "study/study.json", self.report)
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.packager.package(self.root, self.out)
        verification = self.report["historical_replay"]["source_verification"]
        verification["source_mutated"] = False
        verification["final_digest_origin"] = "previously audited Phase C archive"
        self.write(self.root / "study/study.json", self.report)
        with self.assertRaisesRegex(ValueError, "provenance is incomplete or overstated"):
            self.packager.package(self.root, self.out)

    def test_missing_cell_replay_policy_or_summary_cannot_package(self):
        name = self.study.cell_name(2048, 0.5, 11)
        cell = self.report["cells"].pop(name)
        self.write(self.root / "study/study.json", self.report)
        with self.assertRaisesRegex(ValueError, "formal Phase D cell"):
            self.packager.package(self.root, self.out)
        self.report["cells"][name] = cell
        cell["optimizer_policy"]["groups"][-1]["learning_rate"] = 0.0003
        self.write(self.root / "study/study.json", self.report)
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            self.packager.package(self.root, self.out)

    def test_cpu_replay_cannot_masquerade_as_formal_cuda_and_staging_is_preserved(self):
        name = self.study.cell_name(512, 1.0, 7)
        self.changed_replay(name, "best", lambda value: value.update(device="cpu"))
        with self.assertRaisesRegex(ValueError, "device"):
            self.packager.package(self.root, self.out)
        staging = self.out.with_suffix(self.out.suffix + ".writing")
        staging.write_bytes(b"interrupted-package")
        with self.assertRaisesRegex(ValueError, "staging exists"):
            self.packager.package(self.root, self.out)
        self.assertEqual(staging.read_bytes(), b"interrupted-package")


if __name__ == "__main__":
    unittest.main()
