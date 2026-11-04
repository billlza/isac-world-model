"""Exercise submission, failure preservation and complete evidence packaging without Slurm/GPU."""

import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = "245a9dd56a49a72f942328d9c9cf9f27490b4d06"
PIN = "d" * 40
SPEC = importlib.util.spec_from_file_location(
    "capacity_package", PROJECT / "scripts/package_capacity_study.py"
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def tool(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)


class TestCapacityJobs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.remote_root = self.tmp / "hamilton"
        self.remote_root.mkdir()
        self.prior = self.remote_root / "isac-memory-245a9dd56a49"
        self.prior.mkdir()
        self.root = self.remote_root / f"isac-capacity-{PIN[:12]}"
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
   [[ "$PWD" == "$TEST_REMOTE_ROOT/isac-memory-245a9dd56a49" ]] && echo "$TEST_BASE" || echo "${MOCK_CHECKOUT_PIN:-$TEST_PIN}";;
 rev-parse/FETCH_HEAD) echo "$TEST_PIN";;
 rev-parse/refs/remotes/origin/research/capacity-data-h200) echo "${MOCK_REF_PIN:-$TEST_PIN}";;
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
[[ "$1 $2 $3" == "--parsable --export=ALL scripts/hamilton_capacity_study.sh" ]]
[[ "$ISAC_CAPACITY_PIN" == "$TEST_PIN" && "$ISAC_PROJECT_ROOT" == "$PWD" ]]
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
if [[ "$name" == capacity_data_study.py ]]; then
    [[ "$3 $4 $5 $6" == "--device cuda --profile formal" ]]
    mkdir "$dest"; dest="$dest/study.json"
fi
echo '{"status":"completed"}' > "$dest" """,
        )

    def submit(self, **overrides):
        shell = (PROJECT / "scripts/submit_capacity_study_from_mac.sh").read_text()
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
            "ISAC_CAPACITY_PIN": PIN,
            "ISAC_PROJECT_ROOT": str(self.root),
            "ISAC_RUNS_DIR": str(self.root / "runs"),
            "ISAC_PYTHON": str(self.fakepy),
        }
        env.pop("SLURM_ARRAY_JOB_ID", None)
        return subprocess.run(
            ["bash", str(PROJECT / "scripts/hamilton_capacity_study.sh")],
            env=env | overrides,
            text=True,
            capture_output=True,
        )

    def test_submit_once_and_report_saved_receipt_without_prior_results(self):
        first = self.submit()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("Slurm job: 391", first.stdout)
        self.assertIn('capacity_backup="$HOME/Downloads/isac-capacity-391"', first.stdout)
        self.assertIn("status.json", first.stdout)
        self.prior.rmdir()  # Existing receipt lookup does not require old checkpoint/result directories.
        second = self.submit()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("Already submitted", second.stdout)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])
        request = json.loads((self.root / ".capacity-submission/request.json").read_text())
        self.assertEqual(request["git_commit"], PIN)

    def test_unknown_receipt_and_scheduler_error_are_not_retried(self):
        failed = self.submit(MOCK_SBATCH_FAIL="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertTrue((self.root / ".capacity-submission/receipt.pending").exists())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0)
        self.assertIn("claim exists without a confirmed receipt", retried.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_malformed_success_receipt_is_retained_and_not_retried(self):
        malformed = self.submit(MOCK_RECEIPT="warning\n391;ham")
        self.assertNotEqual(malformed.returncode, 0)
        self.assertIn("Unexpected scheduler receipt", malformed.stderr)
        self.assertNotEqual(self.submit().returncode, 0)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_wrong_host_pin_or_dirty_source_cannot_submit(self):
        for case in (
            {"MOCK_UNAME": "Linux"},
            {"MOCK_REF_PIN": "e" * 40},
            {"MOCK_HOST": "unexpected.example"},
            {"MOCK_CHECKOUT_PIN": "e" * 40},
            {"MOCK_DIRTY": "1"},
            {"MOCK_UNTRACKED": "1"},
            {"MOCK_RUNTIME_FAIL": "1"},
        ):
            with self.subTest(case=case):
                self.assertNotEqual(self.submit(**case).returncode, 0)
                self.assertFalse((self.tmp / "history").exists())
                self.assertFalse((self.root / ".capacity-submission").exists())

    def test_study_failure_preserves_outputs_and_cannot_overwrite_them(self):
        run = self.compute("392", MOCK_FAIL_AT="capacity_data_study.py")
        self.assertEqual(run.returncode, 17, run.stderr)
        out = self.root / "runs/capacity-392"
        before = (out / "status.json").read_bytes()
        status = json.loads(before)
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("failed", "study", 17))
        self.assertTrue((out / "preflight.json").is_file())
        self.assertFalse((out / "isac-capacity-392.tar.gz").exists())
        self.assertNotEqual(self.compute("392").returncode, 0)
        self.assertEqual((out / "status.json").read_bytes(), before)

    def test_complete_job_and_packaging_failure_have_distinct_status(self):
        completed = self.compute("393")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        status = json.loads((self.root / "runs/capacity-393/status.json").read_text())
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("completed", "done", 0))
        self.assertEqual(
            (self.tmp / "history").read_text().splitlines(),
            ["hamilton_preflight.py", "capacity_data_study.py", "package_capacity_study.py"],
        )
        failed = self.compute("394", MOCK_FAIL_AT="package_capacity_study.py")
        self.assertEqual(failed.returncode, 17, failed.stderr)
        out = self.root / "runs/capacity-394"
        self.assertTrue((out / "study/study.json").is_file())
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "package"))


class TestCapacityPackage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "isac-capacity-391.tar.gz"
        self.study = {
            "status": "completed",
            "profile": "formal",
            "git_commit": PIN,
            "slurm_job_id": "391",
            "runtime": {"device": "cuda"},
            "cells": {},
        }
        pairs = [
            (seed, condition)
            for seed in range(95_871_000, 95_871_024)
            for condition in ("control", "maneuver")
        ]
        names = [
            f"a{width}-data-{episodes}-init-{seed}"
            for width in (96, 512)
            for episodes in (96, 1536)
            for seed in (7, 11)
        ]
        methods = names + ["kalman", "gated-imm"]
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
        for width in (96, 512):
            for episodes in (96, 1536):
                for seed in (7, 11):
                    name = f"a{width}-data-{episodes}-init-{seed}"
                    prefix = f"study/cells/{name}"
                    epochs = 128 if episodes == 96 else 8
                    updates = list(range(96, 1537, 96))
                    initial = hashlib.sha256(f"initial-{width}-{seed}".encode()).hexdigest()
                    parameters = 172246 if width == 96 else 3883798
                    manifest = {
                        "seed": seed,
                        "epochs": epochs,
                        "completed_epochs": epochs,
                        "episodes": episodes,
                        "batch_size": 8,
                        "validation_episodes": 192,
                        "validation_every_updates": 96,
                        "optimizer_updates": 1536,
                        "validation_opportunities": 16,
                        "validation_updates": updates,
                        "train_episode_seeds": list(range(700_000, 700_000 + episodes)),
                        "validation_episode_seeds": list(range(74_611_000, 74_611_192)),
                        "parameters": parameters,
                        "resume_source": None,
                        "seed_plan": {
                            "initialization": seed,
                            "train_seed_base": 700_000,
                            "validation_seed_base": 74_611_000,
                            "batch_order_seed": 37,
                        },
                        "runtime": {"device": "cuda"},
                        "model": {"architecture": "structured-attention", "hidden_dim": width},
                        "model_selection_status": "selected",
                        "best_update": 1536,
                        "best_validation_rmse_m": 1.0 / 1536,
                        "initial_parameters_sha256": initial,
                    }
                    manifest_hash = self.write(f"{prefix}/manifest.json", manifest)
                    history = [
                        {"optimizer_updates": step, "validation_position_rmse_m": 1.0 / step}
                        for step in updates
                    ]
                    history_hash = self.write(f"{prefix}/history.json", history)
                    epoch_history_hash = self.write(
                        f"{prefix}/epoch_history.json",
                        [
                            {"epoch": k, "optimizer_updates": k * episodes // 8, "train_loss": 0.1}
                            for k in range(1, epochs + 1)
                        ],
                    )
                    weights = f"fixture-{name}".encode()
                    model = self.root / prefix / "model.pt"
                    model.write_bytes(weights)
                    checkpoint_hash = hashlib.sha256(weights).hexdigest()
                    model.with_suffix(".pt.sha256").write_text(checkpoint_hash + "\n")
                    self.study["cells"][name] = {
                        "status": "completed",
                        "hidden_dim": width,
                        "training_episodes": episodes,
                        "initialization": seed,
                        "parameters": parameters,
                        "optimizer_updates": 1536,
                        "validation_opportunities": 16,
                        "validation_updates": updates,
                        "best_update": 1536,
                        "best_validation_rmse_m": 1.0 / 1536,
                        "initial_parameters_sha256": initial,
                        "checkpoint_sha256": checkpoint_hash,
                        "manifest_sha256": manifest_hash,
                        "history_sha256": history_hash,
                        "epoch_history_sha256": epoch_history_hash,
                        "checkpoint": f"cells/{name}/model.pt",
                    }
                    self.write(
                        f"study/diagnostics/{name}.json",
                        {
                            "status": "completed",
                            "git_commit": PIN,
                            "cell": name,
                            "checkpoint_sha256": checkpoint_hash,
                            "episodes": [
                                {"seed": s, "condition": c, "report": {"status": "completed", "events": []}}
                                for s, c in pairs
                            ],
                        },
                    )
        summary = {
            condition: {"methods": {name: {"mean_auc_m_s": 0.1} for name in methods}}
            for condition in ("control", "maneuver")
        }
        self.fresh = {
            "status": "completed",
            "profile": "formal",
            "git_commit": PIN,
            "summary": summary,
            "rows": [
                {
                    "seed": seed,
                    "condition": condition,
                    "scores": {
                        name: {
                            "position_error_auc_m_s": 0.1,
                            "three_step_forecast_rmse_m": 0.2,
                            "post_errors_m": [0.1] * 11,
                        }
                        for name in methods
                    },
                }
                for seed, condition in pairs
            ],
        }
        self.study["results"] = {"fresh_test": summary}
        self.write("study/study.json", self.study)
        self.write("study/fresh_test.json", self.fresh)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, indent=2) + "\n").encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_complete_bundle_contains_all_records_and_hashes_but_no_weights(self):
        result = packager.package(self.root, self.out)
        self.assertEqual(result["status"], "packaged")
        self.assertEqual(result["sha256"], hashlib.sha256(self.out.read_bytes()).hexdigest())
        with tarfile.open(self.out) as archive:
            names = archive.getnames()
            self.assertEqual(len([n for n in names if n.startswith("study/diagnostics/")]), 8)
            self.assertEqual(len([n for n in names if n.endswith("/history.json")]), 8)
            self.assertFalse(any(n.endswith(".pt") for n in names))
            bundle = json.load(archive.extractfile("bundle_manifest.json"))
            self.assertEqual(bundle["slurm_job_id"], "391")
            for name, meta in bundle["artifacts"].items():
                raw = archive.extractfile(name).read()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"])
                self.assertEqual(len(raw), meta["bytes"])
            status = json.load(archive.extractfile("job_status_at_packaging.json"))
            self.assertEqual(status["status"], "running")
        original = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            packager.package(self.root, self.out)
        self.assertEqual(self.out.read_bytes(), original)

    def test_missing_cell_or_wrong_job_cannot_create_archive(self):
        cell = self.study["cells"].pop("a512-data-1536-init-11")
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "formal capacity/data cell"):
            packager.package(self.root, self.out)
        self.assertFalse(self.out.exists())
        self.study["cells"]["a512-data-1536-init-11"] = cell
        self.study["slurm_job_id"] = "392"
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "matching formal CUDA"):
            packager.package(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_changed_weights_or_training_history_are_detected(self):
        model = self.root / "study/cells/a512-data-1536-init-11/model.pt"
        before = model.read_bytes()
        model.write_bytes(before + b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
            packager.package(self.root, self.out)
        model.write_bytes(before)
        history = model.parent / "history.json"
        history.write_bytes(history.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "History changed"):
            packager.package(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_incomplete_classical_forecast_or_duplicate_test_episode_is_rejected(self):
        value = self.fresh["rows"][0]["scores"]["gated-imm"].pop("three_step_forecast_rmse_m")
        self.write("study/fresh_test.json", self.fresh)
        with self.assertRaisesRegex(ValueError, "three_step_forecast_rmse_m"):
            packager.package(self.root, self.out)
        self.fresh["rows"][0]["scores"]["gated-imm"]["three_step_forecast_rmse_m"] = value
        self.fresh["rows"][-1] = self.fresh["rows"][0]
        self.write("study/fresh_test.json", self.fresh)
        with self.assertRaisesRegex(ValueError, "episode identities"):
            packager.package(self.root, self.out)

    def test_nan_diagnostics_and_interrupted_publication_are_not_hidden(self):
        path = "study/diagnostics/a96-data-96-init-7.json"
        report = json.loads((self.root / path).read_text())
        report["episodes"][0]["report"]["bad_value"] = float("nan")
        self.write(path, report)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            packager.package(self.root, self.out)
        self.assertFalse(self.out.exists())
        partial = self.out.with_suffix(self.out.suffix + ".writing")
        partial.write_bytes(b"interrupted")
        with self.assertRaisesRegex(ValueError, "staging exists"):
            packager.package(self.root, self.out)
        self.assertEqual(partial.read_bytes(), b"interrupted")

    def test_epoch_history_selected_score_and_diagnostic_completion_are_bound(self):
        cell = self.study["cells"]["a96-data-96-init-7"]
        path = self.root / "study/cells/a96-data-96-init-7/epoch_history.json"
        before = path.read_bytes()
        path.write_bytes(before + b" ")
        with self.assertRaisesRegex(ValueError, "Epoch history changed"):
            packager.package(self.root, self.out)
        path.write_bytes(before)
        original_score = cell["best_validation_rmse_m"]
        cell["best_validation_rmse_m"] = 0.0
        self.write("study/study.json", self.study)
        with self.assertRaisesRegex(ValueError, "Selected validation score differs"):
            packager.package(self.root, self.out)
        cell["best_validation_rmse_m"] = original_score
        self.write("study/study.json", self.study)
        diagnostic_path = "study/diagnostics/a96-data-96-init-7.json"
        diagnostic = json.loads((self.root / diagnostic_path).read_text())
        diagnostic["episodes"][0]["report"]["status"] = "running"
        self.write(diagnostic_path, diagnostic)
        with self.assertRaisesRegex(ValueError, "Missing correction diagnostics"):
            packager.package(self.root, self.out)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
