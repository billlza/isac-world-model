"""Exercise full-bundle NCC setup/pilot submission without SSH, Slurm or a GPU."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REF = "refs/remotes/origin/research/ncc-mig-pilot"


def tool(path, body):
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)


def output(result):
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


class TestNccMigSubmission(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.remote_home = self.tmp / "ncc"
        self.remote_home.mkdir()
        self.local_repo = self.tmp / "mac-repo"
        self.local_repo.mkdir()
        self.git = shutil.which("git")
        if self.git is None:
            self.skipTest("Git is required to verify a real standalone bundle")
        self.git_env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Submission Test",
            "GIT_AUTHOR_EMAIL": "submission@example.invalid",
            "GIT_COMMITTER_NAME": "Submission Test",
            "GIT_COMMITTER_EMAIL": "submission@example.invalid",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
        self.git_run("init", "--quiet")
        (self.local_repo / "src").mkdir()
        (self.local_repo / "scripts").mkdir()
        (self.local_repo / "src/model.py").write_text("# first revision\n")
        for script in ("ncc_runtime_setup.sh", "ncc_mig_pilot.sh"):
            (self.local_repo / "scripts" / script).write_text("#!/bin/bash\nexit 0\n")
        self.git_run("add", ".")
        self.git_run("commit", "--quiet", "-m", "Base source")
        self.base = self.git_run("rev-parse", "HEAD").stdout.strip()
        (self.local_repo / "src/model.py").write_text("# pinned revision\n")
        self.git_run("commit", "--quiet", "-am", "Pinned source")
        self.pin = self.git_run("rev-parse", "HEAD").stdout.strip()
        self.git_run("update-ref", REF, self.pin)
        self.root = self.remote_home / f"isac-ncc-mig-{self.pin[:12]}"
        self.runtime = self.remote_home / f"isac-ncc-runtime-{self.pin[:12]}"
        self.claim = self.root / ".ncc-mig-submission"
        self.history = self.tmp / "history"
        self.env = {
            **self.git_env,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "TEST_REMOTE_HOME": str(self.remote_home),
            "TEST_PIN": self.pin,
            "TEST_HISTORY": str(self.history),
        }
        tool(self.bin / "uname", 'echo "${MOCK_UNAME:-Darwin}"')
        tool(self.bin / "hostname", 'echo "${MOCK_HOST:-ncc1}"')
        tool(self.bin / "id", '[[ "$1" == -un ]]; echo "${MOCK_USER:-testuser}"')
        tool(
            self.bin / "scp",
            """[[ "$2" == testuser@ncc1.clients.dur.ac.uk:* ]]
cp "$1" "$TEST_REMOTE_HOME/${2#*:}"
""",
        )
        tool(
            self.bin / "ssh",
            """[[ "$1" == testuser@ncc1.clients.dur.ac.uk ]]
shift
read -r -a args <<< "${1#*-- }"
exec bash -s -- "${args[@]}"
""",
        )
        tool(
            self.bin / "sbatch",
            """
[[ "$ISAC_NCC_PIN" == "$TEST_PIN" && "$ISAC_PROJECT_ROOT" == "$PWD" ]]
[[ "$ISAC_RUNS_DIR" == "$PWD/runs" ]]
[[ "$ISAC_NCC_RUNTIME_ROOT" == "$TEST_REMOTE_HOME/isac-ncc-runtime-${TEST_PIN:0:12}" ]]
[[ ! -e "$ISAC_NCC_RUNTIME_ROOT" ]]
[[ -z "${SBATCH_PARTITION:-}" && -z "${SBATCH_GRES:-}" ]]
case "${!#}" in
  scripts/ncc_runtime_setup.sh)
    [[ "$*" == '--parsable --export=ALL scripts/ncc_runtime_setup.sh' ]]
    [[ -z "${ISAC_NCC_SETUP_JOB_ID:-}" ]]
    stage=setup
    receipt=${MOCK_SETUP_RECEIPT:-411;ncc}
    ;;
  scripts/ncc_mig_pilot.sh)
    [[ "$*" == '--parsable --export=ALL --dependency=afterok:411 --kill-on-invalid-dep=yes scripts/ncc_mig_pilot.sh' ]]
    [[ "$ISAC_NCC_SETUP_JOB_ID" == 411 ]]
    [[ -s .ncc-mig-submission/setup.receipt ]]
    stage=pilot
    receipt=${MOCK_PILOT_RECEIPT:-412;ncc}
    ;;
  *) exit 97 ;;
esac
printf '%s\\n' "$stage" >> "$TEST_HISTORY"
if [[ "$stage" == "${MOCK_FAIL_STAGE:-}" ]]; then
  printf '%s submission rejected by mock scheduler\\n' "$stage" >&2
  exit 72
fi
printf '%s\\n' "$receipt"
""",
        )

    def git_run(self, *args, cwd=None):
        return subprocess.run(
            [self.git, *args],
            cwd=cwd or self.local_repo,
            env=self.git_env,
            text=True,
            capture_output=True,
            check=True,
        )

    def make_remote_checkout(self):
        self.git_run("clone", "--quiet", str(self.local_repo), str(self.root))

    def submit(self, *, pin=None, **overrides):
        shell = (PROJECT / "scripts/submit_ncc_mig_pilot_from_mac.sh").read_text()
        # Redirect only the remote fixture's paths; never repurpose the real HOME.
        for variable in ("root", "runtime_root", "bundle"):
            fragment = f'\n{variable}="$HOME/'
            self.assertEqual(shell.count(fragment), 1)
            shell = shell.replace(fragment, f'\n{variable}="{self.remote_home}/')
        launcher = self.tmp / "launcher.sh"
        launcher.write_text(shell)
        return subprocess.run(
            ["bash", str(launcher), pin or self.pin, "testuser"],
            cwd=self.local_repo,
            env=self.env | overrides,
            text=True,
            capture_output=True,
        )

    def submits(self):
        return self.history.read_text().splitlines() if self.history.exists() else []

    def test_full_bundle_bootstraps_fresh_ncc_and_submits_both_jobs_once(self):
        first = self.submit(SBATCH_PARTITION="wrong", SBATCH_GRES="gpu:hopper:8")
        self.assertEqual(first.returncode, 0, output(first))
        self.assertIn("CPU runtime setup job: 411", first.stdout)
        self.assertIn("MIG A512 pilot job: 412 (afterok:411)", first.stdout)
        self.assertIn("runs/ncc-mig-412/status.json", first.stdout)
        self.assertEqual(self.git_run("rev-parse", "HEAD", cwd=self.root).stdout.strip(), self.pin)
        # A real empty destination verifies and fetches the full two-commit history.
        self.assertEqual(self.git_run("rev-list", "--count", "HEAD", cwd=self.root).stdout.strip(), "2")
        self.assertFalse(self.runtime.exists())
        self.assertEqual((self.claim / "setup.receipt").read_text(), "411;ncc\n")
        self.assertEqual((self.claim / "pilot.receipt").read_text(), "412;ncc\n")
        self.assertEqual((self.claim / "setup.job-id").read_text(), "411\n")
        self.assertEqual((self.claim / "pilot.job-id").read_text(), "412\n")
        request = json.loads((self.claim / "request.json").read_text())
        self.assertEqual((request["git_commit"], request["profile"]), (self.pin, "development"))
        self.assertEqual(request["gres"], "gpu:1g.10gb:1")
        before = {p.name: p.read_bytes() for p in self.claim.iterdir()}
        second = self.submit()
        self.assertEqual(second.returncode, 0, output(second))
        self.assertIn("no jobs will be resubmitted", second.stdout)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.claim.iterdir()})
        self.assertEqual(self.submits(), ["setup", "pilot"])
        self.assertFalse(list(self.remote_home.glob("*.bundle")))

    def test_setup_scheduler_error_keeps_error_receipt_and_blocks_retry(self):
        failed = self.submit(MOCK_FAIL_STAGE="setup")
        self.assertNotEqual(failed.returncode, 0, output(failed))
        self.assertIn("setup submission rejected", (self.claim / "setup.submit.error").read_text())
        self.assertTrue((self.claim / "setup.receipt.pending").exists())
        self.assertFalse((self.claim / "setup.receipt").exists())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, output(retried))
        self.assertIn("without a confirmed setup receipt", retried.stderr)
        self.assertEqual(self.submits(), ["setup"])

    def test_setup_malformed_success_receipt_does_not_submit_gpu_or_retry(self):
        failed = self.submit(MOCK_SETUP_RECEIPT="unexpected\n411;ncc")
        self.assertNotEqual(failed.returncode, 0, output(failed))
        self.assertIn("Invalid saved Slurm receipt", failed.stderr)
        self.assertIn("unexpected", (self.claim / "setup.receipt.pending").read_text())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, output(retried))
        self.assertEqual(self.submits(), ["setup"])

    def test_gpu_scheduler_error_keeps_setup_job_and_does_not_retry_either(self):
        failed = self.submit(MOCK_FAIL_STAGE="pilot")
        self.assertNotEqual(failed.returncode, 0, output(failed))
        self.assertEqual((self.claim / "setup.job-id").read_text(), "411\n")
        self.assertIn("pilot submission rejected", (self.claim / "pilot.submit.error").read_text())
        self.assertTrue((self.claim / "pilot.receipt.pending").exists())
        self.assertFalse((self.claim / "pilot.receipt").exists())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, output(retried))
        self.assertIn("CPU runtime setup job: 411", retried.stdout)
        self.assertIn("no confirmed receipt", retried.stderr)
        self.assertEqual(self.submits(), ["setup", "pilot"])

    def test_gpu_malformed_success_receipt_is_not_resubmitted(self):
        failed = self.submit(MOCK_PILOT_RECEIPT="unexpected\n412;ncc")
        self.assertNotEqual(failed.returncode, 0, output(failed))
        self.assertIn("unexpected", (self.claim / "pilot.receipt.pending").read_text())
        retried = self.submit()
        self.assertNotEqual(retried.returncode, 0, output(retried))
        self.assertEqual(self.submits(), ["setup", "pilot"])

    def assert_rejected_without_submission(self, **overrides):
        rejected = self.submit(**overrides)
        self.assertNotEqual(rejected.returncode, 0, output(rejected))
        self.assertEqual(self.submits(), [], output(rejected))
        self.assertFalse(self.claim.exists(), output(rejected))
        return rejected

    def test_non_mac_cannot_submit(self):
        self.assert_rejected_without_submission(MOCK_UNAME="Linux")

    def test_wrong_published_pin_cannot_submit(self):
        self.assert_rejected_without_submission(pin="e" * 40)

    def test_wrong_remote_host_cannot_submit(self):
        self.assert_rejected_without_submission(MOCK_HOST="login2.ham8.dur.ac.uk")

    def test_wrong_remote_login_identity_cannot_submit(self):
        self.assert_rejected_without_submission(MOCK_USER="otheruser")

    def test_existing_checkout_at_wrong_pin_is_not_overwritten(self):
        self.make_remote_checkout()
        self.git_run("checkout", "--detach", self.base, cwd=self.root)
        self.assert_rejected_without_submission()
        self.assertEqual(self.git_run("rev-parse", "HEAD", cwd=self.root).stdout.strip(), self.base)

    def test_tracked_changes_cannot_submit(self):
        self.make_remote_checkout()
        (self.root / "src/model.py").write_text("# uncommitted change\n")
        self.assert_rejected_without_submission()
        self.assertEqual((self.root / "src/model.py").read_text(), "# uncommitted change\n")

    def test_untracked_source_cannot_submit(self):
        self.make_remote_checkout()
        (self.root / "scripts/shadow.py").write_text("# untracked source\n")
        self.assert_rejected_without_submission()

    def test_existing_runtime_without_submission_is_preserved(self):
        self.runtime.mkdir()
        marker = self.runtime / "preserved.txt"
        marker.write_text("existing runtime\n")
        rejected = self.assert_rejected_without_submission()
        self.assertIn("Runtime directory already exists", rejected.stderr)
        self.assertEqual(marker.read_text(), "existing runtime\n")

    def test_partial_claim_without_pin_is_preserved_and_not_submitted(self):
        self.make_remote_checkout()
        self.claim.mkdir()
        (self.claim / "stage").write_text("partial\n")
        rejected = self.submit()
        self.assertNotEqual(rejected.returncode, 0, output(rejected))
        self.assertEqual(self.submits(), [])
        self.assertIn("Existing submission claim is incomplete", rejected.stderr)
        self.assertEqual((self.claim / "stage").read_text(), "partial\n")


if __name__ == "__main__":
    unittest.main()
