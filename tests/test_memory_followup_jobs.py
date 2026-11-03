"""Exercise the Hamilton handoff without cluster credentials, Slurm, or a GPU."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = "b2547848b09b8c8cb3e7a03ce0154595465006d6"
PIN = "c" * 40


def tool(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)


class TestMemoryFollowupJobs(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.home = self.tmp / "hamilton"
        self.home.mkdir()
        self.prior = self.home / "isac-recovery-b2547848b09b"
        self.prior.mkdir()
        self.array = self.prior / "runs/array-19341792"
        self.array.mkdir(parents=True)
        self.root = self.home / f"isac-memory-{PIN[:12]}"
        self.fakepy = self.bin / "school-python"
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "TEST_HOME": str(self.home),
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
 rev-parse/HEAD) [[ "$PWD" == "$TEST_HOME/isac-recovery-b2547848b09b" ]] && echo "$TEST_BASE" || echo "$TEST_PIN";;
 rev-parse/FETCH_HEAD|rev-parse/refs/remotes/origin/research/memory-evidence-seeds) echo "$TEST_PIN";;
 merge-base/--is-ancestor) [[ "$3" == "$TEST_BASE" && "$4" == "$TEST_PIN" ]];;
 bundle/create) echo bundle > "$3";;
 bundle/verify|fetch/*|diff/--quiet|ls-files/--others) ;;
 worktree/add) mkdir "$4";;
 *) echo "Unexpected Git command: $*" >&2; exit 97;;
esac""",
        )
        tool(self.bin / "uname", "echo Darwin")
        tool(self.bin / "hostname", "echo login2.ham8.dur.ac.uk")
        tool(self.bin / "scp", 'cp "$1" "$TEST_HOME/${2#*:}"')
        tool(
            self.bin / "ssh",
            'shift; export HOME="$TEST_HOME"; exec bash -s -- ${1#*-- }',
        )
        tool(self.bin / "module", '[[ "$1 $2" == "load python/3.13.9cuda" ]]')
        tool(
            self.bin / "sbatch",
            """echo submit >> "$TEST_HISTORY"
[[ "$1 $2 $3" == "--parsable --export=ALL scripts/hamilton_memory_followup.sh" ]]
[[ -z "${MOCK_SBATCH_FAIL:-}" ]] || exit 72
echo "391;ham" """,
        )
        tool(
            self.fakepy,
            """
if [[ "$1" == '-' ]]; then cat >/dev/null; exit 0; fi
name=${1##*/}
shift
echo "$name" >> "$TEST_HISTORY"
[[ "$name" != "${MOCK_FAIL_AT:-}" ]] || exit 17
while [[ "$1" != --out ]]; do shift; done
dest=$2
if [[ "$name" == crossed_seed_study.py ]]; then
    mkdir "$dest"; dest="$dest/crossed.json"
fi
echo '{"status":"completed"}' > "$dest" """,
        )

    def prepare_prior(self):
        (self.prior / "runs/comparison-19341792.json").write_text("{}")
        for seed in (7, 11, 13):
            seed_dir = self.array / f"seed-{seed}"
            seed_dir.mkdir()
            (seed_dir / "study.json").write_text("{}")
            for width in (96, 192):
                (seed_dir / f"recovery-{width}.json").write_text("{}")
                for arch in ("structured", "structured-attention"):
                    cp = seed_dir / f"{arch}-{width}"
                    cp.mkdir()
                    (cp / "model.pt").write_bytes(b"fixture")

    def submit(self, **overrides):
        shell = (PROJECT / "scripts/submit_memory_followup_from_mac.sh").read_text()
        placeholder = "/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python"
        self.assertEqual(shell.count(placeholder), 1)
        launcher = self.tmp / "launcher.sh"
        launcher.write_text(shell.replace(placeholder, str(self.fakepy)))
        return subprocess.run(
            ["bash", str(launcher), PIN, "testuser"], env=self.env | overrides, text=True, capture_output=True
        )

    def test_confirmed_submission_once_and_unknown_receipt_not_retried(self):
        self.prepare_prior()
        a = self.submit()
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertIn("Slurm job: 391", a.stdout)
        b = self.submit()
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertIn("Already submitted", b.stdout)
        (self.root / ".memory-submission/receipt").unlink()
        c = self.submit()
        self.assertNotEqual(c.returncode, 0)
        self.assertIn("claim exists without a confirmed receipt", c.stderr)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def test_missing_checkpoint_does_not_submit_and_scheduler_error_is_not_retried(self):
        self.prepare_prior()
        cp = self.array / "seed-13/structured-attention-192/model.pt"
        cp.unlink()
        missing = self.submit()
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("Missing prior checkpoint", missing.stderr)
        self.assertFalse((self.root / ".memory-submission").exists())
        cp.write_bytes(b"fixture")
        rejected = self.submit(MOCK_SBATCH_FAIL="1")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertNotEqual(self.submit().returncode, 0)
        self.assertEqual((self.tmp / "history").read_text().splitlines(), ["submit"])

    def compute(self, job, **overrides):
        self.root.mkdir(exist_ok=True)
        env = self.env | {
            "SLURM_JOB_ID": job,
            "ISAC_FOLLOWUP_PIN": PIN,
            "ISAC_PROJECT_ROOT": str(self.root),
            "ISAC_RUNS_DIR": str(self.root / "runs"),
            "ISAC_ROUND2_DIR": str(self.array),
            "ISAC_PYTHON": str(self.fakepy),
        }
        env.pop("SLURM_ARRAY_JOB_ID", None)
        return subprocess.run(
            ["bash", str(PROJECT / "scripts/hamilton_memory_followup.sh")],
            env=env | overrides,
            text=True,
            capture_output=True,
        )

    def test_frozen_failure_stops_crossed_and_preserves_evidence(self):
        run = self.compute("392", MOCK_FAIL_AT="frozen_recovery.py")
        self.assertEqual(run.returncode, 17, run.stderr)
        out = self.root / "runs/followup-392"
        status = json.loads((out / "status.json").read_text())
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("failed", "frozen", 17))
        self.assertTrue((out / "prior-inputs.json").is_file())
        self.assertFalse((out / "crossed").exists())
        self.assertNotEqual(self.compute("392").returncode, 0)

    def test_compute_success_runs_in_order(self):
        run = self.compute("393")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(
            (self.tmp / "history").read_text().splitlines(),
            [
                "hamilton_preflight.py",
                "verify_round2_inputs.py",
                "frozen_recovery.py",
                "crossed_seed_study.py",
                "package_memory_followup.py",
            ],
        )
        status = json.loads((self.root / "runs/followup-393/status.json").read_text())
        self.assertEqual((status["status"], status["stage"], status["exit_code"]), ("completed", "done", 0))
