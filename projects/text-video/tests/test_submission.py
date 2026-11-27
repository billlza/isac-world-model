"""Exercise launch safety in isolated Git checkouts, with a fake scheduler and data CLI.

These tests do not claim Hamilton or GPU validation. The allocation script performs
the real module/version/device/backward checks before starting the real trainer.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    scripts = repo / "projects/text-video/scripts"
    source = repo / "projects/text-video/src/text_video"
    scripts.mkdir(parents=True)
    source.mkdir(parents=True)
    for name in ("submit_hamilton.sh", "hamilton_train.sh"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    (source / "__init__.py").write_text("")
    (source / "__main__.py").write_text(
        "import argparse, hashlib, json\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('command', choices=['validate-data'])\n"
        "p.add_argument('--manifest', type=Path, required=True)\n"
        "p.add_argument('--config', type=Path, required=True)\n"
        "a = p.parse_args()\n"
        "json.loads(a.config.read_text())\n"
        "m = json.loads(a.manifest.read_text())\n"
        "asset = a.manifest.parent / m['asset']\n"
        "fingerprint = hashlib.sha256(a.manifest.read_bytes() + asset.read_bytes()).hexdigest()\n"
        "print(json.dumps({'data_fingerprint': fingerprint, 'clips': 1}))\n"
    )
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Launch Test", "-c", "user.email=launch@example.test", "commit", "-qm", "fixture"],
        check=True,
    )
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    config, manifest, asset = (tmp_path / name for name in ("config.json", "manifest.json", "clip.bin"))
    config.write_text('{"schema_version": 1}\n')
    manifest.write_text('{"asset": "clip.bin"}\n')
    asset.write_bytes(b"fixture asset")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # The hostname override is confined to this test interpreter, not a production bypass.
    interpreter = fake_bin / "launch-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import os, socket, sys\n"
        "socket.getfqdn = lambda: os.environ.get('MOCK_HOST', 'login1.ham8.dur.ac.uk')\n"
        "sys.argv = sys.argv[1:]\n"
        "exec(compile(sys.stdin.read(), 'submit_hamilton.py', 'exec'), {'__name__': '__main__'})\n"
    )
    interpreter.chmod(0o700)
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['MOCK_SBATCH_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 'sbatch_env': [k for k in os.environ if k.startswith('SBATCH_')]}) + '\\n')\n"
        "print(os.environ.get('MOCK_RECEIPT', '793021;hamilton'))\n"
        "raise SystemExit(int(os.environ.get('MOCK_SBATCH_RC', '0')))\n"
    )
    sbatch.chmod(0o700)
    out = tmp_path / "train-output"
    calls = tmp_path / "sbatch-calls.jsonl"
    env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
           "TEXT_VIDEO_LAUNCH_PYTHON": str(interpreter), "MOCK_SBATCH_CALLS": str(calls)}
    return {"repo": repo, "script": scripts / "submit_hamilton.sh", "source": source,
            "config": config, "manifest": manifest, "asset": asset, "out": out,
            "prepared": out.with_name(out.name + ".submission"), "calls": calls, "sha": sha, "env": env}


def launch(checkout, *extra, sha=None):
    return subprocess.run(
        ["bash", str(checkout["script"]), sha or checkout["sha"], str(checkout["config"]),
         str(checkout["manifest"]), str(checkout["out"]), *extra],
        env=checkout["env"], capture_output=True, text=True, check=False,
    )


def read_calls(checkout):
    return [json.loads(line) for line in checkout["calls"].read_text().splitlines()]


def test_check_prepares_pinned_request_without_submission_or_training(checkout):
    first = launch(checkout)
    assert first.returncode == 0, first.stderr
    second = launch(checkout, "--check")
    assert second.returncode == 0, second.stderr
    assert json.loads(first.stdout)["status"] == "prepared_not_submitted"
    assert json.loads(second.stdout)["gpu_validation"] == "not_executed"
    assert not checkout["calls"].exists()
    assert not checkout["out"].exists()
    request = json.loads((checkout["prepared"] / "request.json").read_text())
    assert request["git_commit"] == checkout["sha"]
    assert request["initialization"] == "random"
    assert request["RL"] == request["RLHF"] == "not_implemented_not_executed"
    assert len(request["inputs"]["config"]["sha256"]) == 64
    assert len(request["data_validation"]["data_fingerprint"]) == 64
    assert "projects/text-video/scripts/hamilton_train.sh" in request["source_sha256"]


@pytest.mark.parametrize("change", ["tracked", "untracked", "wrong_sha"])
def test_refuses_dirty_or_wrong_commit(checkout, change):
    if change == "tracked":
        (checkout["source"] / "__init__.py").write_text("# modified\n")
    if change == "untracked":
        (checkout["source"] / "uncommitted.py").write_text("# uncommitted\n")
    result = launch(checkout, "--submit", sha="0" * 40 if change == "wrong_sha" else None)
    assert result.returncode != 0
    assert "clean committed checkout" in result.stderr or "full current HEAD" in result.stderr
    assert not checkout["prepared"].exists()
    assert not checkout["calls"].exists()


def test_submission_has_only_verified_resource_requests_and_receipt(checkout):
    checkout["env"].update(SBATCH_CPUS_PER_TASK="64", SBATCH_MEM="400G", SBATCH_ACCOUNT="unverified")
    result = launch(checkout, "--submit")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "submitted_not_completed"
    call, = read_calls(checkout)
    command = call["argv"]
    assert command[command.index("-p") + 1] == "cuda"
    assert "--gres=gpu:h200_nvl:1" in command
    assert "--time=04:00:00" in command
    assert not any(arg.startswith(("--cpus", "--mem", "--account", "--ntasks", "--nodes")) or arg in ("-c", "-A", "-N", "-n") for arg in command)
    assert call["sbatch_env"] == []
    assert command[-2].endswith("/hamilton_train.sh")
    assert command[-1] == str(checkout["prepared"] / "request.json")
    assert (checkout["prepared"] / "claim/receipt").read_text().strip() == "793021;hamilton"
    assert not checkout["out"].exists()
    second = launch(checkout, "--submit")
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["status"] == "already_submitted"
    assert len(read_calls(checkout)) == 1


@pytest.mark.parametrize("receipt,code", [("793021;hamilton", "1"), ("unexpected scheduler response", "0")])
def test_ambiguous_submission_claim_blocks_retry(checkout, receipt, code):
    checkout["env"].update(MOCK_RECEIPT=receipt, MOCK_SBATCH_RC=code)
    first = launch(checkout, "--submit")
    assert first.returncode != 0
    assert "claim retained" in first.stderr
    second = launch(checkout, "--submit")
    assert second.returncode != 0
    assert "without a confirmed receipt" in second.stderr
    assert len(read_calls(checkout)) == 1
    assert (checkout["prepared"] / "claim/receipt.pending").exists()
    assert not (checkout["prepared"] / "claim/receipt").exists()


@pytest.mark.parametrize("changed", ["config", "asset"])
def test_changed_external_input_cannot_reuse_preparation(checkout, changed):
    assert launch(checkout).returncode == 0
    checkout[changed].write_text('{"changed": true}\n')
    result = launch(checkout, "--submit")
    assert result.returncode != 0
    assert "Existing preparation differs" in result.stderr
    assert not checkout["calls"].exists()


def test_existing_output_and_missing_assets_are_refused(checkout):
    checkout["out"].mkdir()
    result = launch(checkout, "--submit")
    assert result.returncode != 0 and "OUT already exists" in result.stderr
    checkout["out"].rmdir()
    checkout["asset"].unlink()
    result = launch(checkout, "--submit")
    assert result.returncode != 0 and "Dataset validation failed" in result.stderr
    assert not checkout["calls"].exists()


def test_sft_requires_checkpoint_and_pretraining_refuses_one(checkout):
    checkpoint = checkout["out"].parent / "pretrain.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    for extra in (("sft",), ("pretrain", str(checkpoint))):
        result = launch(checkout, *extra)
        assert result.returncode != 0 and "sft requires INIT_CHECKPOINT" in result.stderr
    result = launch(checkout, "sft", str(checkpoint))
    assert result.returncode == 0, result.stderr
    request = json.loads((checkout["prepared"] / "request.json").read_text())
    assert request["stage"] == "sft" and request["initialization"] == "checkpoint"
    assert request["inputs"]["init_checkpoint"]["path"] == str(checkpoint)


def test_non_hamilton_host_cannot_submit(checkout):
    checkout["env"]["MOCK_HOST"] = "local-development.example"
    result = launch(checkout, "--submit")
    assert result.returncode != 0 and "Hamilton login host" in result.stderr
    assert not checkout["calls"].exists()
    assert not (checkout["prepared"] / "claim").exists()


def test_allocation_script_refuses_login_execution_and_records_error(checkout):
    assert launch(checkout).returncode == 0
    environment = {key: value for key, value in checkout["env"].items() if not key.startswith("SLURM_")}
    result = subprocess.run(
        ["bash", str(checkout["script"].with_name("hamilton_train.sh")), str(checkout["prepared"] / "request.json")],
        cwd=checkout["repo"], env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0 and "single Slurm allocation" in result.stderr
    status = json.loads((checkout["prepared"] / "job_status.json").read_text())
    assert status["status"] == "error"
    assert status["phase"] == "allocation_or_module_startup"
    assert not checkout["out"].exists()


def test_module_failure_is_recorded_without_starting_training(checkout):
    assert launch(checkout).returncode == 0
    module_args = checkout["out"].parent / "module-args"
    environment = {key: value for key, value in checkout["env"].items() if not key.startswith("SLURM_")}
    environment.update({
        "SLURM_JOB_ID": "42", "MODULE_ARGS": str(module_args),
        "BASH_FUNC_module%%": '() { printf "%s\\n" "$*" > "$MODULE_ARGS"; return 19; }',
    })
    result = subprocess.run(
        ["bash", str(checkout["script"].with_name("hamilton_train.sh")), str(checkout["prepared"] / "request.json")],
        cwd=checkout["repo"], env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 19
    assert module_args.read_text().strip() == "load python/3.13.9cuda"
    status = json.loads((checkout["prepared"] / "job_status.json").read_text())
    assert status["status"] == "error" and status["exit_code"] == 19
    assert not checkout["out"].exists()


def media_check(import_module):
    source = (SCRIPTS / "hamilton_train.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "check_media_dependencies")
    namespace = {"importlib": SimpleNamespace(import_module=import_module), "sys": sys}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "hamilton_media_preflight", "exec"), namespace)
    return namespace["check_media_dependencies"]()


def test_media_dependencies_record_imported_versions():
    versions = {"PIL": "11.2.1", "imageio": "2.37.0", "imageio_ffmpeg": "0.6.0"}
    imported = []

    def import_module(name):
        imported.append(name)
        return SimpleNamespace(__version__=versions[name.split(".")[0]])

    assert media_check(import_module) == {"Pillow": "11.2.1", "imageio": "2.37.0", "imageio-ffmpeg": "0.6.0"}
    assert {"PIL.Image", "imageio.v3", "imageio_ffmpeg"}.issubset(imported)


@pytest.mark.parametrize("missing,distribution", [("PIL.Image", "Pillow"), ("imageio.v3", "imageio"), ("imageio_ffmpeg", "imageio-ffmpeg")])
def test_missing_media_dependency_is_an_explicit_failure(missing, distribution):
    def import_module(name):
        if name == missing:
            raise ModuleNotFoundError(name)
        return SimpleNamespace(__version__="test-version")

    with pytest.raises(RuntimeError, match=f"Required media dependency {distribution}") as error:
        media_check(import_module)
    assert "PYTHONNOUSERSITE=1" in str(error.value)
    assert "No dependencies were installed" in str(error.value)
