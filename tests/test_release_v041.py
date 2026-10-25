"""Release failure modes are tested without contacting GitHub or installing packages."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepared(tmp):
    e = script("evidence")
    root = tmp / "source"
    root.mkdir()
    for name in e.TOP_LEVEL:
        (root / name).write_text("fixture")
    out = tmp / "evidence"
    out.mkdir()
    steps = []
    for name in sorted(e.REQUIRED_ALL):
        (out / f"{name}.log").write_text(name + "\n")
        steps.append({"name": name, "status": "passed", "exit_code": 0, "log": f"{name}.log"})
    source = e.source_snapshot(root)
    report = {
        "schema_version": 2,
        "profile": "all",
        "status": "passed",
        "steps": steps,
        "source_sha256": source,
        "source_tree_sha256": e.tree_digest(source),
        "artifact_sha256": e.output_snapshot(out),
    }
    p = out / "ci-report.json"
    p.write_text(json.dumps(report))
    return e, root, p, report


def test_full_evidence_accepts_unchanged_bytes(tmp_path):
    e, root, p, _ = prepared(tmp_path)
    assert e.verify(p, root)["checks"] == len(e.REQUIRED_ALL)


@pytest.mark.parametrize(
    "field", ["source", "log", "step", "partial", "missing", "duplicate", "empty-hashes"]
)
def test_release_gate_rejects_incomplete_or_changed_evidence(tmp_path, field):
    e, root, p, r = prepared(tmp_path)
    if field == "source":
        (root / "uv.lock").write_text("changed")
    elif field == "log":
        (p.parent / r["steps"][0]["log"]).write_text("changed")
    elif field == "step":
        r["steps"][0]["status"] = "skipped"
    elif field == "partial":
        r["profile"] = "python"
    elif field == "missing":
        r["steps"].pop()
    elif field == "duplicate":
        r["steps"].append(r["steps"][0])
    else:
        r["artifact_sha256"] = {}
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        e.verify(p, root)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a\\b", ""])
def test_evidence_rejects_path_escape(tmp_path, name):
    with pytest.raises(ValueError):
        script("evidence").safe_child(tmp_path, name)


def test_evidence_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("x")
    link = tmp_path / "alias"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        script("evidence").safe_child(tmp_path, "alias")


@pytest.mark.parametrize(
    "name", ["other/repo", "billlza/../repo", "billlza/x;echo", "https://github.com/billlza/repo"]
)
def test_publication_rejects_wrong_target(name):
    with pytest.raises(ValueError):
        script("publish_private").validate_repo_name(name)


def test_publication_requires_lock_and_runtime_in_commit():
    p = script("publish_private")
    assert {"uv.lock", ".python-version", "rust-toolchain.toml"}.issubset(p.REQUIRED_FILES)


@pytest.mark.parametrize("private,full_name", [(False, "billlza/isac-world-model"), (True, "other/repo")])
def test_publication_never_accepts_public_or_other_repository(private, full_name):
    with pytest.raises(ValueError):
        script("publish_private").validate_repo(
            {"private": private, "full_name": full_name}, "billlza/isac-world-model"
        )


def good_run():
    return {
        "headSha": "a" * 40,
        "status": "completed",
        "conclusion": "success",
        "url": "https://github.com/billlza/isac-world-model/actions/runs/1",
        "jobs": [
            {"name": "ci-gate", "status": "completed", "conclusion": "success"},
            {"name": "Python", "status": "completed", "conclusion": "success"},
        ],
    }


def test_exact_remote_run_accepted():
    script("publish_private").validate_run(good_run(), "a" * 40)


@pytest.mark.parametrize("fault", ["wrong-sha", "failed", "cancelled", "skipped", "missing-gate", "pending"])
def test_remote_success_cannot_hide_failure(fault):
    r = copy.deepcopy(good_run())
    if fault == "wrong-sha":
        r["headSha"] = "b" * 40
    elif fault == "missing-gate":
        r["jobs"].pop(0)
    elif fault == "pending":
        r["status"] = "in_progress"
    else:
        r["jobs"][1]["conclusion"] = fault
    with pytest.raises(ValueError):
        script("publish_private").validate_run(r, "a" * 40)


def test_timeout_does_not_change_successful_upload(tmp_path, monkeypatch):
    p = script("publish_private")
    monkeypatch.setattr(p, "gh_json", lambda *a, **k: [])
    tick = iter([0, 2])
    monkeypatch.setattr(p.time, "monotonic", lambda: next(tick))
    receipt = {"upload_status": "passed", "ci_status": "pending"}
    path = tmp_path / "receipt.json"
    with pytest.raises(TimeoutError):
        p.wait_ci("billlza/isac-world-model", "a" * 40, receipt, path, 1, tmp_path)
    assert json.loads(path.read_text())["upload_status"] == "passed"
    assert receipt["ci_status"] == "timed_out"


def test_clean_environment_strips_host_package_overrides(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/host-source")
    monkeypatch.setenv("UV_INDEX_URL", "https://unreviewed.invalid")
    monkeypatch.setenv("VIRTUAL_ENV", "/host-venv")
    env = script("clean_install").clean_env()
    assert not {"PYTHONPATH", "UV_INDEX_URL", "VIRTUAL_ENV"} & env.keys()
    assert env["PYTHONNOUSERSITE"] == "1"


def test_bootstrap_existing_environment_stops_before_install(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".venv").mkdir()
    target = tmp_path / "scripts/bootstrap.sh"
    target.write_bytes((ROOT / "scripts/bootstrap.sh").read_bytes())
    env = dict(os.environ, RECREATE_VENV="0")
    result = subprocess.run(["bash", str(target)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert not (tmp_path / ".bootstrap").exists()


def test_archive_restores_exact_history_without_rewriting_source(tmp_path):
    p = script("publish_private")
    origin = tmp_path / "original"
    origin.mkdir()
    for name in p.REQUIRED_FILES:
        path = origin / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (origin / ".gitignore").write_text("history.bundle\n")

    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=origin, capture_output=True, text=True, check=True
        ).stdout.strip()

    run("init", "-b", "main")
    run("add", ".")
    run(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--no-gpg-sign",
        "-m",
        "fixture",
    )
    sha = run("rev-parse", "HEAD")
    archive = tmp_path / "archive"
    archive.mkdir()
    for name in p.REQUIRED_FILES:
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((origin / name).read_bytes())
    run("bundle", "create", str(archive / "history.bundle"), "main")
    before = (archive / "uv.lock").read_bytes()
    p.restore_history(archive)
    assert p.local_commit(archive) == sha
    assert (archive / "uv.lock").read_bytes() == before
    assert not p.command(["git", "remote"], root=archive).stdout.strip()
