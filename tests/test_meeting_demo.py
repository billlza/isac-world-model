"""Regression coverage for clean-clone demos and generated JavaScript."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from isac_wm.viewer import TEMPLATE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("group_demo", ROOT / "scripts/group_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_viewer_embeds_escaped_javascript_newline():
    assert "D.limitations.join('\\n')" in TEMPLATE
    assert "D.limitations.join('\n')" not in TEMPLATE


def test_demo_missing_checkpoint_fails_before_creating_output(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Checkpoint"):
        demo.prepare(out, checkpoint=tmp_path / "absent.pt")
    assert not out.exists()


def test_demo_never_overwrites_previous_run(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("evidence")
    with pytest.raises(FileExistsError):
        demo.prepare(out)
    assert (out / "keep.txt").read_text() == "evidence"


@pytest.mark.parametrize("timeout", [False, True])
def test_demo_failure_retains_manifest_and_nonzero_result(tmp_path, monkeypatch, timeout):
    def fail(*args, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired(args[0], 1800)
        return subprocess.CompletedProcess(args[0], 3)

    monkeypatch.setattr(demo.subprocess, "run", fail)
    out = tmp_path / "failure"
    assert demo.main(["--out", str(out), "--quick"]) == 1
    report = json.loads((out / "demo-manifest.json").read_text())
    assert report["status"] == "failed"
    assert (out / "training.log").exists()
    assert not (out / "index.html").exists()


def test_demo_index_contains_all_cases_and_escapes_labels(tmp_path):
    summaries = {name: {"<unsafe>": {"position_rmse_m": 0.5}} for name, _, _ in demo.CASES}
    demo.write_index(tmp_path, summaries)
    text = (tmp_path / "index.html").read_text()
    assert "<unsafe>" not in text
    assert "&lt;unsafe&gt;" in text
    assert all(f"{name}/replay.html" in text for name, _, _ in demo.CASES)
    assert "not Mac/GPU benchmarks" in text


def test_browser_is_required_by_remote_ci():
    import yaml

    workflow = yaml.load((ROOT / ".github/workflows/tests.yml").read_text(), Loader=yaml.BaseLoader)
    assert "viewer" in workflow["jobs"]["ci-gate"]["needs"]
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["viewer"]["steps"])
    assert "--locked" in commands
    assert "scripts/browser_smoke.py" in commands
    assert "--extra browser" in commands


def test_demo_rejects_checkpoint_symlink(tmp_path):
    target = tmp_path / "target.pt"
    target.write_bytes(b"example")
    link = tmp_path / "alias.pt"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="Checkpoint"):
        demo.prepare(tmp_path / "out", checkpoint=link)
