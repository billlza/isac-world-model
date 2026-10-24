import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest

from isac_wm.artifacts import read_checkpoint, save_checkpoint
from isac_wm.events import EventQueue, Observation

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repository_contracts():
    assert script("check_contracts").check() == []


@pytest.mark.parametrize("seed", [7301, 17, 99, 201, 599])
def test_python_queue_matches_native_protocol_oracle(seed):
    commands, expected = script("native_parity").make_transcript(seed)
    queue = EventQueue()
    for command, answer in zip(commands, expected):
        fields = command.split()
        # The Rust protocol also tests lexical validation. Compare semantic
        # commands in the exact integer range of float64 on the Python side.
        if any(not v.isascii() or not v.isdecimal() for v in fields[1:]):
            assert answer == "ERR"
            continue
        if fields[0] == "P" and len(fields) != 4:
            assert answer == "ERR"
            continue
        try:
            if fields[0] == "P":
                uid, cap, available = map(int, fields[1:])
                queue.push(Observation(uid, "radio", float(cap), float(available), np.zeros(4), 1))
                actual = "OK"
            else:
                # The final drain is larger than every generated observation;
                # no assertion is made about nanosecond precision near u64 max.
                ready = queue.release(float(fields[1]))
                actual = "READY " + ",".join(str(event.uid) for event in ready)
        except (ValueError, OverflowError):
            actual = "ERR"
        assert actual == answer


def test_ci_missing_tools_block_instead_of_pass(tmp_path, monkeypatch):
    ci = script("ci")
    monkeypatch.setattr(ci.shutil, "which", lambda _: None)
    assert ci.main(["--profile", "native", "--out", str(tmp_path)]) == 2
    import json

    report = json.loads((tmp_path / "ci-report.json").read_text())
    assert report["status"] == "blocked"
    assert all(step["status"] == "blocked" for step in report["steps"])


def test_ci_nonzero_command_is_failure(tmp_path):
    import os
    import sys

    ci = script("ci")
    result = ci.run_step("bad", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path, os.environ.copy())
    assert result["status"] == "failed" and result["exit_code"] == 7


@pytest.mark.parametrize("contents", ["x" * 64, "a" * 63, "a" * 129, "A" * 64])
def test_checkpoint_rejects_malformed_checksum(tmp_path, contents):
    p = tmp_path / "a.pt"
    save_checkpoint(p, {"schema_version": 2})
    p.with_suffix(".pt.sha256").write_text(contents)
    with pytest.raises(ValueError):
        read_checkpoint(p)


def test_checkpoint_rejects_symlink_checksum(tmp_path):
    p = tmp_path / "a.pt"
    save_checkpoint(p, {"schema_version": 2})
    checksum = p.with_suffix(".pt.sha256")
    target = tmp_path / "target"
    target.write_text(checksum.read_text())
    checksum.unlink()
    checksum.symlink_to(target)
    with pytest.raises(ValueError):
        read_checkpoint(p)


def test_checkpoint_rejects_duplicate_zip_names(tmp_path):
    p = tmp_path / "a.pt"
    with zipfile.ZipFile(p, "w") as archive:
        archive.writestr("dup", "first")
        with pytest.warns(UserWarning):
            archive.writestr("dup", "second")
    with pytest.raises(ValueError, match="Duplicate"):
        read_checkpoint(p)


def test_toolchain_probe_does_not_install(monkeypatch):
    from isac_wm import toolchain

    monkeypatch.setattr(toolchain.shutil, "which", lambda _: None)
    report = toolchain.inspect_toolchain()
    assert report["rust_required"] == "1.99.1"
    assert all(t["status"] == "missing" for t in report["tools"].values())


def test_toolchain_probe_handles_timeout(monkeypatch):
    import subprocess

    from isac_wm import toolchain

    monkeypatch.setattr(toolchain.shutil, "which", lambda name: "/bin/" + name)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("test", 5)

    monkeypatch.setattr(toolchain.subprocess, "run", timeout)
    assert all(t["status"] == "unavailable" for t in toolchain.inspect_toolchain()["tools"].values())


def test_ci_timeout_fails_instead_of_hanging(tmp_path):
    import os
    import sys

    ci = script("ci")
    result = ci.run_step(
        "timeout",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        os.environ.copy(),
        timeout=0.1,
    )
    assert result["status"] == "failed" and result["exit_code"] == 124
