import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location("dep_policy", ROOT / "scripts/dependencies.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_and_all_declared_pins_agree():
    assert module().check_policy() == []


@pytest.mark.parametrize(
    "found,wanted,expected",
    [
        ("2.14.1+cpu", "2.14.1", True),
        ("2.14.1+cu130", "2.14.1", True),
        ("2.14.0", "2.14.1", False),
        ("2.15.0.dev1", "2.14.1", False),
        ("nonsense", "2.14.1", False),
        ("0.17.2", "0.17.2", True),
    ],
)
def test_backend_suffix_does_not_hide_version_drift(found, wanted, expected):
    assert module().matches(found, wanted) is expected


def test_canonical_names():
    assert module().canonical("PyYAML") == "pyyaml"
    assert module().canonical("typing_extensions") == "typing-extensions"


def test_manifest_rejects_prereleases(tmp_path):
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/versions.json").write_text(
        json.dumps({"schema_version": 1, "packages": {"torch": "2.15.0rc1"}})
    )
    with pytest.raises(ValueError, match="Non-stable"):
        module().load_manifest(tmp_path)


def test_registry_host_is_restricted():
    with pytest.raises(ValueError, match="official PyPI"):
        module().fetch_json("https://untrusted.invalid/pypi/torch/json")


def test_latest_audit_does_not_install(monkeypatch):
    mod = module()
    monkeypatch.setattr(
        mod, "fetch_json", lambda _: {"info": {"version": "1.0.0"}, "urls": [{"yanked": False}]}
    )
    manifest = {"packages": {"toy": "1.0.0"}}
    assert mod.latest(manifest)["status"] == "current"
    manifest["packages"]["toy"] = "0.9.0"
    assert mod.latest(manifest)["status"] == "updates_available"


@pytest.mark.parametrize("files", [[], [{"yanked": True}]])
def test_yanked_or_empty_release_does_not_pass(files, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, "fetch_json", lambda _: {"info": {"version": "1.0.0"}, "urls": files})
    assert mod.latest({"packages": {"toy": "1.0.0"}})["status"] == "blocked"


def test_registry_outage_is_blocked(monkeypatch):
    mod = module()

    def fail(_):
        raise OSError("offline")

    monkeypatch.setattr(mod, "fetch_json", fail)
    assert mod.latest({"packages": {"toy": "1.0.0"}})["status"] == "blocked"


def test_bootstrap_never_changes_global_rust_default():
    script = (ROOT / "scripts/bootstrap.sh").read_text()
    assert 'export CARGO_HOME="$ROOT/.toolchains/cargo"' in script
    assert 'export RUSTUP_HOME="$ROOT/.toolchains/rustup"' in script
    assert "--no-modify-path" in script
    assert "uv sync --locked" in script
    assert "sudo " not in script


def test_dependency_policy_detects_unpinned_direct_dependency(tmp_path):
    import shutil

    for relative in [
        "ci/versions.json",
        "ci/constraints.txt",
        "pyproject.toml",
        "rust-toolchain.toml",
        ".python-version",
    ]:
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dst)
    p = tmp_path / "pyproject.toml"
    p.write_text(p.read_text().replace("numpy==2.5.4", "numpy>=2.5.4"))
    assert any("Unpinned" in e for e in module().check_policy(tmp_path))


def test_audit_skip_is_not_a_pass():
    spec = importlib.util.spec_from_file_location("audit_policy", ROOT / "scripts/audit_dependencies.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.validate_report({"dependencies": []})
    assert mod.validate_report({"dependencies": [{"name": "torch", "skip_reason": "not found"}]})
    assert mod.validate_report({"dependencies": [{"name": "x", "vulns": [{"id": "TEST"}]}]})
    assert not mod.validate_report({"dependencies": [{"name": "x", "vulns": []}]})
