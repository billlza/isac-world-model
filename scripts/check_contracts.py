"""Offline repository/CI policy checks. Not a general vulnerability scanner."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path = ROOT) -> list[str]:
    errors = []
    for path in sorted((root / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                errors.append(f"{path}:{node.lineno}: dynamic code execution")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "load"
            ):
                if not any(
                    k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True
                    for k in node.keywords
                ):
                    errors.append(f"{path}:{node.lineno}: torch.load must use weights_only=True")
    config = tomllib.loads((root / "pyproject.toml").read_text())
    version = ast.parse((root / "src/isac_wm/__init__.py").read_text())
    versions = [
        n.value.value
        for n in version.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    ]
    if versions != [config["project"]["version"]]:
        errors.append("Package version mismatch")
    cargo = tomllib.loads((root / "crates/event-core/Cargo.toml").read_text())
    if cargo["package"]["version"] != config["project"]["version"]:
        errors.append("Rust/Python package version mismatch")
    for path in sorted((root / ".github/workflows").glob("*.yml")):
        # BaseLoader keeps the GitHub key `on` as a string, unlike YAML 1.1 bool resolution.
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        triggers = workflow.get("on", {})
        if "pull_request_target" in triggers:
            errors.append(f"{path}: privileged PR trigger is forbidden")
        if workflow.get("permissions") != {"contents": "read"}:
            errors.append(f"{path}: require read-only default permissions")
        for name, job in workflow.get("jobs", {}).items():
            if not job.get("timeout-minutes"):
                errors.append(f"{path}/{name}: missing timeout")
            if job.get("continue-on-error") == "true":
                errors.append(f"{path}/{name}: required job cannot ignore errors")
            for step in job.get("steps", []):
                uses = step.get("uses", "")
                if uses and not uses.startswith("./") and not re.fullmatch(r"[^@]+@[a-f0-9]{40}", uses):
                    errors.append(f"{path}/{name}: action is not SHA-pinned: {uses}")
                if uses.startswith("actions/checkout@"):
                    if step.get("with", {}).get("persist-credentials") != "false":
                        errors.append(f"{path}/{name}: checkout must not persist credentials")
                if step.get("continue-on-error") == "true":
                    errors.append(f"{path}/{name}: step cannot ignore errors")
    return errors


if __name__ == "__main__":
    failures = check()
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"Repository contract check: {'FAILED' if failures else 'PASS'}")
    raise SystemExit(bool(failures))
