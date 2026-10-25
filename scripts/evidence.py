"""Bind CI results to source and evidence bytes, not a free-standing green status.

Hashes detect changes; they do not authenticate the publisher or prove program safety.
The source scope includes build inputs and workflow/installer code. Research outputs
and prose-only documentation have separate release-archive integrity checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = (
    "pyproject.toml",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "uv.lock",
    ".python-version",
    "MANIFEST.in",
    ".gitignore",
    "README.md",
    "RIGHTS.md",
    "AGENTS.md",
    "CHANGELOG.md",
)
SOURCE_DIRS = ("src", "tests", "scripts", "crates", ".github", "ci")
REQUIRED_ALL = frozenset(
    {
        "contracts",
        "dependency-policy",
        "dependency-environment",
        "pip-check",
        "environment-snapshot",
        "lock-check",
        "python-tests",
        "integration",
        "distribution",
        "wheel-install",
        "ruff",
        "ruff-format",
        "dependency-audit",
        "rust-toolchain",
        "rustfmt",
        "clippy",
        "rust-tests",
        "native-build",
        "native-parity",
        "reproducible-install",
    }
)


def sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(root: Path = ROOT) -> dict[str, str]:
    files = [root / name for name in TOP_LEVEL]
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            if any(p == "__pycache__" or p.endswith(".egg-info") for p in path.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink in source scope: {path}")
            if path.is_file() and path.suffix in {
                ".py",
                ".rs",
                ".toml",
                ".yml",
                ".yaml",
                ".sh",
                ".txt",
                ".json",
                ".lock",
            }:
                files.append(path)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(set(files))}


def tree_digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Unsafe evidence path")
    child = root.joinpath(*path.parts)
    if not child.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes report directory")
    for parent in [child, *child.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symlink in evidence path")
    return child


def output_snapshot(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name != "ci-report.json"
    }


def verify(report_path: Path, root: Path = ROOT, *, require_all: bool = True) -> dict:
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != 2 or report.get("status") != "passed":
        raise ValueError("Need a passed schema-v2 CI report")
    steps = report.get("steps", [])
    names = [s.get("name") for s in steps]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicated CI steps")
    if require_all and (report.get("profile") != "all" or set(names) != REQUIRED_ALL):
        raise ValueError("Partial profile cannot satisfy the full release gate")
    if any(s.get("status") != "passed" or s.get("exit_code") != 0 for s in steps):
        raise ValueError("A required check did not pass")
    actual = source_snapshot(root)
    if report.get("source_sha256") != actual or report.get("source_tree_sha256") != tree_digest(actual):
        raise ValueError("Source changed since CI")
    artifacts = report.get("artifact_sha256", {})
    if not artifacts:
        raise ValueError("Missing evidence hashes")
    for step in steps:
        if step.get("log") not in artifacts:
            raise ValueError("Missing check log from evidence inventory")
    for relative, expected in artifacts.items():
        if sha256(safe_child(report_path.parent, relative)) != expected:
            raise ValueError(f"Evidence changed: {relative}")
    return {
        "status": "passed",
        "source_tree_sha256": tree_digest(actual),
        "source_files": len(actual),
        "evidence_files": len(artifacts),
        "checks": len(steps),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify(args.report, require_all=not args.allow_partial)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
