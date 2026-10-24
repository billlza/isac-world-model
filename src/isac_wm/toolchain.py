"""Read-only tool discovery; never installs software or triggers rustup downloads."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import tempfile


def inspect_toolchain() -> dict:
    report = {"rust_required": "1.99.1", "tools": {}, "packages": {}}
    for name in ("rustup", "cargo", "rustc", "cc", "gh", "uv"):
        path = shutil.which(name)
        if path is None:
            report["tools"][name] = {"status": "missing"}
            continue
        # Avoid a project-local toolchain file causing rustup auto-installation.
        command = [path, "--version"]
        if name in {"cargo", "rustc"} and shutil.which("rustup"):
            command = [shutil.which("rustup"), "run", "1.99.1", name, "--version"]
        try:
            process = subprocess.run(
                command, cwd=tempfile.gettempdir(), check=False, capture_output=True, text=True, timeout=5
            )
            report["tools"][name] = {
                "status": "available" if process.returncode == 0 else "unavailable",
                "path": path,
                "version": process.stdout.splitlines()[:1],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["tools"][name] = {"status": "unavailable", "path": path, "error": type(exc).__name__}
    for package in (
        "torch",
        "numpy",
        "pytest",
        "pytest-cov",
        "coverage",
        "ruff",
        "build",
        "pip",
        "uv",
        "setuptools",
        "wheel",
    ):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    return report
