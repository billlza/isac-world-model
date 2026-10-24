"""CI runner: success=0, failed check=1, unavailable prerequisite=2.

Each selected check is mandatory. No missing tool silently becomes a pass.
Network provisioning is deliberately separate (scripts/bootstrap.sh).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import signal
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_step(name, command, out, env, timeout=300):
    start = time.perf_counter()
    log = out / f"{name}.log"
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Terminate the whole owned process group, including smoke children.
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                raise
        status = "passed" if code == 0 else "failed"
    except (FileNotFoundError, PermissionError) as exc:
        log.write_text(str(exc) + "\n")
        status, code = "blocked", None
    except subprocess.TimeoutExpired:
        with log.open("a") as stream:
            stream.write("\nCI step exceeded its deadline\n")
        status, code = "failed", 124
    print(f"{name}: {status}", flush=True)
    return {"name": name, "status": status, "exit_code": code,
            "duration_s": time.perf_counter() - start, "command": command, "log": log.name}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["python", "native", "quality", "all"], default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "validation/local-ci")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("CI output directory must be empty; choose a new path to preserve prior evidence")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    python = sys.executable
    specs = []
    profile = args.profile
    if profile in {"python", "all"}:
        specs += [
            ("contracts", [python, "scripts/check_contracts.py"], 60),
            ("python-tests", [python, "-m", "pytest", "-p", "pytest_cov", "--cov=isac_wm", "--cov-fail-under=88",
                              "--cov-report=term-missing", f"--cov-report=xml:{out / 'coverage.xml'}",
                              f"--junitxml={out / 'pytest.xml'}"], 180),
            ("integration", [python, "scripts/ci_smoke.py", "--out", str(out / "smoke")], 300),
            ("distribution", [python, "scripts/build_dist.py", "--out", str(out / "dist")], 90),
            ("wheel-install", [python, "scripts/test_wheel.py", "--dist", str(out / "dist")], 120),
        ]
    if profile in {"quality", "all"}:
        specs += [("ruff", [python, "-m", "ruff", "check", "src", "tests", "scripts"], 60)]
    if profile in {"native", "all"}:
        specs += [
            ("rustfmt", ["cargo", "fmt", "--all", "--", "--check"], 120),
            ("clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--locked", "--offline", "--", "-D", "warnings"], 180),
            ("rust-tests", ["cargo", "test", "--workspace", "--locked", "--offline"], 180),
            ("native-build", ["cargo", "build", "--workspace", "--release", "--locked", "--offline"], 180),
            ("native-parity", [python, "scripts/native_parity.py", "--binary",
                               str(ROOT / "target/release/isac-event-replay"), "--out", str(out / "native-parity.json")], 120),
        ]
    def source_snapshot():
        files = [p for base in ("src", "tests", "scripts", "crates", ".github", "ci")
                 for p in (ROOT / base).rglob("*")
                 if p.is_file() and p.suffix in {".py", ".rs", ".toml", ".yml", ".sh", ".txt"}
                 and not any(part.endswith(".egg-info") or part == "__pycache__" for part in p.parts)]
        files += [ROOT / p for p in ("pyproject.toml", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    source_before = source_snapshot()
    results = []
    for name, command, timeout in specs:
        blocked = None
        if name == "ruff" and importlib.util.find_spec("ruff") is None:
            blocked = "ruff is not installed"
        if name in {"rustfmt", "clippy", "rust-tests", "native-build"} and not shutil.which("cargo"):
            blocked = "cargo is not installed"
        if name == "native-parity" and not any(r["name"] == "native-build" and r["status"] == "passed" for r in results):
            blocked = "a successful native build is required"
        if blocked:
            (out / f"{name}.log").write_text(blocked + "\n")
            results.append({"name": name, "status": "blocked", "reason": blocked,
                            "command": command, "log": f"{name}.log"})
            print(f"{name}: blocked ({blocked})", flush=True)
        else:
            results.append(run_step(name, command, out, env, timeout))
    if source_snapshot() != source_before:
        results.append({"name": "source-stability", "status": "failed", "reason": "Source changed during CI"})
    status = ("failed" if any(r["status"] == "failed" for r in results) else
              "blocked" if any(r["status"] == "blocked" for r in results) else "passed")
    versions = {}
    for package in ("torch", "numpy", "pytest", "pytest-cov", "ruff", "setuptools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"schema_version": 1, "profile": profile, "status": status,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "packages": versions, "source_sha256": source_before, "steps": results}
    (out / "ci-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{profile} CI: {status.upper()}; report={out / 'ci-report.json'}")
    return {"passed": 0, "failed": 1, "blocked": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
