"""Cold-cache locked installation and a second, offline, isolated installation.

Uses the already selected CPython interpreter. Does not claim to reproduce the OS,
CPython build, or Rust distribution offline. No system site-packages are inherited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = """import importlib.metadata as m, json, pathlib, site, sys
import isac_wm, numpy, torch
prefix = pathlib.Path(sys.prefix).resolve()
assert pathlib.Path(isac_wm.__file__).resolve().is_relative_to(prefix)
assert pathlib.Path(numpy.__file__).resolve().is_relative_to(prefix)
assert pathlib.Path(torch.__file__).resolve().is_relative_to(prefix)
assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
torch.manual_seed(123)
x=torch.randn(8,4, requires_grad=True); y=x.square().sum(); y.backward()
assert torch.isfinite(x.grad).all()
print(json.dumps({'python':sys.version, 'prefix':str(prefix), 'module':isac_wm.__file__,
'packages':dict(sorted((d.metadata['Name'].lower().replace('_','-'),d.version) for d in m.distributions()))}))
"""


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("UV_", "PIP_", "PYTHON")) or key in {
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
        }:
            env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", UV_NO_PROGRESS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Do not overwrite an installation report")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "Python environment, locked packages and installed application; same OS/interpreter",
        "runs": [],
    }
    uv = shutil.which("uv")
    if not uv:
        raise FileNotFoundError("uv is required")
    original_lock = (ROOT / "uv.lock").read_bytes()
    report["lock_sha256"] = hashlib.sha256(original_lock).hexdigest()
    try:
        with tempfile.TemporaryDirectory(prefix="isac-reproduce-") as temp:
            work = Path(temp)
            source = work / "project"
            source.mkdir()
            # Explicit build inputs only; no .venv, user data, secrets or old results.
            for name in (
                "pyproject.toml",
                "uv.lock",
                ".python-version",
                "MANIFEST.in",
                "README.md",
                "RIGHTS.md",
                "CHANGELOG.md",
                "AGENTS.md",
                "Cargo.toml",
                "Cargo.lock",
                "rust-toolchain.toml",
            ):
                shutil.copy2(ROOT / name, source / name)
            shutil.copytree(
                ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
            )
            env = clean_env()
            env["UV_CACHE_DIR"] = str(work / "empty-cache")
            env["UV_PYTHON_DOWNLOADS"] = "never"
            for mode in ("cold-online", "warm-offline"):
                prefix = work / mode
                env["UV_PROJECT_ENVIRONMENT"] = str(prefix)
                command = [
                    uv,
                    "sync",
                    "--locked",
                    "--extra",
                    "dev",
                    "--no-editable",
                    "--project",
                    str(source),
                    "--python",
                    sys.executable,
                ]
                if mode == "warm-offline":
                    command.append("--offline")
                started = time.perf_counter()
                with (out.parent / f"{mode}.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=work,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                    python = str(prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
                    subprocess.run(
                        [python, "-m", "pip", "check"],
                        cwd=work,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=60,
                    )
                    probe = subprocess.run(
                        [python, "-I", "-c", PROBE],
                        cwd=work,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,
                    )
                    # Run application code outside the source checkout in both environments.
                    subprocess.run(
                        [
                            python,
                            "-I",
                            "-m",
                            "isac_wm",
                            "train",
                            "--out",
                            str(work / f"{mode}-training"),
                            "--epochs",
                            "1",
                            "--episodes",
                            "4",
                            "--steps",
                            "12",
                            "--batch-size",
                            "2",
                            "--device",
                            "cpu",
                            "--threads",
                            "2",
                        ],
                        cwd=work,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=180,
                    )
                result = json.loads(probe.stdout)
                result.update(mode=mode, status="passed", elapsed_s=time.perf_counter() - started)
                report["runs"].append(result)
                if (source / "uv.lock").read_bytes() != original_lock:
                    raise ValueError("Locked installation changed uv.lock")
            a, b = report["runs"]
            if a["packages"] != b["packages"]:
                raise ValueError("Cold and offline package inventories differ")
            report["package_count"] = len(a["packages"])
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "package_count": report["package_count"],
                "lock_sha256": report["lock_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
