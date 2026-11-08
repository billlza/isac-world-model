#!/bin/bash
#SBATCH --account=ncc_users
#SBATCH --partition=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=isac-ncc-runtime
#SBATCH --output=slurm-%j.out
# Install the explicitly selected NCC runtime in a new user-owned directory.
# This CPU job does not establish GPU compatibility or execute CUDA kernels.
set -euo pipefail
umask 077
exec /usr/bin/python3 -I -u - <<'PY_NCC_RUNTIME'
"""Standard-library installer; external commands are injectable in unit tests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import pwd
import re
import shlex
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {"python": "3.12.3", "numpy": "2.3.4", "torch": "2.9.0+cu128", "torch_cuda": "12.8"}
PROFILE = "ncc-py3123-torch290-cu128"
PROBE = r'''
import json, platform, sys
import numpy, torch
print(json.dumps({
    "python": platform.python_version(),
    "numpy": numpy.__version__, "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "cuda_build": bool(torch.backends.cuda.is_built()),
    "interpreter": sys.executable, "prefix": sys.prefix,
    "base_prefix": sys.base_prefix, "platform": platform.platform(),
    "machine": platform.machine(), "libc": list(platform.libc_ver()),
    "numpy_file": numpy.__file__, "torch_file": torch.__file__,
}))
'''


class SetupError(RuntimeError):
    """The requested NCC installation contract was not met."""


class SetupInterrupted(SetupError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Runtime preparation interrupted by signal {signum}")


def _utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_environment(env):
    # Do not inherit an alternate package index, credentials embedded in a pip
    # index, user-site injection, or a different Python's library path.
    clean = {key: value for key, value in env.items()
             if not key.startswith(("PIP_", "PYTHON"))}
    clean.update(PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE="1", LC_ALL="C",
                 OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    return clean


def _capture(command, *, runner, env, timeout=30):
    completed = runner(command, env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout, check=False)
    if completed.returncode:
        # Avoid echoing arbitrary subprocess output or inherited credentials.
        raise SetupError(f"{Path(command[0]).name} command failed with exit {completed.returncode}")
    return completed.stdout.strip()


def _allocation(env, *, runner, hostname):
    job_id = env.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id) or env.get("SLURM_ARRAY_JOB_ID"):
        raise SetupError("Submit runtime preparation as a single Slurm CPU job")
    if env.get("SLURM_JOB_PARTITION") != "cpu":
        raise SetupError("Runtime preparation requires the cpu partition")
    text = _capture(["scontrol", "show", "job", "-o", job_id], runner=runner, env=env)
    fields = dict(token.split("=", 1) for token in shlex.split(text) if "=" in token)
    current_user = pwd.getpwuid(os.getuid())
    expected = {"JobId": job_id, "JobState": "RUNNING", "Partition": "cpu",
                "Account": "ncc_users", "QOS": "debug",
                "UserId": f"{current_user.pw_name}({current_user.pw_uid})"}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise SetupError("Slurm controller did not confirm this user's running NCC CPU setup job")
    if fields.get("NumNodes") != "1" or fields.get("NumCPUs") != "2":
        raise SetupError("Runtime preparation requires one node and two allocated CPUs")
    nodes = fields.get("NodeList", "")
    if not nodes or nodes == "(null)":
        raise SetupError("Slurm allocation has no compute node")
    allocated = _capture(["scontrol", "show", "hostnames", nodes], runner=runner, env=env)
    allowed = {name.split(".", 1)[0] for name in allocated.splitlines()}
    if hostname().split(".", 1)[0] not in allowed:
        raise SetupError("Runtime setup is not executing on its allocated compute node")
    if "gres/gpu" in fields.get("AllocTRES", ""):
        raise SetupError("Use a CPU allocation for package installation")
    return {"job_id": job_id, "partition": "cpu", "account": "ncc_users", "qos": "debug",
            "user": current_user.pw_name, "node_list": nodes, "allocated_tres": fields.get("AllocTRES")}


def _source(project, pin, *, runner, env):
    command = ["git", "-C", str(project)]
    top = _capture(command + ["rev-parse", "--show-toplevel"], runner=runner, env=env)
    if Path(top).resolve() != project:
        raise SetupError("ISAC_PROJECT_ROOT is not the checkout root")
    if _capture(command + ["rev-parse", "HEAD"], runner=runner, env=env) != pin:
        raise SetupError("Checkout differs from ISAC_NCC_PIN")
    _capture(command + ["diff", "--quiet", "HEAD", "--"], runner=runner, env=env)
    untracked = _capture(command + ["ls-files", "--others", "--exclude-standard", "--",
                                    "src", "scripts", "*.py"], runner=runner, env=env)
    if untracked:
        raise SetupError("Untracked source could shadow the pinned implementation")


def _system_runtime():
    return {"python": platform.python_version(), "interpreter": sys.executable,
            "platform": platform.platform(), "system": platform.system(),
            "machine": platform.machine(), "libc": list(platform.libc_ver())}


def _validate_system(observed):
    if observed.get("python") != EXPECTED["python"] or observed.get("interpreter") != "/usr/bin/python3":
        raise SetupError("NCC setup requires /usr/bin/python3 version 3.12.3; no interpreter fallback")
    if observed.get("system") != "Linux" or observed.get("machine") != "x86_64":
        raise SetupError("NCC setup requires Linux x86_64")
    libc = observed.get("libc", [])
    if len(libc) != 2 or libc[0] != "glibc" or not re.fullmatch(r"[0-9]+\.[0-9]+", libc[1]):
        raise SetupError("Cannot establish the NCC glibc version")
    if tuple(map(int, libc[1].split("."))) < (2, 28):
        raise SetupError("The selected CUDA wheel requires glibc 2.28 or newer")


def _write_record(root, record):
    record["updated_utc"] = _utc()
    temporary = root / ".runtime.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(root / "runtime.json")


def _logged(command, logfile, *, runner, env, timeout):
    with logfile.open("x", encoding="utf-8") as log:
        completed = runner(command, env=env, text=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=False)
    if completed.returncode:
        raise SetupError(f"Command failed with exit {completed.returncode}; see {logfile.name}")


def _fingerprint(path):
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_setup(*, env=None, runner=subprocess.run, system_probe=_system_runtime,
              hostname=socket.gethostname):
    environment = dict(os.environ if env is None else env)
    pin = environment.get("ISAC_NCC_PIN", "")
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise SetupError("ISAC_NCC_PIN must be a full 40-character source commit")
    paths = {}
    for name in ("ISAC_PROJECT_ROOT", "ISAC_RUNS_DIR", "ISAC_NCC_RUNTIME_ROOT"):
        value = environment.get(name, "")
        if not value or not Path(value).is_absolute():
            raise SetupError(f"{name} must be an absolute directory")
        paths[name] = Path(value).resolve()
    project = paths["ISAC_PROJECT_ROOT"]
    root = paths["ISAC_NCC_RUNTIME_ROOT"]
    if root.is_relative_to(project):
        raise SetupError("The NCC runtime directory must be outside the source checkout")
    clean = _clean_environment(environment)
    allocation = _allocation(clean, runner=runner, hostname=hostname)
    _source(project, pin, runner=runner, env=clean)
    # Exclusive ownership precedes all writes. Failed or ready environments are
    # never reused, removed, repaired in place, or silently overwritten.
    # mkdir the original path so an existing dangling leaf symlink is rejected
    # too; resolving the path first must not silently create its target.
    Path(environment["ISAC_NCC_RUNTIME_ROOT"]).mkdir(mode=0o700)
    venv = root / "venv"
    python = venv / "bin" / "python"
    record = {"schema_version": 1, "status": "running", "stage": "system-check",
              "profile": PROFILE, "created_utc": _utc(), "git_commit": pin,
              "slurm_setup_job_id": allocation["job_id"], "slurm": allocation,
              "source_directory": str(project), "runs_directory": str(paths["ISAC_RUNS_DIR"]),
              "runtime_root": str(root), "venv_directory": str(venv),
              "requested_runtime": EXPECTED.copy(), "cuda_execution_validated": False}
    try:
        _write_record(root, record)
        record["system_runtime"] = system_probe()
        _validate_system(record["system_runtime"])
        record["stage"] = "venv"
        _write_record(root, record)
        _logged(["/usr/bin/python3", "-I", "-m", "venv", str(venv)], root / "venv.log",
                runner=runner, env=clean, timeout=120)
        pip = [str(python), "-I", "-m", "pip", "--disable-pip-version-check", "--no-input", "--no-cache-dir"]
        record["pip_version"] = _capture(pip + ["--version"], runner=runner, env=clean)
        reports = []
        installs = (("numpy", "numpy==2.3.4", "https://pypi.org/simple"),
                    ("torch", "torch==2.9.0+cu128", "https://download.pytorch.org/whl/cu128"))
        for name, requirement, index in installs:
            record["stage"] = f"install-{name}"
            _write_record(root, record)
            report = root / f"{name}-install.json"
            command = pip + ["install", "--only-binary=:all:", "--index-url", index,
                             "--report", str(report), requirement]
            _logged(command, root / f"{name}-install.log", runner=runner, env=clean, timeout=1200)
            manifest = json.loads(report.read_text(encoding="utf-8"))
            if not isinstance(manifest.get("install"), list):
                raise SetupError(f"Missing pip artifact provenance for {name}")
            reports.append({**_fingerprint(report), "index_url": index, "requirement": requirement})
        record["installation_reports"] = reports
        record["stage"] = "verify-imports"
        _write_record(root, record)
        _logged(pip + ["check"], root / "pip-check.txt", runner=runner, env=clean, timeout=120)
        freeze = _capture(pip + ["freeze", "--all"], runner=runner, env=clean, timeout=120) + "\n"
        (root / "requirements.freeze.txt").write_text(freeze, encoding="utf-8")
        observed = json.loads(_capture([str(python), "-I", "-c", PROBE],
                                       runner=runner, env=clean, timeout=120))
        record["runtime"] = observed
        if any(observed.get(key) != value for key, value in EXPECTED.items()):
            raise SetupError("Installed runtime differs from the explicitly selected NCC versions")
        if observed.get("cuda_build") is not True:
            raise SetupError("Installed PyTorch is not the requested CUDA build")
        if observed.get("interpreter") != str(python) or observed.get("prefix") != str(venv):
            raise SetupError("Imports did not use this job's new virtual environment")
        if observed.get("prefix") == observed.get("base_prefix"):
            raise SetupError("The interpreter is not in a virtual environment")
        for name in ("numpy_file", "torch_file"):
            if not observed.get(name) or not Path(observed[name]).resolve().is_relative_to(venv):
                raise SetupError("Scientific packages were imported from outside the new environment")
        # Freeze and installation reports record exact resolved transitive
        # dependencies and downloaded artifact hashes. Freeze alone is not a
        # portable, hash-enforced lock file.
        record["dependencies"] = freeze.splitlines()
        record["freeze"] = _fingerprint(root / "requirements.freeze.txt")
        record["pip_check"] = _fingerprint(root / "pip-check.txt")
        _source(project, pin, runner=runner, env=clean)
        record.update(status="ready", stage="done", exit_code=0)
        _write_record(root, record)
        print(json.dumps({"status": "ready", "runtime_manifest": str(root / "runtime.json"),
                          "interpreter": str(python), "cuda_execution_validated": False}), flush=True)
        return record
    except BaseException as failure:
        record.update(status="failed", exit_code=(128 + failure.signum)
                      if isinstance(failure, SetupInterrupted) else 1,
                      failure={"type": type(failure).__name__, "message": str(failure)})
        _write_record(root, record)
        raise


def main():
    def interrupted(signum, _frame):
        raise SetupInterrupted(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run_setup()
    except Exception as failure:
        print(f"NCC runtime setup failed: {failure}", file=sys.stderr, flush=True)
        raise SystemExit(128 + failure.signum if isinstance(failure, SetupInterrupted) else 1) from None


if __name__ == "__main__":
    main()
PY_NCC_RUNTIME
