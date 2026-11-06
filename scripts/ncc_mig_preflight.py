"""Verify one explicitly allocated NCC 1g.10gb MIG instance before training.

This does not install packages, change GPU visibility, or submit jobs. The small
CUDA AdamW check validates execution only; the actual A512 pilot is a separate
stage. In particular, whole-H200 requirements do not apply to this target.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import io
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "1g.10gb"
RUNTIME_PROFILE = "ncc-py3123-torch290-cu128"
VERSIONS = {"python": "3.12.3", "numpy": "2.3.4", "torch": "2.9.0+cu128", "torch_cuda": "12.8"}
GPU_TRES = "gres/gpu:1g.10gb"


class PreflightError(RuntimeError):
    """A required allocation, source, runtime, or CUDA check failed."""


def _atomic_write(path: Path, record: dict[str, Any], *, exclusive: bool = False) -> None:
    """Publish complete JSON atomically; initial publication cannot replace a file."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary_path, path)
        else:
            os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _command(runner, args: list[str], inventory: list[dict[str, Any]]) -> str:
    observation: dict[str, Any] = {"argv": args}
    inventory.append(observation)
    try:
        result = runner(args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=20)
    except subprocess.TimeoutExpired as error:
        observation.update(timed_out=True, timeout_s=20)
        raise PreflightError(f"Command exceeded 20 seconds: {args[0]}") from error
    observation.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode:
        raise PreflightError(f"Command failed ({result.returncode}): {args[0]}: {result.stderr.strip()}")
    return result.stdout.strip()


def _tres(value: str) -> dict[str, str]:
    result = {}
    for entry in value.split(","):
        if "=" not in entry:
            continue
        name, count = entry.split("=", 1)
        if name in result:
            raise PreflightError(f"Duplicate TRES entry: {name}")
        result[name] = count
    return result


def _one_mig_tres(value: str, label: str) -> None:
    entries = _tres(value)
    typed = {key: count for key, count in entries.items() if key.startswith("gres/gpu:")}
    if typed != {GPU_TRES: "1"} or entries.get("gres/gpu") != "1":
        raise PreflightError(f"{label} must identify exactly one gpu:1g.10gb and no other GPU")


def _allocation(env, runner, hostname: str, inventory) -> dict[str, Any]:
    job_id = env.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal():
        raise PreflightError("Run inside an allocated NCC Slurm job, not on a login node")
    host = hostname.split(".", 1)[0]
    if not re.fullmatch(r"gpu[0-9]+", host):
        raise PreflightError(f"Expected an NCC GPU compute node; current host is {hostname!r}")
    if env.get("SLURM_JOB_PARTITION") != "res-gpu-small":
        raise PreflightError("This target requires the NCC res-gpu-small partition")
    if not env.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise PreflightError("Slurm must provide CUDA_VISIBLE_DEVICES; do not set GPU IDs manually")
    query_id = job_id
    if "SLURM_ARRAY_JOB_ID" in env or "SLURM_ARRAY_TASK_ID" in env:
        array, task = env.get("SLURM_ARRAY_JOB_ID", ""), env.get("SLURM_ARRAY_TASK_ID", "")
        if not array.isdecimal() or not task.isdecimal():
            raise PreflightError("Incomplete Slurm array identity")
        query_id = f"{array}_{task}"
    raw = _command(runner, ["scontrol", "show", "job", "--oneliner", query_id], inventory)
    if len(raw.splitlines()) != 1:
        raise PreflightError("scontrol must identify exactly one allocation")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", raw))
    if fields.get("JobId") != job_id:
        raise PreflightError("scontrol returned a different job ID")
    if query_id != job_id and (fields.get("ArrayJobId"), fields.get("ArrayTaskId")) != (array, task):
        raise PreflightError("scontrol returned a different Slurm array element")
    expected = {"JobState": "RUNNING", "Partition": "res-gpu-small", "Account": "ncc_users"}
    for name, value in expected.items():
        if fields.get(name) != value:
            raise PreflightError(f"Slurm {name} must be {value}; observed {fields.get(name)!r}")
    owner = re.fullmatch(r"([^()\s]+)\(([0-9]+)\)", fields.get("UserId", ""))
    if owner is None or int(owner.group(2)) != os.getuid():
        raise PreflightError("Slurm UserId must match this process's real user ID")
    if env.get("SLURM_JOB_ACCOUNT", "ncc_users") != "ncc_users":
        raise PreflightError("SLURM_JOB_ACCOUNT contradicts the NCC ncc_users allocation")
    _one_mig_tres(fields.get("ReqTRES", ""), "Requested TRES")
    _one_mig_tres(fields.get("AllocTRES", ""), "Allocated TRES")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise PreflightError("The MIG check requires exactly one allocated node")
    nodes = _command(runner, ["scontrol", "show", "hostnames", fields.get("NodeList", "")], inventory)
    if nodes.splitlines() != [host]:
        raise PreflightError("The current compute host is not the sole node in this allocation")
    if env.get("SLURMD_NODENAME", host).split(".", 1)[0] != host:
        raise PreflightError("SLURMD_NODENAME does not match the current compute host")
    return {
        "job_id": job_id,
        "query_id": query_id,
        "state": fields["JobState"],
        "partition": fields["Partition"],
        "account": fields["Account"],
        "user": owner.group(1),
        "user_id": int(owner.group(2)),
        "qos": fields.get("QOS"),
        "node": host,
        "requested_tres": fields["ReqTRES"],
        "allocated_tres": fields["AllocTRES"],
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "slurm_job_gpus": env.get("SLURM_JOB_GPUS"),
    }


def _source(expected_commit: str, runner, inventory) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise PreflightError("Expected commit must be an exact lowercase 40-character Git SHA")
    top = _command(runner, ["git", "rev-parse", "--show-toplevel"], inventory)
    if Path(top).resolve() != ROOT.resolve():
        raise PreflightError("The script must run from its own Git repository")
    observed = _command(runner, ["git", "rev-parse", "HEAD"], inventory)
    if observed != expected_commit:
        raise PreflightError(f"Git commit mismatch: expected {expected_commit}, observed {observed}")
    try:
        _command(runner, ["git", "diff", "--quiet", "HEAD", "--"], inventory)
    except PreflightError as error:
        raise PreflightError(f"Tracked source tree is not clean or could not be checked: {error}") from error
    untracked = _command(
        runner,
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "*.py"],
        inventory,
    )
    if untracked:
        raise PreflightError(
            "Source tree is not clean; untracked source paths require review before training"
        )
    return {
        "git_commit": observed,
        "source_tree_clean": True,
        "source_check_scope": "All tracked files and untracked paths in src/, scripts/, or *.py",
        "repository": str(ROOT),
    }


def _runtime_manifest(path: Path, expected_commit: str) -> dict[str, Any]:
    if path.stat().st_size > 1024 * 1024:
        raise PreflightError("Runtime manifest exceeds the 1 MiB evidence limit")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "ready":
        raise PreflightError("Runtime manifest must have status=ready")
    if data.get("schema_version") != 1 or data.get("profile") != RUNTIME_PROFILE:
        raise PreflightError("Runtime manifest schema/profile does not match this NCC runtime")
    if data.get("git_commit") != expected_commit:
        raise PreflightError("Runtime manifest Git pin differs from the expected commit")
    recorded = data.get("runtime", {})
    if not isinstance(recorded, dict) or any(recorded.get(key) != value for key, value in VERSIONS.items()):
        raise PreflightError("Runtime manifest versions do not match the fixed NCC environment")
    if recorded.get("cuda_build") is not True:
        raise PreflightError("Runtime manifest does not establish a CUDA build")
    if recorded.get("interpreter") != sys.executable or recorded.get("prefix") != sys.prefix:
        raise PreflightError("This Python interpreter/prefix differs from the ready runtime manifest")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "profile": data["profile"],
        "git_commit": data["git_commit"],
        "status": data["status"],
    }


def _cuda_identity() -> dict[str, Any]:
    """Get CUDA ordinal zero's MIG-aware UUID, not a guessed physical UUID.

    NVIDIA cuDeviceGetUuid_v2 returns the subscribed MIG compute-instance UUID.
    cuDeviceGetUuid (without _v2) is intentionally not used as a fallback.
    """

    class CUuuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    library = ctypes.CDLL("libcuda.so.1")

    def call(name, argtypes, *args):
        function = getattr(library, name)
        function.argtypes, function.restype = argtypes, ctypes.c_int
        result = function(*args)
        if result != 0:
            raise PreflightError(f"CUDA Driver API {name} failed with code {result}")

    call("cuInit", [ctypes.c_uint], 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    if count.value != 1:
        raise PreflightError("Exactly one CUDA Driver API device must be visible")
    device = ctypes.c_int()
    call("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
    identifier = CUuuid()
    call("cuDeviceGetUuid_v2", [ctypes.POINTER(CUuuid), ctypes.c_int], ctypes.byref(identifier), device)
    total = ctypes.c_size_t()
    call("cuDeviceTotalMem_v2", [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int], ctypes.byref(total), device)
    driver = ctypes.c_int()
    call("cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)], ctypes.byref(driver))
    return {
        "uuid": str(uuid.UUID(bytes=bytes(identifier.bytes))),
        "device_count": count.value,
        "total_memory_bytes": total.value,
        "driver_api_version": driver.value,
        "uuid_api": "cuDeviceGetUuid_v2",
    }


def _mig_inventory(raw: str) -> dict[str, dict[str, Any]]:
    parents, instances, seen = {}, {}, set()
    parent = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        physical = re.fullmatch(r"GPU ([0-9]+): (.+) \(UUID: (GPU-[0-9a-fA-F-]{36})\)", line)
        child = re.fullmatch(r"\s+MIG (\S+)\s+Device\s+([0-9]+):\s+\(UUID:\s+(MIG-[0-9a-fA-F-]{36})\)", line)
        if physical:
            index, name, identifier = physical.groups()
            identifier = "GPU-" + str(uuid.UUID(identifier.removeprefix("GPU-")))
            parent = {"index": int(index), "name": name, "uuid": identifier}
            parents[identifier] = parent
        elif child and parent is not None:
            profile, index, identifier = child.groups()
            identifier = "MIG-" + str(uuid.UUID(identifier.removeprefix("MIG-")))
            instances[identifier] = {"profile": profile, "index": int(index), "parent": dict(parent)}
        else:
            raise PreflightError("Cannot establish MIG/physical parent hierarchy from nvidia-smi -L")
        if identifier in seen:
            raise PreflightError("Duplicate device UUID in nvidia-smi -L")
        seen.add(identifier)
    return {"parents": parents, "instances": instances}


def _hardware(torch, runner, identity_probe, inventory) -> dict[str, Any]:
    raw = _command(runner, ["nvidia-smi", "-L"], inventory)
    listed = _mig_inventory(raw)
    if torch.cuda.device_count() != 1:
        raise PreflightError("Exactly one PyTorch CUDA device must be visible")
    identity = identity_probe()
    inventory.append({"api": "CUDA Driver API identity", "observation": identity})
    if identity.get("device_count") != 1 or identity.get("uuid_api") != "cuDeviceGetUuid_v2":
        raise PreflightError("MIG identity requires one device and cuDeviceGetUuid_v2 evidence")
    normalized = str(uuid.UUID(identity["uuid"]))
    mig_uuid = "MIG-" + normalized
    if mig_uuid not in listed["instances"]:
        if "GPU-" + normalized in listed["parents"]:
            raise PreflightError(
                "A whole physical GPU was selected; this target requires a 1g.10gb MIG instance"
            )
        raise PreflightError("CUDA UUID does not match a MIG child in nvidia-smi -L")
    instance = listed["instances"][mig_uuid]
    if instance["profile"] != PROFILE:
        raise PreflightError(f"Expected MIG profile {PROFILE}; observed {instance['profile']}")
    properties = torch.cuda.get_device_properties(0)
    total = int(properties.total_memory)
    inventory.append(
        {
            "api": "torch.cuda.get_device_properties(0)",
            "observation": {"name": str(properties.name), "total_memory_bytes": total},
        }
    )
    # Loose consistency bounds allow driver-reserved memory. The primary
    # identification is the MIG UUID/profile, never advertised parent capacity.
    if not 8 * 1024**3 <= total <= 11 * 1024**3:
        raise PreflightError(f"CUDA instance memory is inconsistent with 1g.10gb: {total} bytes")
    parent = instance["parent"]
    query = _command(
        runner,
        [
            "nvidia-smi",
            "--id=" + parent["uuid"],
            "--query-gpu=uuid,name,driver_version,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        inventory,
    )
    rows = list(csv.reader(io.StringIO(query)))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise PreflightError("Expected one identifiable parent GPU metadata record")
    parent_uuid, parent_name, driver, mig_mode = (item.strip() for item in rows[0])
    if parent_uuid != parent["uuid"] or parent_name != parent["name"] or mig_mode.lower() != "enabled":
        raise PreflightError("MIG child parent identity/mode did not match the physical inventory")
    return {
        "profile": PROFILE,
        "mig_uuid": mig_uuid,
        "cuda_name": str(properties.name),
        "total_memory_bytes": total,
        "memory_source": "torch.cuda.get_device_properties(0).total_memory; MIG instance, not parent",
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "parent_gpu_uuid": parent_uuid,
        "parent_gpu_name": parent_name,
        "parent_gpu_index": parent["index"],
        "parent_mig_mode": mig_mode,
        "driver_version": driver,
        "cuda_driver_identity": identity,
        "accounting_note": "Slurm gpumem/gpuutil accounting is not required for MIG devices",
    }


def _cuda_smoke(torch) -> dict[str, Any]:
    """Execute finite forward, backward, and AdamW updates on the allocated MIG."""
    torch.manual_seed(97)
    torch.cuda.reset_peak_memory_stats(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 4)).to(
        device="cuda:0", dtype=torch.float32
    )
    data = torch.ones((16, 32), device="cuda:0", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    losses = []
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data)
        loss = prediction.square().mean()
        if prediction.device.type != "cuda" or not bool(torch.isfinite(prediction).all()):
            raise PreflightError("CUDA smoke forward failed device/finite checks")
        if not bool(torch.isfinite(loss)):
            raise PreflightError("CUDA smoke loss is not finite")
        loss.backward()
        for parameter in model.parameters():
            gradient = parameter.grad
            if gradient is None or gradient.device.type != "cuda" or not bool(torch.isfinite(gradient).all()):
                raise PreflightError("CUDA smoke backward failed device/finite checks")
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise PreflightError("CUDA smoke optimizer produced non-finite parameters")
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(0)
    if not any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)):
        raise PreflightError("CUDA smoke optimizer did not change any parameters")
    return {
        "optimizer_steps": 2,
        "losses": losses,
        "elapsed_s": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "scope": "Small FP32 CUDA MLP only; actual A512 training is a separate pilot stage",
    }


def run_preflight(
    out: Path,
    expected_commit: str,
    *,
    runtime_manifest: Path | None = None,
    env=None,
    runner=subprocess.run,
    module_loader=importlib.import_module,
    identity_probe=_cuda_identity,
    hostname=None,
) -> dict[str, Any]:
    env = os.environ if env is None else env
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ncc-single-1g.10gb-mig-preflight",
        "status": "running",
        "expected_commit": expected_commit,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname() if hostname is None else hostname,
        "runtime": {
            "python": platform.python_version(),
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
        },
        "preflight_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_inventory": [],
    }
    _atomic_write(out, record, exclusive=True)
    try:
        record["slurm"] = _allocation(env, runner, record["hostname"], record["raw_inventory"])
        record.update(_source(expected_commit, runner, record["raw_inventory"]))
        if record["runtime"]["python"] != VERSIONS["python"]:
            raise PreflightError("This NCC profile requires Python 3.12.3")
        if runtime_manifest is not None:
            record["runtime_manifest"] = _runtime_manifest(Path(runtime_manifest), expected_commit)
        numpy, torch = module_loader("numpy"), module_loader("torch")
        record["runtime"].update(
            numpy=str(numpy.__version__),
            torch=str(torch.__version__),
            torch_cuda=torch.version.cuda,
            cuda_build=bool(torch.backends.cuda.is_built()),
        )
        if any(record["runtime"].get(key) != value for key, value in VERSIONS.items()):
            raise PreflightError(f"Runtime version mismatch; required {VERSIONS}")
        if not torch.backends.cuda.is_built() or torch.version.cuda is None:
            raise PreflightError("A CUDA PyTorch wheel is required; loading a CUDA module is insufficient")
        if not torch.cuda.is_available():
            raise PreflightError("CUDA is unavailable inside this allocated job")
        record["hardware"] = _hardware(torch, runner, identity_probe, record["raw_inventory"])
        record["cuda_smoke"] = _cuda_smoke(torch)
        record.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
        _atomic_write(out, record)
        return record
    except Exception as error:
        record.update(
            status="failed",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _atomic_write(out, record)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New JSON evidence file; never overwrite")
    parser.add_argument("--expected-commit", required=True, help="Exact expected 40-character source commit")
    parser.add_argument("--runtime-manifest", type=Path, help="Ready manifest from the fixed NCC installer")
    args = parser.parse_args(argv)
    try:
        result = run_preflight(args.out, args.expected_commit, runtime_manifest=args.runtime_manifest)
    except Exception as error:
        print(json.dumps({"status": "failed", "evidence": str(args.out), "error": str(error)}))
        print(f"NCC MIG preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "evidence": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
