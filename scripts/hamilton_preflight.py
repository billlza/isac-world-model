"""Fail-closed, single whole-H200 CUDA smoke check inside a Hamilton Slurm job.

Imports only the standard library until the allocation has been checked. This
does not install packages, submit jobs, or validate a research result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# A deliberately loose floor, not an assertion about advertised GB/GiB. It
# rejects the site's 18 GB slice while leaving room for reserved device memory.
MIN_FULL_H200_BYTES = 120 * 1024**3


class PreflightError(RuntimeError):
    """A required condition has not been established; training must not start."""


def _write(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command(runner, args: list[str]) -> str:
    return runner(args, cwd=ROOT, capture_output=True, text=True, check=True, timeout=20).stdout.strip()


def _allocation(env, runner) -> dict[str, str]:
    job_id = env.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal():
        raise PreflightError("Run this check in an allocated Slurm job, not on a login node")
    if env.get("SLURM_JOB_PARTITION") != "cuda":
        raise PreflightError("The Hamilton single-H200 pilot requires the cuda partition")
    if not env.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise PreflightError("Slurm has not provided CUDA_VISIBLE_DEVICES; do not set GPU IDs manually")
    query_id = job_id
    if "SLURM_ARRAY_JOB_ID" in env or "SLURM_ARRAY_TASK_ID" in env:
        array, task = env.get("SLURM_ARRAY_JOB_ID", ""), env.get("SLURM_ARRAY_TASK_ID", "")
        if not array.isdecimal() or not task.isdecimal():
            raise PreflightError("Incomplete Slurm array element identity")
        query_id = f"{array}_{task}"
    output = _command(runner, ["scontrol", "show", "job", "--oneliner", query_id])
    if len(output.splitlines()) != 1:
        raise PreflightError("scontrol must identify exactly one allocation")

    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", output))
    if query_id != job_id and (fields.get("ArrayJobId"), fields.get("ArrayTaskId")) != (array, task):
        raise PreflightError("scontrol returned a different Slurm array element")
    if fields.get("JobState") != "RUNNING" or fields.get("Partition") != "cuda":
        raise PreflightError("scontrol did not confirm a RUNNING job in the cuda partition")
    allocated = fields.get("AllocTRES", "").split(",")
    requested = ",".join(fields.get(key, "") for key in ("TresPerNode", "TresPerJob")).split(",")
    if "gres/gpu:h200_nvl=1" not in allocated and "gres/gpu:h200_nvl:1" not in requested:
        raise PreflightError("Whole-H200 allocation not established; request --gres=gpu:h200_nvl:1")
    return {
        "job_id": job_id,
        "partition": fields["Partition"],
        "state": fields["JobState"],
        "allocated_tres": fields.get("AllocTRES", ""),
        "requested_tres_per_node": fields.get("TresPerNode", ""),
        "requested_tres_per_job": fields.get("TresPerJob", ""),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
    }


def _device_uuid(properties) -> str:
    value = getattr(properties, "uuid", None)
    if value is None:
        raise PreflightError("PyTorch did not expose a device UUID; cannot match the allocated GPU")
    if isinstance(value, bytes):
        return "GPU-" + str(uuid.UUID(bytes=value))
    value = str(value)
    if value.startswith("MIG-"):
        raise PreflightError("A MIG device was allocated; this pilot requires a whole H200")
    return "GPU-" + str(uuid.UUID(value.removeprefix("GPU-")))


def _device(torch, runner) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise PreflightError("Exactly one CUDA device must be visible to the pilot process")
    properties = torch.cuda.get_device_properties(0)
    name, total = str(properties.name), int(properties.total_memory)
    if "H200" not in name or "MIG" in name.upper() or total < MIN_FULL_H200_BYTES:
        raise PreflightError(f"Expected a whole H200; received {name!r} with {total} bytes")
    device_uuid = _device_uuid(properties)
    output = _command(
        runner,
        [
            "nvidia-smi",
            "--id=" + device_uuid,
            "--query-gpu=uuid,name,driver_version,memory.total,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
    )
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise PreflightError("nvidia-smi did not return one identifiable GPU record")
    observed_uuid, observed_name, driver, memory_mib, mig_mode = (value.strip() for value in rows[0])
    if observed_uuid != device_uuid or "H200" not in observed_name or mig_mode.lower() != "disabled":
        raise PreflightError("The CUDA device was not confirmed as a whole H200 with MIG disabled")
    return {
        "name": name,
        "uuid": device_uuid,
        "total_memory_bytes": total,
        "nvidia_smi_memory_mib": int(memory_mib),
        "driver_version": driver,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "mig_mode": mig_mode,
    }


def _cuda_smoke(torch) -> dict[str, Any]:
    """Execute actual forward/backward/optimizer CUDA work, not an import test."""
    torch.manual_seed(7)
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 4)).cuda()
    data = torch.ones((16, 32), device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data)
        loss = prediction.square().mean()
        if prediction.device.type != "cuda" or not bool(torch.isfinite(loss)):
            raise PreflightError("CUDA smoke forward failed device/finite checks")
        loss.backward()
        for parameter in model.parameters():
            gradient = parameter.grad
            if gradient is None or gradient.device.type != "cuda" or not bool(torch.isfinite(gradient).all()):
                raise PreflightError("CUDA smoke backward failed device/finite checks")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    if not any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)):
        raise PreflightError("CUDA smoke optimizer did not update any parameters")
    return {
        "optimizer_steps": 2,
        "losses": losses,
        "elapsed_s": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scope": "Small CUDA MLP only; project training and resume are separate pilot stages",
    }


def run_preflight(
    out: Path,
    *,
    env=None,
    runner=subprocess.run,
    torch_loader=importlib.import_module,
) -> dict[str, Any]:
    env = os.environ if env is None else env
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hamilton-single-h200-preflight",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "executable": sys.executable},
        "preflight_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Exclusive creation preserves earlier evidence, including a failed check.
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    try:
        record["slurm"] = _allocation(env, runner)
        record["git_commit"] = _command(runner, ["git", "rev-parse", "HEAD"])
        torch = torch_loader("torch")
        record["runtime"].update(torch=str(torch.__version__), torch_cuda=torch.version.cuda)
        if not torch.backends.cuda.is_built() or torch.version.cuda is None:
            raise PreflightError("CPU-only PyTorch build; loading a CUDA module cannot fix a CPU wheel")
        if not torch.cuda.is_available():
            raise PreflightError("PyTorch was built with CUDA but CUDA is unavailable in this job")
        record["gpu"] = _device(torch, runner)
        record["cuda_smoke"] = _cuda_smoke(torch)
        record["status"] = "passed"
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write(out, record)
        return record
    except Exception as error:
        record.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        _write(out, record)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New JSON evidence file; never overwritten")
    args = parser.parse_args(argv)
    try:
        result = run_preflight(args.out)
    except PreflightError as error:
        print(f"Hamilton preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "evidence": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
