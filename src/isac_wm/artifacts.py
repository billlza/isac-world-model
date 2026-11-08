"""Bounded loading, strict JSON, integrity checks and atomic local artifacts.

Checksums detect accidental modification when compared to a trusted manifest;
they are not publisher authentication. Only load checkpoints you trust.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import torch

MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
# Continuation also stores Adam moments and the selected model. Keep inference
# artifacts on their original budget; only explicit resume APIs use this bound.
MAX_RESUME_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink artifact")
    fd, tmp = tempfile.mkstemp(prefix=".isac-wm-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def save_checkpoint(path: Path, value: dict) -> None:
    _save_checkpoint(path, value, MAX_CHECKPOINT_BYTES)


def save_resume_checkpoint(path: Path, value: dict) -> None:
    """Save bounded training continuation state, not an inference artifact."""
    _validate_resume_checkpoint(value)
    _save_checkpoint(path, value, MAX_RESUME_CHECKPOINT_BYTES)


def _save_checkpoint(path: Path, value: dict, max_bytes: int) -> None:
    output = io.BytesIO()
    torch.save(value, output)
    if output.tell() > max_bytes:
        raise ValueError(f"Checkpoint exceeds the {max_bytes}-byte artifact budget")
    atomic_write(path, output.getvalue())
    atomic_write(Path(str(path) + ".sha256"), (sha256_file(path) + "\n").encode())


def read_checkpoint(path: Path) -> dict:
    return _read_checkpoint(path, MAX_CHECKPOINT_BYTES)


def read_resume_checkpoint(path: Path) -> dict:
    """Load continuation under its finite budget and the usual integrity checks."""
    result = _read_checkpoint(path, MAX_RESUME_CHECKPOINT_BYTES)
    _validate_resume_checkpoint(result)
    return result


def _read_checkpoint(path: Path, max_bytes: int) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Checkpoint must be a regular non-symlink file")
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("Checkpoint file-size limit exceeded")
    # Deserialize the SAME bounded bytes whose hash was verified. A later path
    # replacement cannot change the object between verification and loading.
    with path.open("rb") as f:
        raw = f.read(max_bytes + 1)
    if not 0 < len(raw) <= max_bytes:
        raise ValueError("Checkpoint file-size limit exceeded")
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_symlink():
        raise ValueError("Checkpoint checksum must not be a symlink")
    if sidecar.exists():
        if not sidecar.is_file() or sidecar.stat().st_size > 128:
            raise ValueError("Invalid checkpoint checksum sidecar")
        expected = sidecar.read_text(encoding="ascii").strip()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Invalid SHA-256 checksum format")
        if expected != hashlib.sha256(raw).hexdigest():
            raise ValueError("Checkpoint SHA-256 mismatch")
    stream = io.BytesIO(raw)
    if not zipfile.is_zipfile(stream):
        raise ValueError("Only PyTorch ZIP checkpoints are supported")
    with zipfile.ZipFile(stream) as z:
        infos = z.infolist()
        if len(infos) > 100_000 or sum(i.file_size for i in infos) > max_bytes:
            raise ValueError("Expanded checkpoint exceeds resource budget")
        if len({i.filename for i in infos}) != len(infos) or any(i.flag_bits & 1 for i in infos):
            raise ValueError("Duplicate or encrypted checkpoint members")
    stream.seek(0)
    result = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(result, dict):
        raise ValueError("Checkpoint root must be a mapping")
    if result.get("schema_version", 1) not in (1, 2):
        raise ValueError("Unsupported checkpoint schema")
    return result


def _validate_resume_checkpoint(value: dict) -> None:
    """Identify existing continuation payloads without changing their schema."""
    required = {
        "state_dict",
        "optimizer",
        "numpy_rng_json",
        "torch_rng",
        "completed_epochs",
        "best_checkpoint",
        "history",
        "model_config",
        "scene_config",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("Resume checkpoint is missing training continuation fields")
    if (
        not all(isinstance(value[name], dict) for name in ("state_dict", "model_config", "scene_config"))
        or not isinstance(value["optimizer"], dict)
        or not isinstance(value["optimizer"].get("state"), dict)
        or not isinstance(value["optimizer"].get("param_groups"), list)
        or not isinstance(value["numpy_rng_json"], str)
        or not isinstance(value["torch_rng"], torch.Tensor)
        or type(value["completed_epochs"]) is not int
        or value["completed_epochs"] < 1
        or not isinstance(value["history"], list)
        or (value["best_checkpoint"] is not None and not isinstance(value["best_checkpoint"], dict))
    ):
        raise ValueError("Resume checkpoint contains invalid training continuation fields")


def model_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(cpu.dtype).encode())
        h.update(str(tuple(cpu.shape)).encode())
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()
