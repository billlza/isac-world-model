"""Bounded loading, strict JSON, integrity checks and atomic local artifacts.

Checksums detect accidental modification when compared to a trusted manifest;
they are not publisher authentication. Only load checkpoints you trust.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import torch

MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
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
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False)+"\n").encode())


def save_checkpoint(path: Path, value: dict) -> None:
    import io
    output = io.BytesIO(); torch.save(value, output)
    if output.tell() > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds this prototype's 256 MiB artifact budget")
    atomic_write(path, output.getvalue())
    atomic_write(Path(str(path)+".sha256"), (sha256_file(path)+"\n").encode())


def read_checkpoint(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Checkpoint must be a regular non-symlink file")
    if not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint file-size limit exceeded")
    sidecar = Path(str(path)+".sha256")
    if sidecar.exists() and sidecar.read_text().strip() != sha256_file(path):
        raise ValueError("Checkpoint SHA-256 mismatch")
    if not zipfile.is_zipfile(path):
        raise ValueError("Only PyTorch ZIP checkpoints are supported")
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) > 100_000 or sum(i.file_size for i in z.infolist()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Expanded checkpoint exceeds resource budget")
    result = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(result, dict):
        raise ValueError("Checkpoint root must be a mapping")
    if result.get("schema_version", 1) not in (1, 2):
        raise ValueError("Unsupported checkpoint schema")
    return result


def model_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(cpu.dtype).encode())
        h.update(str(tuple(cpu.shape)).encode()); h.update(cpu.numpy().tobytes())
    return h.hexdigest()
