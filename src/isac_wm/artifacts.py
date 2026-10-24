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
    import io

    output = io.BytesIO()
    torch.save(value, output)
    if output.tell() > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds this prototype's 256 MiB artifact budget")
    atomic_write(path, output.getvalue())
    atomic_write(Path(str(path) + ".sha256"), (sha256_file(path) + "\n").encode())


def read_checkpoint(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Checkpoint must be a regular non-symlink file")
    if not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint file-size limit exceeded")
    # Deserialize the SAME bounded bytes whose hash was verified. A later path
    # replacement cannot change the object between verification and loading.
    with path.open("rb") as f:
        raw = f.read(MAX_CHECKPOINT_BYTES + 1)
    if not 0 < len(raw) <= MAX_CHECKPOINT_BYTES:
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
        if len(infos) > 100_000 or sum(i.file_size for i in infos) > MAX_CHECKPOINT_BYTES:
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


def model_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(cpu.dtype).encode())
        h.update(str(tuple(cpu.shape)).encode())
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()
