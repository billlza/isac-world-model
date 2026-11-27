"""Atomic, hashed artifacts with explicit format and provenance checks."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

import numpy as np
import torch


ARCHITECTURE = "text_video_flow_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity():
    package = Path(__file__).resolve().parent
    sources = {p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}
    result = {"source_fingerprint": canonical_hash(sources), "source_files": sources}
    probe = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True, check=False)
    if probe.returncode == 0:
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=package, capture_output=True, text=True, check=True)
        result.update(git_commit=probe.stdout.strip(), git_dirty=bool(dirty.stdout.strip()))
    else:
        result.update(git_commit=None, git_dirty=None)
    return result


def runtime_identity(device):
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": str(np.__version__), "platform": platform.platform(),
        "device": str(device), "cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "executable": sys.executable,
    }


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint_paths(path):
    # Resolve the parent, never the public symlink itself.
    path = Path(path)
    path = path.parent.resolve() / path.name
    return path, path.with_name(path.name + ".sha256"), path.with_name("." + path.name + ".versions")


@contextmanager
def _checkpoint_lock(store, *, exclusive):
    # Readers keep their generation alive while a writer would otherwise prune
    # it. flock also serializes independent writers, not only Python threads.
    mode = "a+b" if exclusive else "rb"
    with (store / "lock").open(mode) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _current_generation(store):
    current = store / "current"
    if not current.is_symlink():
        if current.exists():
            raise ValueError("Checkpoint current pointer must be a symlink")
        return None
    generation = current.resolve(strict=True)
    if generation.parent != store or not generation.name.startswith("generation-") or not generation.is_dir():
        raise ValueError("Checkpoint current pointer has an invalid target")
    return generation


def _write_generation(store, payload, previous, *, source=None):
    generation = store / ("generation-" + uuid.uuid4().hex)
    generation.mkdir()
    try:
        weights = generation / "weights.pt"
        with weights.open("wb") as stream:
            if source is None:
                torch.save(payload, stream)
            else:
                with source.open("rb") as original:
                    shutil.copyfileobj(original, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(weights)
        with (generation / "weights.pt.sha256").open("w", encoding="ascii") as stream:
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_json(generation / "generation.json", {"previous": previous.name if previous else None})
        _fsync_directory(generation)
        _fsync_directory(store)
        return generation, digest
    except BaseException:
        # This directory has not been published by _commit_generation yet.
        shutil.rmtree(generation)
        raise


def _commit_generation(store, generation):
    temporary = store / (".current-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(generation.name, target_is_directory=True)
        # This is the single commit point for BOTH the weights and their hash.
        os.replace(temporary, store / "current")
        _fsync_directory(store)
    finally:
        temporary.unlink(missing_ok=True)


def _export_targets(path, sidecar, store):
    return (
        (path, f"{store.name}/current/weights.pt"),
        (sidecar, f"{store.name}/current/weights.pt.sha256"),
    )


def _install_exports(path, sidecar, store):
    for export, target in _export_targets(path, sidecar, store):
        if export.is_symlink() and os.readlink(export) == target:
            continue
        temporary = export.with_name("." + export.name + ".export-" + uuid.uuid4().hex)
        try:
            temporary.symlink_to(target)
            os.replace(temporary, export)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _verify_pair(path, sidecar):
    if not sidecar.is_file():
        raise ValueError(f"Checkpoint hash sidecar missing: {sidecar}")
    actual = sha256_file(path)
    if sidecar.read_text(encoding="ascii").strip() != actual:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    return actual


def _validate_exports(path, sidecar, store, generation):
    if not sidecar.is_file():
        raise ValueError(f"Checkpoint hash sidecar missing: {sidecar}")
    digest = (generation / "weights.pt.sha256").read_text(encoding="ascii").strip()
    for export, expected in _export_targets(path, sidecar, store):
        if export.is_symlink():
            if os.readlink(export) != expected:
                raise ValueError(f"Checkpoint export target mismatch: {export}")
        elif export == path:
            # A crash during migration can leave one flat file and one symlink.
            # Both still describe the same old generation until the next commit.
            if sha256_file(export) != digest:
                raise ValueError(f"Checkpoint SHA256 mismatch: {export}")
        elif export.read_text(encoding="ascii").strip() != digest:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}")


def _prune_generations(store):
    current = _current_generation(store)
    keep = set()
    if current is not None:
        keep.add(current.name)
        previous = json.loads((current / "generation.json").read_text(encoding="utf-8"))["previous"]
        if previous is not None:
            if not isinstance(previous, str) or Path(previous).name != previous or not previous.startswith("generation-"):
                raise ValueError("Invalid checkpoint generation history")
            keep.add(previous)
    for generation in store.iterdir():
        if generation.name.startswith("generation-") and generation.name not in keep:
            if generation.is_symlink() or not generation.is_dir():
                raise ValueError("Invalid entry in checkpoint generation store")
            shutil.rmtree(generation)
    _fsync_directory(store)


def save_checkpoint(path, payload):
    """Publish a real .pt file plus hash with one atomic generation-pointer swap.

    Public .pt/.sha256 exports remain ordinary readable file paths. Copying both
    with symlinks dereferenced creates a portable flat pair accepted by the loader.
    A flat pair can also be resumed and migrated without invalidating its bytes.
    """
    path, sidecar, store = _checkpoint_paths(path)
    store.mkdir(exist_ok=True)
    with _checkpoint_lock(store, exclusive=True):
        current = _current_generation(store)
        if current is None and (path.exists() or sidecar.exists()):
            # Preserve an imported flat pair before replacing either public path.
            old_digest = _verify_pair(path, sidecar)
            legacy, copied_digest = _write_generation(store, None, None, source=path)
            if copied_digest != old_digest:
                raise ValueError("Checkpoint changed while importing its flat files")
            _commit_generation(store, legacy)
            current = legacy
        if current is not None:
            _validate_exports(path, sidecar, store, current)
        _prune_generations(store)
        _install_exports(path, sidecar, store)
        candidate, digest = _write_generation(store, payload, current)
        try:
            _commit_generation(store, candidate)
        except BaseException:
            # An exception can occur AFTER os.replace has committed. Determine
            # the actual pointer before removing an unpublished candidate.
            if not (store / "current").is_symlink() or os.readlink(store / "current") != candidate.name:
                shutil.rmtree(candidate)
            raise
        _prune_generations(store)
        return digest


def _read_checkpoint_pair(path, sidecar):
    if not sidecar.is_file():
        raise ValueError(f"Checkpoint hash sidecar missing: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").strip()
    # Hash and deserialize the same open file, even for a portable flat pair.
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("architecture") != ARCHITECTURE or payload.get("schema_version") != 1:
        raise ValueError("Checkpoint belongs to a different architecture or unsupported schema")
    return payload


def load_checkpoint(path):
    path, sidecar, store = _checkpoint_paths(path)
    if not (store / "lock").exists():
        return _read_checkpoint_pair(path, sidecar)
    with _checkpoint_lock(store, exclusive=False):
        generation = _current_generation(store)
        if generation is None:
            return _read_checkpoint_pair(path, sidecar)
        _validate_exports(path, sidecar, store, generation)
        return _read_checkpoint_pair(generation / "weights.pt", generation / "weights.pt.sha256")
