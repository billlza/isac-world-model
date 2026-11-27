"""Interruptions cannot publish mismatched checkpoint bytes and hash sidecars."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import threading

import pytest
import torch

from text_video import io


def payload(step):
    return {
        "schema_version": 1, "architecture": io.ARCHITECTURE,
        "kind": "weights", "step": step, "value": torch.tensor([step]),
    }


def flat_pair(path, step):
    torch.save(payload(step), path)
    path.with_name(path.name + ".sha256").write_text(io.sha256_file(path) + "\n", encoding="ascii")


def generation_directories(path):
    store = io._checkpoint_paths(path)[2]
    return [entry for entry in store.iterdir() if entry.name.startswith("generation-")]


def test_real_torch_files_portability_and_bounded_generations(tmp_path):
    path = tmp_path / "resume.pt"
    for step in range(6):
        digest = io.save_checkpoint(path, payload(step))
        assert io.sha256_file(path) == digest
        assert io.load_checkpoint(path)["step"] == step
        assert torch.load(path, weights_only=True)["step"] == step
        assert len(generation_directories(path)) == min(2, step + 1)
    assert path.is_symlink()
    assert path.with_name(path.name + ".sha256").is_symlink()

    copied = tmp_path / "portable.pt"
    shutil.copyfile(path, copied)
    shutil.copyfile(path.with_name(path.name + ".sha256"), copied.with_name(copied.name + ".sha256"))
    assert not copied.is_symlink()
    assert io.load_checkpoint(copied)["step"] == 5
    io.save_checkpoint(copied, payload(6))
    assert io.load_checkpoint(copied)["step"] == 6
    assert len(generation_directories(copied)) == 2


@pytest.mark.parametrize("damage", ["missing_hash", "bad_hash", "bad_weights"])
def test_missing_or_corrupt_public_pair_is_rejected(tmp_path, damage):
    path = tmp_path / "weights.pt"
    io.save_checkpoint(path, payload(1))
    sidecar = path.with_name(path.name + ".sha256")
    if damage == "missing_hash":
        sidecar.unlink()
    elif damage == "bad_hash":
        sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="sidecar missing|SHA256 mismatch"):
        io.load_checkpoint(path)


@pytest.mark.parametrize("point", [
    "serialize", "weights_sync", "hash_sync", "metadata", "generation_sync",
    "before_commit", "after_commit", "commit_sync", "prune",
])
def test_interrupted_update_preserves_a_committed_generation(tmp_path, monkeypatch, point):
    path = tmp_path / "resume.pt"
    io.save_checkpoint(path, payload(0))
    io.save_checkpoint(path, payload(1))
    store = io._checkpoint_paths(path)[2]
    previous = io._current_generation(store)
    old_generations = set(generation_directories(path))
    older = next(generation for generation in old_generations if generation != previous)
    fired = []

    def fail():
        fired.append(point)
        raise OSError(f"injected {point}")

    with monkeypatch.context() as changes:
        if point == "serialize":
            def broken_save(value, stream):
                stream.write(b"partial checkpoint")
                fail()
            changes.setattr(torch, "save", broken_save)
        elif point in ("weights_sync", "hash_sync"):
            original = os.fsync
            target_name = "weights.pt" if point == "weights_sync" else "weights.pt.sha256"

            def broken_sync(descriptor):
                # File identity is obtained from the descriptor under test, not
                # from timing assumptions about how often directory sync runs.
                target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
                if target.name == target_name and target.parent not in old_generations:
                    fail()
                return original(descriptor)

            changes.setattr(os, "fsync", broken_sync)
        elif point == "metadata":
            original = io.write_json

            def broken_metadata(target, value):
                if Path(target).name == "generation.json":
                    fail()
                return original(target, value)

            changes.setattr(io, "write_json", broken_metadata)
        elif point in ("generation_sync", "commit_sync"):
            original = io._fsync_directory

            def broken_directory_sync(target):
                target = Path(target)
                if point == "generation_sync" and target.name.startswith("generation-") and target not in old_generations:
                    fail()
                if point == "commit_sync" and target == store and io._current_generation(store) != previous:
                    fail()
                return original(target)

            changes.setattr(io, "_fsync_directory", broken_directory_sync)
        elif point in ("before_commit", "after_commit"):
            original = os.replace

            def broken_replace(source, target):
                if Path(target) == store / "current" and point == "before_commit":
                    fail()
                result = original(source, target)
                if Path(target) == store / "current":
                    fail()
                return result

            changes.setattr(os, "replace", broken_replace)
        else:
            original = shutil.rmtree

            def broken_prune(target, *args, **kwargs):
                if Path(target) == older:
                    fail()
                return original(target, *args, **kwargs)

            changes.setattr(shutil, "rmtree", broken_prune)
        with pytest.raises(OSError, match="injected"):
            io.save_checkpoint(path, payload(2))
    assert fired, "The selected interruption point must actually execute"
    expected = 2 if point in ("after_commit", "commit_sync", "prune") else 1
    assert io.load_checkpoint(path)["step"] == expected
    # Even when the new pointer committed just before the interruption, the
    # immediately preceding checkpoint remains independently recoverable.
    assert io.load_checkpoint(previous / "weights.pt")["step"] == 1
    io.save_checkpoint(path, payload(3))
    assert io.load_checkpoint(path)["step"] == 3
    assert len(generation_directories(path)) == 2


@pytest.mark.parametrize("point", ["first_export", "second_export", "before_new_commit", "after_new_commit"])
def test_flat_pair_migration_is_interruption_safe(tmp_path, monkeypatch, point):
    path = tmp_path / "portable.pt"
    sidecar = path.with_name(path.name + ".sha256")
    flat_pair(path, 1)
    store = io._checkpoint_paths(path)[2]
    original = os.replace
    commits = 0

    def interrupted_replace(source, target):
        nonlocal commits
        target = Path(target)
        if target == store / "current":
            commits += 1
            if commits == 2 and point == "before_new_commit":
                raise OSError("injected migration interruption")
        result = original(source, target)
        if (
            (point == "first_export" and target == path)
            or (point == "second_export" and target == sidecar)
            or (point == "after_new_commit" and target == store / "current" and commits == 2)
        ):
            raise OSError("injected migration interruption")
        return result

    with monkeypatch.context() as changes:
        changes.setattr(os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="injected migration"):
            io.save_checkpoint(path, payload(2))
    assert io.load_checkpoint(path)["step"] == (2 if point == "after_new_commit" else 1)
    io.save_checkpoint(path, payload(3))
    assert io.load_checkpoint(path)["step"] == 3
    assert len(generation_directories(path)) == 2


def test_initial_write_can_be_retried_after_interruption(tmp_path, monkeypatch):
    path = tmp_path / "first.pt"
    original = torch.save

    def interrupted(value, stream):
        stream.write(b"partial")
        raise OSError("injected initial interruption")

    with monkeypatch.context() as changes:
        changes.setattr(torch, "save", interrupted)
        with pytest.raises(OSError, match="injected initial"):
            io.save_checkpoint(path, payload(1))
    assert torch.save is original
    io.save_checkpoint(path, payload(2))
    assert io.load_checkpoint(path)["step"] == 2
    assert len(generation_directories(path)) == 1


def test_reader_pins_generation_until_deserialization_finishes(tmp_path, monkeypatch):
    path = tmp_path / "resume.pt"
    io.save_checkpoint(path, payload(1))
    started = threading.Event()
    release = threading.Event()
    writer_started = threading.Event()
    writer_finished = threading.Event()
    original = io._read_checkpoint_pair

    def paused_reader(weights, sidecar):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Reader test was not released")
        return original(weights, sidecar)

    def writer():
        writer_started.set()
        io.save_checkpoint(path, payload(2))
        writer_finished.set()

    monkeypatch.setattr(io, "_read_checkpoint_pair", paused_reader)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(io.load_checkpoint, path)
        try:
            assert started.wait(timeout=3)
            writing = pool.submit(writer)
            assert writer_started.wait(timeout=3)
            assert not writer_finished.wait(timeout=0.05)
        finally:
            release.set()
        assert reading.result(timeout=3)["step"] == 1
        writing.result(timeout=3)
    assert io.load_checkpoint(path)["step"] == 2
