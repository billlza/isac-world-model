"""Data integrity, isolation, decoding and procedural-development contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import torch

from text_video.data import DataValidationError, VideoDataset, make_procedural_dataset, validate_manifest


def _media(tmp_path: Path, name: str, pixels: np.ndarray, *, compressed: bool = True) -> Path:
    destination = tmp_path / name
    with destination.open("wb") as stream:
        (np.savez_compressed if compressed else np.savez)(stream, frames=pixels)
    return destination


def _row(path: Path, *, sample_id: str = "one", group_id: str = "episode-one", split: str = "train", count: int = 4) -> dict:
    return {
        "schema_version": 1,
        "id": sample_id,
        "group_id": group_id,
        "split": split,
        "caption": "A red sphere moves right.",
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fps": 8.0,
        "reference_weight": [0.0, 0.25, 0.75, 1.0][:count],
        "provenance": {"source": "original-test-render", "license": "CC0-1.0"},
    }


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, dict, np.ndarray]:
    pixels = np.arange(4 * 10 * 12 * 3, dtype=np.int64).reshape(4, 10, 12, 3).astype(np.uint8)
    path = _media(tmp_path, "one.npz", pixels)
    return path, _row(path), pixels


def test_layout_normalization_prefix_and_external_weights(tmp_path: Path, sample: tuple) -> None:
    _, row, pixels = sample
    data = VideoDataset(_manifest(tmp_path, [row]), "train", frames=3, height=10, width=12)
    item = data[0]
    expected = torch.from_numpy(pixels[:3].copy()).permute(3, 0, 1, 2).float() / 127.5 - 1
    torch.testing.assert_close(item["video"], expected)
    torch.testing.assert_close(item["reference_weight"], torch.tensor([0.0, 0.25, 0.75]))
    assert item["video"].dtype == torch.float32
    assert item["video"].is_contiguous()
    assert item["sample_id"] == "one" and item["group_id"] == "episode-one"
    assert data.captions == (row["caption"],)
    assert len(data.manifest_sha256) == 64


def test_explicit_resize(tmp_path: Path, sample: tuple) -> None:
    _, row, _ = sample
    item = VideoDataset(_manifest(tmp_path, [row]), "train", 4, 8, 16)[0]
    assert item["video"].shape == (3, 4, 8, 16)
    assert torch.isfinite(item["video"]).all()
    assert -1 <= item["video"].min() <= item["video"].max() <= 1


@pytest.mark.parametrize("field,value", [
    ("schema_version", True),
    ("caption", "  "),
    ("fps", 0),
    ("fps", float("inf")),
    ("reference_weight", [1, 1, 1, float("nan")]),
    ("reference_weight", [1, 1, 1, -0.1]),
    ("reference_weight", [1, 1, 1, 1.1]),
    ("reference_weight", [1, 1, 1, True]),
    ("reference_weight", [1, 1]),
    ("provenance", {"source": "original"}),
    ("provenance", {"source": " ", "license": "CC0-1.0"}),
    ("sha256", "not-a-digest"),
    ("split", "val"),
    ("group_id", " episode-one"),
    ("path", "https://example.com/video.mp4"),
])
def test_invalid_metadata_is_rejected(tmp_path: Path, sample: tuple, field: str, value: object) -> None:
    _, row, _ = sample
    row[field] = value
    with pytest.raises(DataValidationError):
        VideoDataset(_manifest(tmp_path, [row]), "train", 4, 8, 8)


def test_missing_reference_weights_are_not_silently_trusted(tmp_path: Path, sample: tuple) -> None:
    _, row, _ = sample
    del row["reference_weight"]
    with pytest.raises(DataValidationError, match="reference_weight"):
        validate_manifest(_manifest(tmp_path, [row]))


@pytest.mark.parametrize("content", ["", "\n", '{"schema_version":1', '{"id":"a","id":"b"}\n'])
def test_empty_truncated_or_ambiguous_json_is_rejected(tmp_path: Path, content: str) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(content, encoding="utf-8")
    with pytest.raises(DataValidationError):
        validate_manifest(manifest)


def test_integrity_checked_at_initialization_and_on_each_access(tmp_path: Path, sample: tuple) -> None:
    path, row, _ = sample
    manifest = _manifest(tmp_path, [row])
    data = VideoDataset(manifest, "train", 4, 8, 8)
    path.write_bytes(path.read_bytes() + b"unexpected mutation")
    with pytest.raises(DataValidationError, match="sha256 mismatch"):
        data[0]
    with pytest.raises(DataValidationError, match="sha256 mismatch"):
        VideoDataset(manifest, "train", 4, 8, 8)


def test_corrupt_file_is_rejected_even_with_matching_digest(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not a zip archive")
    with pytest.raises(DataValidationError, match="decoding failed"):
        validate_manifest(_manifest(tmp_path, [_row(path)]))


@pytest.mark.parametrize("pixels", [
    np.zeros((4, 8, 8, 3), dtype=np.float32),
    np.zeros((4, 8, 8, 4), dtype=np.uint8),
    np.zeros((0, 8, 8, 3), dtype=np.uint8),
    np.zeros((8, 8, 3), dtype=np.uint8),
])
def test_invalid_array_contract_is_rejected(tmp_path: Path, pixels: np.ndarray) -> None:
    path = _media(tmp_path, "invalid.npz", pixels)
    with pytest.raises(DataValidationError, match="frames|uint8"):
        validate_manifest(_manifest(tmp_path, [_row(path)]))


def test_insufficient_frames_is_not_padded(tmp_path: Path, sample: tuple) -> None:
    _, row, _ = sample
    with pytest.raises(DataValidationError, match="insufficient frames"):
        VideoDataset(_manifest(tmp_path, [row]), "train", 5, 8, 8)


def test_episode_group_cannot_cross_splits(tmp_path: Path, sample: tuple) -> None:
    _, first, pixels = sample
    second_path = _media(tmp_path, "two.npz", 255 - pixels)
    second = _row(second_path, sample_id="two", split="test")
    with pytest.raises(DataValidationError, match="Group .* across splits"):
        validate_manifest(_manifest(tmp_path, [first, second]))


def test_identical_file_bytes_cannot_cross_splits(tmp_path: Path, sample: tuple) -> None:
    _, first, _ = sample
    second = dict(first, id="two", group_id="episode-two", split="validation")
    with pytest.raises(DataValidationError, match="duplicate video bytes"):
        validate_manifest(_manifest(tmp_path, [first, second]))


def test_identical_frames_repacked_in_new_file_cannot_cross_splits(tmp_path: Path, sample: tuple) -> None:
    _, first, pixels = sample
    second_path = _media(tmp_path, "repacked.npz", pixels, compressed=False)
    second = _row(second_path, sample_id="two", group_id="episode-two", split="test")
    assert first["sha256"] != second["sha256"]
    with pytest.raises(DataValidationError, match="duplicate decoded video content"):
        validate_manifest(_manifest(tmp_path, [first, second]))


def test_duplicate_ids_and_absent_split_are_rejected(tmp_path: Path, sample: tuple) -> None:
    _, row, _ = sample
    with pytest.raises(DataValidationError, match="Duplicate sample id"):
        validate_manifest(_manifest(tmp_path, [row, row]))
    with pytest.raises(DataValidationError, match="no samples"):
        VideoDataset(_manifest(tmp_path, [row]), "validation", 4, 8, 8)


def test_gif_and_mp4_decode_exact_payload(tmp_path: Path) -> None:
    pixels = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    for index in range(4):
        pixels[index, :, index * 3:index * 3 + 4] = (220, 40, 30)
    for suffix in (".gif", ".mp4"):
        path = tmp_path / f"clip{suffix}"
        options = {"duration": 125, "loop": 0} if suffix == ".gif" else {"fps": 8}
        iio.imwrite(path, pixels, **options)
        row = _row(path)
        item = VideoDataset(_manifest(tmp_path, [row]), "train", 4, 16, 16)[0]
        assert item["video"].shape == (3, 4, 16, 16)
        assert (item["video"][:, 0] - item["video"][:, -1]).abs().sum() > 0


def test_procedural_corpus_is_deterministic_varied_and_explicitly_synthetic(tmp_path: Path) -> None:
    first = make_procedural_dataset(tmp_path / "first", train=7, validation=2, test=2, frames=4, height=16, width=16)
    second = make_procedural_dataset(tmp_path / "second", train=7, validation=2, test=2, frames=4, height=16, width=16)
    records = validate_manifest(first)
    assert first.read_bytes() == second.read_bytes()
    assert len(records) == 11
    assert {row["split"] for row in records} == {"train", "validation", "test"}
    assert len({row["group_id"] for row in records}) == 11
    assert all(row["provenance"]["kind"] == "synthetic-development" for row in records)
    assert all(len(row["caption"].encode("utf-8")) <= 128 for row in records)
    data = VideoDataset(first, "train", 4, 16, 16)
    static = data[0]["video"]
    torch.testing.assert_close(static[:, 0], static[:, -1])
    moving = data[1]["video"]
    assert (moving[:, 0] - moving[:, -1]).abs().mean() > 0.02
    assert "while" in data[4]["caption"]
    assert "behind a gray pillar" in data[5]["caption"]
    assert "rises and falls" in data[6]["caption"]
    with pytest.raises(FileExistsError, match="overwrite"):
        make_procedural_dataset(first.parent)
