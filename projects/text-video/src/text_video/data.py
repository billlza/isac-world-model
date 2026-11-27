"""Local captioned-video data with explicit provenance and isolated episode splits.

The JSONL schema is deliberately small. Every row contains ``schema_version=1``,
``id``, ``group_id``, ``split`` (train/validation/test), ``caption``, ``path``,
``sha256``, ``fps``, ``reference_weight``, and ``provenance.source/license``.
Weights describe the supplied reference frames, not model confidence. They are
required for every source frame; there is no inferred/default quality score.

Files are checked before use, including across splits. Provenance declarations
are validated for completeness; this does not independently establish ownership
or the accuracy of a declaration. The procedural generator below is an original,
small development corpus, not a substitute for general-domain training video.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

SPLITS = ("train", "validation", "test")
PROCEDURAL_VERSION = "text-video-procedural-v1"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_COLORS = {
    "red": (221, 63, 58),
    "blue": (58, 107, 223),
    "green": (55, 174, 96),
    "yellow": (234, 191, 46),
    "purple": (158, 81, 206),
    "orange": (236, 128, 43),
}
_SHAPES = ("sphere", "cube", "cone")


class DataValidationError(ValueError):
    """A data or provenance contract was violated; the sample is not skipped."""


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str, *, identity: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise DataValidationError(f"{name} must be a nonempty text string")
    if identity and value != value.strip():
        raise DataValidationError(f"{name} must not have surrounding whitespace")
    return value


def _reject_constant(value: str) -> None:
    raise DataValidationError(f"Nonfinite JSON constant {value!r} is not permitted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _check_finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DataValidationError("Nonfinite metadata is not permitted")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite_tree(item)


def _decode_video(payload: bytes, suffix: str, label: str) -> np.ndarray:
    try:
        if suffix == ".npz":
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if "frames" not in archive.files:
                    raise DataValidationError(f"{label}: NPZ must contain a 'frames' array")
                frames = archive["frames"]
        else:
            import imageio.v3 as iio

            # Bytes bind decoding to the exact payload whose digest was checked.
            frames = iio.imread(payload, extension=suffix)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(f"{label}: video decoding failed: {exc}") from exc
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise DataValidationError(f"{label}: frames must be uint8 RGB, with no implicit dtype conversion")
    if frames.ndim != 4 or frames.shape[-1] != 3 or any(size == 0 for size in frames.shape):
        raise DataValidationError(f"{label}: expected nonempty frames [T,H,W,3], got {frames.shape}")
    return np.ascontiguousarray(frames)


def _read_verified(row: dict[str, Any], manifest_dir: Path) -> np.ndarray:
    source = Path(row["path"])
    path = source if source.is_absolute() else manifest_dir / source
    if path.suffix.lower() not in (".npz", ".mp4", ".gif"):
        raise DataValidationError(f"{row['id']}: supported files are .npz, .mp4, and .gif")
    try:
        if not path.is_file():
            raise DataValidationError(f"{row['id']}: source is not a local regular file: {path}")
        payload = path.read_bytes()
    except OSError as exc:
        raise DataValidationError(f"{row['id']}: cannot read local video: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise DataValidationError(f"{row['id']}: sha256 mismatch; source bytes changed or metadata is wrong")
    frames = _decode_video(payload, path.suffix.lower(), row["id"])
    if len(row["reference_weight"]) != len(frames):
        raise DataValidationError(
            f"{row['id']}: reference_weight length {len(row['reference_weight'])} "
            f"does not match decoded frame count {len(frames)}"
        )
    return frames


def _validate_row(row: Any, line: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DataValidationError(f"Line {line}: each JSONL row must be an object")
    required = {
        "schema_version", "id", "group_id", "split", "caption", "path", "sha256",
        "fps", "reference_weight", "provenance",
    }
    missing = required - row.keys()
    if missing:
        raise DataValidationError(f"Line {line}: missing required fields: {', '.join(sorted(missing))}")
    _check_finite_tree(row)
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise DataValidationError(f"Line {line}: schema_version must be integer 1")
    for key in ("id", "group_id", "split", "path"):
        _text(row[key], f"Line {line} {key}", identity=True)
    _text(row["caption"], f"Line {line} caption")
    if row["split"] not in SPLITS:
        raise DataValidationError(f"Line {line}: split must be one of {SPLITS}")
    if "://" in row["path"]:
        raise DataValidationError(f"Line {line}: path must be local; downloading is not supported")
    if not isinstance(row["sha256"], str) or not _SHA256.fullmatch(row["sha256"]):
        raise DataValidationError(f"Line {line}: sha256 must be a 64-character hexadecimal digest")
    row["sha256"] = row["sha256"].lower()
    fps = row["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise DataValidationError(f"Line {line}: fps must be finite and positive")
    provenance = row["provenance"]
    if not isinstance(provenance, dict):
        raise DataValidationError(f"Line {line}: provenance must be an object")
    for key in ("source", "license"):
        _text(provenance.get(key), f"Line {line} provenance.{key}")
    weights = row["reference_weight"]
    if not isinstance(weights, list) or not weights:
        raise DataValidationError(f"Line {line}: reference_weight must be a nonempty per-frame array")
    for weight in weights:
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 1
        ):
            raise DataValidationError(f"Line {line}: reference_weight entries must be finite numbers in [0,1]")
    return row


def validate_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Validate all rows, media and split isolation; return the declared records.

    Both identical file bytes and identical decoded RGB sequences are rejected
    across splits. Distinct clips of one episode must share ``group_id`` so the
    group-level check can also protect against adjacent-clip leakage.
    """
    path = Path(manifest_path).resolve()
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DataValidationError(f"Cannot read UTF-8 manifest: {exc}") from exc
    if not content.strip():
        raise DataValidationError("Manifest is empty")
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    groups: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    pixel_hashes: dict[str, str] = {}
    for line_number, text in enumerate(content.splitlines(), start=1):
        if not text.strip():
            raise DataValidationError(f"Line {line_number}: blank JSONL records are not permitted")
        try:
            row = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"Line {line_number}: invalid JSON record: {exc}") from exc
        row = _validate_row(row, line_number)
        sample_id, group_id, split = row["id"], row["group_id"], row["split"]
        if sample_id in identifiers:
            raise DataValidationError(f"Duplicate sample id {sample_id!r}")
        identifiers.add(sample_id)
        if group_id in groups and groups[group_id] != split:
            raise DataValidationError(f"Group {group_id!r} appears across splits")
        groups[group_id] = split
        digest = row["sha256"]
        if digest in file_hashes and file_hashes[digest] != split:
            raise DataValidationError(f"{sample_id}: duplicate video bytes across splits")
        file_hashes[digest] = split
        frames = _read_verified(row, path.parent)
        pixel_digest = hashlib.sha256(str(frames.shape).encode("ascii") + frames.tobytes()).hexdigest()
        row["_decoded_sha256"] = pixel_digest
        if pixel_digest in pixel_hashes and pixel_hashes[pixel_digest] != split:
            raise DataValidationError(f"{sample_id}: duplicate decoded video content across splits")
        pixel_hashes[pixel_digest] = split
        records.append(row)
    return records


class VideoDataset(Dataset):
    """Use a contiguous prefix of T frames and an explicitly requested resolution.

    No temporal interpolation, random crop, augmentation, sample substitution, or
    missing-quality default is performed. RGB resizing is bilinear. All records
    are validated at initialization; each access rechecks the selected file.
    """

    def __init__(
        self, manifest_path: str | Path, split: str, frames: int, height: int, width: int
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.frames = _positive_int(frames, "frames")
        self.height = _positive_int(height, "height")
        self.width = _positive_int(width, "width")
        if split not in SPLITS:
            raise DataValidationError(f"split must be one of {SPLITS}")
        self.split = split
        all_records = validate_manifest(self.manifest_path)
        self.records = [row for row in all_records if row["split"] == split]
        if not self.records:
            raise DataValidationError(f"Manifest has no samples for split {split!r}")
        for row in self.records:
            if len(row["reference_weight"]) < frames:
                raise DataValidationError(f"{row['id']}: insufficient frames for requested length {frames}")
        self.captions = tuple(row["caption"] for row in self.records)
        self.sample_ids = tuple(row["id"] for row in self.records)
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        frames = _read_verified(row, self.manifest_path.parent)[: self.frames]
        video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).to(torch.float32)
        video = video / 127.5 - 1.0
        if video.shape[-2:] != (self.height, self.width):
            video = F.interpolate(video, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return {
            "video": video.permute(1, 0, 2, 3).contiguous(),
            "caption": row["caption"],
            "sample_id": row["id"],
            "group_id": row["group_id"],
            "reference_weight": torch.tensor(row["reference_weight"][: self.frames], dtype=torch.float32),
        }


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


def _draw_object(image: Image.Image, shape: str, color: tuple[int, int, int], x: float, y: float, r: float) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.ellipse((x - 1.1 * r, y - 0.2 * r, x + 1.1 * r, y + 0.25 * r), fill=(15, 18, 25, 65))
    if shape == "sphere":
        for layer in range(18):
            fraction = 1 - layer / 19
            radius = r * fraction
            cx, cy = x - (1 - fraction) * 0.32 * r, y - r - (1 - fraction) * 0.3 * r
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=_shade(color, 0.5 + 0.6 * (1 - fraction)))
        draw.ellipse((x - 0.5 * r, y - 1.65 * r, x - 0.12 * r, y - 1.3 * r), fill=(255, 255, 255, 120))
    elif shape == "cube":
        a, b = (x - r, y - 1.5 * r), (x + 0.35 * r, y - 1.3 * r)
        c, d = (x + r, y - 1.8 * r), (x - 0.35 * r, y - 2 * r)
        aa, bb, cc = (x - r, y - 0.2 * r), (x + 0.35 * r, y), (x + r, y - 0.5 * r)
        draw.polygon([a, b, bb, aa], fill=_shade(color, 0.95))
        draw.polygon([b, c, cc, bb], fill=_shade(color, 0.60))
        draw.polygon([a, b, c, d], fill=_shade(color, 1.20))
    else:
        top = (x - 0.1 * r, y - 2 * r)
        draw.polygon([top, (x - r, y - 0.18 * r), (x, y + 0.1 * r)], fill=_shade(color, 1.05))
        draw.polygon([top, (x, y + 0.1 * r), (x + r, y - 0.18 * r)], fill=_shade(color, 0.70))
        draw.ellipse((x - r, y - 0.4 * r, x + r, y + 0.1 * r), fill=_shade(color, 0.68))


def _render_clip(rng: np.random.Generator, count: int, height: int, width: int, scenario: int) -> tuple[np.ndarray, str]:
    # Supersampling makes the small geometric development scenes visually legible.
    scale = 3
    h, w = height * scale, width * scale
    horizon = int(h * 0.45)
    wall = rng.integers(115, 190, size=3)
    floor = rng.integers(80, 160, size=3)
    background = np.empty((h, w, 3), dtype=np.float32)
    for y in range(h):
        base = wall if y < horizon else floor
        factor = 1.06 - 0.16 * y / max(1, h - 1)
        background[y] = base * factor
    background += rng.integers(-3, 4, size=background.shape)
    base_image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8))
    grid = ImageDraw.Draw(base_image, "RGBA")
    for endpoint in np.linspace(-w, 2 * w, 9):
        grid.line([(w * 0.5, horizon), (float(endpoint), h)], fill=(30, 40, 50, 45), width=scale)
    for k in range(1, 7):
        y = horizon + (h - horizon) * (k / 7) ** 1.7
        grid.line([(0, y), (w, y)], fill=(30, 40, 50, 45), width=scale)
    color_names = rng.choice(tuple(_COLORS), size=2, replace=False).tolist()
    shape, other_shape = rng.choice(_SHAPES, size=2).tolist()
    subject = f"A {color_names[0]} {shape}"
    captions = (
        f"{subject} stays still on a tiled floor.",
        f"{subject} moves from left to right on a tiled floor.",
        f"{subject} moves from right to left on a tiled floor.",
        f"{subject} moves toward the camera on a tiled floor.",
        f"{subject} moves from left to right while a {color_names[1]} {other_shape} stays still.",
        f"{subject} moves from left to right behind a gray pillar.",
        f"{subject} rises and falls above a tiled floor.",
    )
    result = np.empty((count, height, width, 3), dtype=np.uint8)
    for frame in range(count):
        t = frame / max(1, count - 1)
        image = base_image.copy()
        radius = min(w, h) * 0.12
        x, y = w * 0.5, h * 0.82
        if scenario in (1, 4, 5):
            x = w * (0.22 + 0.56 * t)
        elif scenario == 2:
            x = w * (0.78 - 0.56 * t)
        elif scenario == 3:
            y = h * (0.65 + 0.22 * t)
            radius *= 0.65 + 0.65 * t
        elif scenario == 6:
            y -= h * 0.23 * math.sin(math.pi * t)
        if scenario == 4:
            _draw_object(image, other_shape, _COLORS[color_names[1]], w * 0.7, h * 0.64, radius * 0.7)
        _draw_object(image, shape, _COLORS[color_names[0]], x, y, radius)
        if scenario == 5:
            draw = ImageDraw.Draw(image)
            draw.rectangle((w * 0.44, h * 0.39, w * 0.54, h * 0.87), fill=(105, 110, 120))
            draw.polygon([(w * 0.54, h * 0.39), (w * 0.59, h * 0.35), (w * 0.59, h * 0.82), (w * 0.54, h * 0.87)], fill=(75, 82, 93))
            draw.polygon([(w * 0.44, h * 0.39), (w * 0.49, h * 0.35), (w * 0.59, h * 0.35), (w * 0.54, h * 0.39)], fill=(145, 150, 159))
        result[frame] = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))
    return result, captions[scenario]


def make_procedural_dataset(
    out_dir: str | Path, *, train: int = 256, validation: int = 32, test: int = 32,
    frames: int = 8, height: int = 32, width: int = 32, seed: int = 7,
) -> Path:
    """Write original captioned geometric videos for development and return JSONL.

    Episodes include static, directional, depth-changing, two-object, occluded,
    and rising/falling motion. They have deterministic varied backgrounds and
    shaded geometry. All three splits are independent episodes. The corpus is
    explicitly synthetic; its captions and unit reference weights are supplied
    by this renderer, without a pretrained model or scraped/generated content.
    Existing destination files are never overwritten.
    """
    for name, count in (("train", train), ("validation", validation), ("test", test)):
        _positive_int(count, name, minimum=0)
    if train + validation + test == 0:
        raise DataValidationError("At least one procedural sample is required")
    _positive_int(frames, "frames")
    _positive_int(height, "height", minimum=8)
    _positive_int(width, "width", minimum=8)
    _positive_int(seed, "seed", minimum=0)
    out = Path(out_dir).resolve()
    manifest = out / "manifest.jsonl"
    if manifest.exists() or (out / "videos").exists():
        raise FileExistsError(f"Refusing to overwrite an existing procedural corpus in {out}")
    (out / "videos").mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for split_index, (split, count) in enumerate(zip(SPLITS, (train, validation, test), strict=True)):
        for index in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, index]))
            scenario = index % 7
            clip, caption = _render_clip(rng, frames, height, width, scenario)
            sample_id = f"procedural-{seed}-{split}-{index:06d}"
            relative = Path("videos") / f"{sample_id}.npz"
            destination = out / relative
            with destination.open("xb") as stream:
                np.savez_compressed(stream, frames=clip)
            records.append({
                "schema_version": 1,
                "id": sample_id,
                "group_id": sample_id,
                "split": split,
                "caption": caption,
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "fps": 8.0,
                "reference_weight": [1.0] * frames,
                "provenance": {
                    "source": PROCEDURAL_VERSION,
                    "license": "CC0-1.0",
                    "kind": "synthetic-development",
                    "generator_version": PROCEDURAL_VERSION,
                    "seed": seed,
                    "scenario": scenario,
                },
            })
    with manifest.open("x", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return manifest
