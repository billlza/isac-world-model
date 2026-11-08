"""Package the complete formal capacity/data study without transferring model weights.

The bundle reports scientific-stage completion. The scheduler's terminal state
remains in sacct and the separately retrieved status.json; a preflight RUNNING
snapshot must not be mistaken for the state of the finished job.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

WIDTHS = (96, 512)
DATA_COUNTS = (96, 1536)
INITIALIZATIONS = (7, 11)
UPDATES = list(range(96, 1537, 96))
TRAIN_BASE, VALIDATION_BASE, TEST_BASE, ORDER_SEED = 700_000, 74_611_000, 95_871_000, 37
MAX_JSON_BYTES = 64 * 1024**2
MAX_BUNDLE_INPUT_BYTES = 256 * 1024**2
MAX_CHECKPOINT_BYTES = 256 * 1024**2


def cell_name(width, episodes, seed):
    return f"a{width}-data-{episodes}-init-{seed}"


CELLS = {
    cell_name(width, episodes, seed): (width, episodes, seed)
    for width in WIDTHS
    for episodes in DATA_COUNTS
    for seed in INITIALIZATIONS
}
METHODS = set(CELLS) | {"kalman", "gated-imm"}
PAIRS = {
    (seed, condition) for seed in range(TEST_BASE, TEST_BASE + 24) for condition in ("control", "maneuver")
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite JSON value in study evidence")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _path(root, relative, budget):
    candidate = root / relative
    require(
        candidate.is_file() and not candidate.is_symlink(), f"Missing or non-regular evidence: {relative}"
    )
    require(candidate.resolve().is_relative_to(root), f"Evidence escapes run directory: {relative}")
    require(0 < candidate.stat().st_size <= budget, f"Evidence exceeds size budget or is empty: {relative}")
    return candidate


def _json(root, relative, files):
    path = _path(root, relative, MAX_JSON_BYTES)
    raw = path.read_bytes()
    require(len(raw) <= MAX_JSON_BYTES, f"Evidence grew beyond size budget: {relative}")
    require(
        sum(map(len, files.values())) + len(raw) <= MAX_BUNDLE_INPUT_BYTES, "Evidence bundle is too large"
    )
    value = json.loads(raw)
    _finite(value)
    files[relative] = raw
    return value


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _checkpoint(root, relative, expected, files):
    require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), "Invalid checkpoint digest")
    path = _path(root, relative, MAX_CHECKPOINT_BYTES)
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    require(actual == expected, f"Checkpoint changed: {relative}")
    sidecar = relative + ".sha256"
    raw = _path(root, sidecar, 256).read_bytes()
    require(raw.decode("ascii").strip() == expected, f"Checkpoint sidecar mismatch: {relative}")
    files[sidecar] = raw


def _training(root, name, cell, files):
    width, episodes, seed = CELLS[name]
    prefix = f"study/cells/{name}"
    expected_parameters = 14 * width**2 + 410 * width + 3862
    epochs = 1536 // (episodes // 8)
    require(
        cell.get("status") == "completed"
        and cell.get("hidden_dim") == width
        and cell.get("training_episodes") == episodes
        and cell.get("initialization") == seed
        and cell.get("parameters") == expected_parameters
        and cell.get("optimizer_updates") == 1536
        and cell.get("validation_opportunities") == 16
        and cell.get("validation_updates") == UPDATES,
        f"Incomplete or unexpected formal cell: {name}",
    )
    manifest = _json(root, f"{prefix}/manifest.json", files)
    history = _json(root, f"{prefix}/history.json", files)
    epoch_history = _json(root, f"{prefix}/epoch_history.json", files)
    require(
        _digest(files[f"{prefix}/manifest.json"]) == cell.get("manifest_sha256"), f"Manifest changed: {name}"
    )
    require(
        _digest(files[f"{prefix}/history.json"]) == cell.get("history_sha256"), f"History changed: {name}"
    )
    require(
        _digest(files[f"{prefix}/epoch_history.json"]) == cell.get("epoch_history_sha256"),
        f"Epoch history changed: {name}",
    )
    expected = {
        "seed": seed,
        "epochs": epochs,
        "completed_epochs": epochs,
        "episodes": episodes,
        "batch_size": 8,
        "validation_episodes": 192,
        "validation_every_updates": 96,
        "optimizer_updates": 1536,
        "validation_opportunities": 16,
        "validation_updates": UPDATES,
        "train_episode_seeds": list(range(TRAIN_BASE, TRAIN_BASE + episodes)),
        "validation_episode_seeds": list(range(VALIDATION_BASE, VALIDATION_BASE + 192)),
        "parameters": expected_parameters,
        "resume_source": None,
        "seed_plan": {
            "initialization": seed,
            "train_seed_base": TRAIN_BASE,
            "validation_seed_base": VALIDATION_BASE,
            "batch_order_seed": ORDER_SEED,
        },
    }
    require(
        all(key in manifest and manifest[key] == value for key, value in expected.items()),
        f"Training protocol mismatch: {name}",
    )
    require(
        manifest.get("runtime", {}).get("device") == "cuda"
        and manifest.get("model", {}).get("architecture") == "structured-attention"
        and manifest.get("model", {}).get("hidden_dim") == width
        and manifest.get("model_selection_status") == "selected",
        f"Training device, architecture or checkpoint selection mismatch: {name}",
    )
    require(
        isinstance(history, list)
        and len(history) == 16
        and [row.get("optimizer_updates") for row in history] == UPDATES,
        f"Validation history is incomplete: {name}",
    )
    require(
        isinstance(epoch_history, list)
        and len(epoch_history) == epochs
        and [row.get("epoch") for row in epoch_history] == list(range(1, epochs + 1))
        and [row.get("optimizer_updates") for row in epoch_history]
        == [k * episodes // 8 for k in range(1, epochs + 1)],
        f"Epoch history is incomplete: {name}",
    )
    require(
        all(
            isinstance(row.get("validation_position_rmse_m"), (int, float))
            and row["validation_position_rmse_m"] >= 0
            for row in history
        ),
        f"Invalid validation scores: {name}",
    )
    best = min(history, key=lambda row: row["validation_position_rmse_m"])
    require(
        cell.get("best_update") == manifest.get("best_update") == best["optimizer_updates"],
        f"Selected checkpoint does not match the declared validation rule: {name}",
    )
    require(
        cell.get("best_validation_rmse_m")
        == manifest.get("best_validation_rmse_m")
        == best["validation_position_rmse_m"],
        f"Selected validation score differs from history: {name}",
    )
    initial = cell.get("initial_parameters_sha256")
    require(
        isinstance(initial, str)
        and re.fullmatch("[0-9a-f]{64}", initial)
        and initial == manifest.get("initial_parameters_sha256"),
        f"Initialization fingerprint mismatch: {name}",
    )
    require(cell.get("checkpoint") == f"cells/{name}/model.pt", f"Unexpected checkpoint path: {name}")
    _checkpoint(root, f"{prefix}/model.pt", cell.get("checkpoint_sha256"), files)
    return initial


def _pairs(rows, label):
    require(isinstance(rows, list) and len(rows) == 48, f"Incomplete paired episodes: {label}")
    observed = [(row.get("seed"), row.get("condition")) for row in rows]
    require(
        len(set(observed)) == 48 and set(observed) == PAIRS, f"Fresh-test episode identities differ: {label}"
    )


def package(root: Path, out: Path):
    root, out = Path(root).resolve(strict=True), Path(out)
    require(root.is_dir(), "Expected a completed run directory")
    staging = out.with_suffix(out.suffix + ".writing")
    require(not (out.exists() or out.is_symlink()), "Archive already exists; earlier evidence is preserved")
    require(
        not (staging.exists() or staging.is_symlink()),
        "Interrupted archive staging exists; choose a new output",
    )
    files = {}
    preflight = _json(root, "preflight.json", files)
    study = _json(root, "study/study.json", files)
    status = _json(root, "status.json", files)
    revision, job = preflight.get("git_commit"), preflight.get("slurm", {}).get("job_id")
    require(isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision), "Invalid source revision")
    require(isinstance(job, str) and re.fullmatch("[0-9]+", job), "Invalid Slurm job identity")
    require(
        preflight.get("status") == "passed"
        and study.get("status") == "completed"
        and study.get("profile") == "formal"
        and study.get("git_commit") == revision
        and study.get("slurm_job_id") == job
        and study.get("runtime", {}).get("device") == "cuda",
        "Only completed, matching formal CUDA studies can be packaged",
    )
    require(
        status.get("slurm_job_id") == job
        and status.get("git_commit") == revision
        and (status.get("status"), status.get("stage")) in {("running", "package"), ("completed", "done")},
        "Job status does not identify the packaging stage of this study",
    )
    gpu = preflight.get("gpu", {})
    require(
        "H200" in gpu.get("name", "")
        and str(gpu.get("mig_mode", "")).lower() == "disabled"
        and gpu.get("total_memory_bytes", 0) >= 120 * 1024**3,
        "Preflight does not establish a whole H200",
    )
    cells = study.get("cells", {})
    require(set(cells) == set(CELLS), "Missing or unexpected formal capacity/data cell")
    starts = {}
    for name, cell in cells.items():
        initial = _training(root, name, cell, files)
        width, _, seed = CELLS[name]
        require(
            starts.setdefault((width, seed), initial) == initial, "Initial weights differ across data sizes"
        )
    require(
        all(starts[width, 7] != starts[width, 11] for width in WIDTHS),
        "Initialization levels are not distinct",
    )

    fresh = _json(root, "study/fresh_test.json", files)
    require(
        fresh.get("status") == "completed"
        and fresh.get("profile") == "formal"
        and fresh.get("git_commit") == revision,
        "Fresh test is incomplete or from a different experiment",
    )
    _pairs(fresh.get("rows"), "fresh_test")
    for row in fresh["rows"]:
        scores = row.get("scores", {})
        require(set(scores) == METHODS, "Fresh test is missing a learned or classical method")
        for name, scores_for_method in scores.items():
            for field in ("position_error_auc_m_s", "three_step_forecast_rmse_m"):
                value = scores_for_method.get(field)
                require(isinstance(value, (int, float)) and value >= 0, f"Missing/non-finite {field}: {name}")
            post = scores_for_method.get("post_errors_m")
            require(
                isinstance(post, list)
                and len(post) == 11
                and all(isinstance(v, (int, float)) and v >= 0 for v in post),
                f"Incomplete recovery window: {name}",
            )
    summary = fresh.get("summary", {})
    require(set(summary) == {"control", "maneuver"}, "Missing challenge condition in summary")
    require(
        all(set(value.get("methods", {})) == METHODS for value in summary.values()),
        "Incomplete method summary",
    )
    require(study.get("results", {}).get("fresh_test") == summary, "Study and fresh-test summaries disagree")

    for name, cell in cells.items():
        diagnostics = _json(root, f"study/diagnostics/{name}.json", files)
        require(
            diagnostics.get("status") == "completed"
            and diagnostics.get("git_commit") == revision
            and diagnostics.get("cell") == name
            and diagnostics.get("checkpoint_sha256") == cell["checkpoint_sha256"],
            f"Correction diagnostics are incomplete or have different provenance: {name}",
        )
        _pairs(diagnostics.get("episodes"), f"diagnostics/{name}")
        require(
            all(
                isinstance(row.get("report"), dict) and row["report"].get("status") == "completed"
                for row in diagnostics["episodes"]
            ),
            f"Missing correction diagnostics: {name}",
        )

    # Preserve the actual status at packaging; never fabricate a terminal Slurm state.
    files["job_status_at_packaging.json"] = files.pop("status.json")
    readout = {
        "git_commit": revision,
        "slurm_job_id": job,
        "profile": "formal",
        "completed_cells": len(cells),
        "updates_per_cell": 1536,
        "validation_opportunities_per_cell": 16,
        "fresh_test": summary,
    }
    files["readout.json"] = json.dumps(readout, indent=2, allow_nan=False).encode() + b"\n"
    bundle = {
        "schema_version": 1,
        "kind": "capacity-data-evidence-bundle",
        "status": "completed",
        "completion_scope": (
            "All declared training, fresh-test and correction-diagnostic records were verified and packaged. "
            "Terminal scheduler completion is reported separately by sacct and status.json. "
            "job_status_at_packaging.json is an actual packaging-time snapshot."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "slurm_job_id": job,
        "profile": "formal",
        "artifacts": {name: {"sha256": _digest(raw), "bytes": len(raw)} for name, raw in files.items()},
        "checkpoint_transfer": "Weights remain on Hamilton; checkpoint digests were verified before packaging",
    }
    files["bundle_manifest.json"] = json.dumps(bundle, indent=2, allow_nan=False).encode() + b"\n"
    # Hard-link publication fails if the destination appears concurrently; no previous file is replaced.
    with staging.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(content), 0o600
                archive.addfile(info, io.BytesIO(content))
        raw.flush()
        os.fsync(raw.fileno())
    out.hardlink_to(staging)
    staging.unlink()
    with out.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"status": "packaged", "archive": str(out), "sha256": digest, "bytes": out.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.run_dir, args.out), allow_nan=False))


if __name__ == "__main__":
    main()
