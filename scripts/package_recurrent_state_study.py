"""Verify all four formal Phase B cells and package evidence without model weights."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from package_capacity_study import _checkpoint, _digest, _json, require
from recurrent_state_study import (
    LEARNING_RATE,
    PARAMETERS,
    VISION_WEIGHT,
    VISUAL_PARAMETERS,
    cell_name,
    cells,
    design,
    summarize,
    validate_training_evidence,
)

SCHOOL_RUNTIME = {"python": "3.13.9", "torch": "2.9.0+cu128", "numpy": "2.3.4", "device": "cuda"}


def require_runtime(runtime, label):
    require(
        isinstance(runtime, dict) and all(runtime.get(k) == v for k, v in SCHOOL_RUNTIME.items()),
        f"Pinned Hamilton runtime differs: {label}",
    )


def package(run_dir, destination):
    root = Path(run_dir).resolve(strict=True)
    out = Path(destination).absolute()
    staging = out.with_suffix(out.suffix + ".writing")
    require(not out.exists() and not out.is_symlink(), "Evidence archive already exists")
    require(not staging.exists() and not staging.is_symlink(), "Evidence archive staging exists")
    files = {}
    preflight = _json(root, "preflight.json", files)
    study = _json(root, "study/study.json", files)
    status = _json(root, "status.json", files)
    revision, job = study.get("git_commit", ""), study.get("slurm_job_id", "")
    require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
        and isinstance(job, str)
        and re.fullmatch(r"[0-9]+", job) is not None,
        "Invalid pinned source or Slurm identity",
    )
    spec = design("formal")
    require(
        study.get("status") == "completed"
        and study.get("profile") == "formal"
        and study.get("design") == spec
        and study.get("scene") == spec["scene"]
        and study.get("fresh_test_consumed") is False
        and study.get("test_episode_seeds") == [],
        "Expected completed matching formal CUDA Phase B study without test consumption",
    )
    require_runtime(study.get("runtime"), "study")
    require(
        preflight.get("status") == "passed"
        and preflight.get("git_commit") == revision
        and preflight.get("slurm", {}).get("job_id") == job
        and status.get("git_commit") == revision
        and status.get("slurm_job_id") == job,
        "Preflight, study and stage identities differ",
    )
    require(
        (status.get("status"), status.get("stage")) in {("running", "package"), ("completed", "done")}
        and (
            (status["status"] == "running" and status.get("exit_code") is None)
            or (status["status"] == "completed" and status.get("exit_code") == 0)
        ),
        "Job has not reached successful study packaging",
    )
    preflight_runtime = preflight.get("runtime", {})
    require(
        preflight_runtime.get("python") == SCHOOL_RUNTIME["python"]
        and preflight_runtime.get("torch") == SCHOOL_RUNTIME["torch"]
        and preflight_runtime.get("torch_cuda") == "12.8",
        "Pinned Hamilton preflight runtime differs",
    )
    gpu = preflight.get("gpu", {})
    require(
        "H200" in gpu.get("name", "")
        and gpu.get("mig_mode") == "Disabled"
        and gpu.get("total_memory_bytes", 0) >= 120 * 1024**3,
        "Whole H200 evidence is missing",
    )
    expected_cells = {cell_name(*key): key for key in cells(spec)}
    records = study.get("cells", {})
    require(
        isinstance(records, dict) and set(records) == set(expected_cells),
        "Missing or unexpected formal recurrent-state cell",
    )
    require(study.get("cell_order") == list(expected_cells), "Study cell order differs from registration")
    starts, visual_starts = {}, {}
    event_counts, diagnostic_counts = None, None
    checkpoints = {}
    for name, key in expected_cells.items():
        cell = records[name]
        norm, initialization = key
        require(
            isinstance(cell, dict)
            and cell.get("status") == "completed"
            and cell.get("latent_state_norm") == norm
            and cell.get("learning_rate") == LEARNING_RATE
            and cell.get("vision_loss_weight") == VISION_WEIGHT
            and cell.get("initialization") == initialization
            and cell.get("parameters") == PARAMETERS
            and cell.get("visual_encoder_parameters") == VISUAL_PARAMETERS
            and cell.get("checkpoint_metadata_validated") is True,
            f"Cell protocol differs: {name}",
        )
        prefix = f"study/cells/{name}"
        evidence = {}
        for item in ("manifest", "history", "epoch_history", "visual_diagnostics"):
            relative = f"{prefix}/{item}.json"
            evidence[item] = _json(root, relative, files)
            require(
                _digest(files[relative]) == cell.get(f"{item}_sha256"), f"Changed {item} evidence: {name}"
            )
        require_runtime(evidence["manifest"].get("runtime"), name)
        metrics = validate_training_evidence(
            evidence["manifest"],
            evidence["history"],
            evidence["epoch_history"],
            evidence["visual_diagnostics"],
            spec,
            key,
            "cuda",
        )
        require(cell.get("metrics") == metrics, f"Cell metrics differ from training evidence: {name}")
        for cache, field in (
            (starts, "initial_parameters_sha256"),
            (visual_starts, "initial_visual_parameters_sha256"),
        ):
            fingerprint = evidence["manifest"][field]
            require(
                cell.get(field) == fingerprint
                and cache.setdefault(initialization, fingerprint) == fingerprint,
                "Initial weights differ across normalization conditions",
            )
        if event_counts is None:
            event_counts = metrics["event_counts_seen"]
            diagnostic_counts = metrics["diagnostic_event_counts"]
        require(
            event_counts == metrics["event_counts_seen"], "Training event cadence differs across paired cells"
        )
        require(
            diagnostic_counts == metrics["diagnostic_event_counts"],
            "Diagnostic event cadence differs across paired cells",
        )
        relative = f"cells/{name}/model.pt"
        require(cell.get("checkpoint") == relative, "Unexpected selected checkpoint path")
        _checkpoint(root, f"study/{relative}", cell.get("checkpoint_sha256", ""), files)
        checkpoints[name] = {
            "path": f"study/{relative}",
            "sha256": cell["checkpoint_sha256"],
            "best_update": evidence["manifest"]["best_update"],
            "best_validation_rmse_m": metrics["best_validation_rmse_m"],
            "latent_state_norm": norm,
            "checkpoint_metadata_validated": True,
            "weights_included": False,
        }
    require(
        starts[7] != starts[11] and visual_starts[7] != visual_starts[11],
        "Different initialization seeds have identical weights",
    )
    expected_summary = summarize(records, spec)
    require(study.get("summary") == expected_summary, "Study summary differs from all four cell records")
    files["job_status_at_packaging.json"] = files.pop("status.json")
    readout = {
        "schema_version": 1,
        "phase": "B",
        "git_commit": revision,
        "slurm_job_id": job,
        "fresh_test_consumed": False,
        "design": spec,
        "summary": expected_summary,
        "scope": (
            "Paired recurrent-state normalization at fixed A512 capacity, visual supervision, "
            "data, update cadence and selection budget; descriptive training/validation results only"
        ),
    }
    files["readout.json"] = (json.dumps(readout, indent=2, allow_nan=False) + "\n").encode()
    bundle = {
        "schema_version": 1,
        "kind": "isac-recurrent-state-phase-b",
        "status": "complete_scientific_stage",
        "completion_scope": (
            "All four Phase B training/validation cells verified; terminal scheduler state is supplied separately"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "slurm_job_id": job,
        "profile": "formal",
        "fresh_test_consumed": False,
        "weights_included": False,
        "checkpoints": checkpoints,
        "artifacts": {
            name: {"sha256": _digest(raw), "bytes": len(raw)} for name, raw in sorted(files.items())
        },
    }
    files["bundle_manifest.json"] = (json.dumps(bundle, indent=2, allow_nan=False) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for name, payload in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(payload), 0o600
                archive.addfile(info, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    os.link(staging, out)
    staging.unlink()
    result_bytes = out.read_bytes()
    return {
        "status": "packaged",
        "archive": str(out),
        "sha256": _digest(result_bytes),
        "bytes": len(result_bytes),
        "weights_included": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.run_dir, args.out)), flush=True)


if __name__ == "__main__":
    main()
