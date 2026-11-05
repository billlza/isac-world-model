"""Verify and package Phase A JSON evidence; model weights stay on Hamilton."""

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
from vision_learning_study import (
    PARAMETERS,
    VISUAL_PARAMETERS,
    cell_name,
    cells,
    design,
    summarize,
    validate_training_evidence,
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
        re.fullmatch(r"[0-9a-f]{40}", revision) is not None and re.fullmatch(r"[0-9]+", job) is not None,
        "Invalid pinned source or Slurm identity",
    )
    spec = design("formal")
    require(
        study.get("status") == "completed"
        and study.get("profile") == "formal"
        and study.get("design") == spec
        and study.get("runtime", {}).get("device") == "cuda"
        and study.get("fresh_test_consumed") is False
        and study.get("test_episode_seeds") == [],
        "Expected completed matching formal CUDA Phase A study without test consumption",
    )
    require(
        preflight.get("status") == "passed"
        and preflight.get("git_commit") == revision
        and preflight.get("slurm", {}).get("job_id") == job
        and status.get("git_commit") == revision
        and status.get("slurm_job_id") == job,
        "Preflight, study and stage identities differ",
    )
    require(
        (status.get("status"), status.get("stage")) in {("running", "package"), ("completed", "done")},
        "Job has not reached successful study packaging",
    )
    gpu = preflight.get("gpu", {})
    require(
        "H200" in gpu.get("name", "")
        and gpu.get("mig_mode") == "Disabled"
        and gpu.get("total_memory_bytes", 0) >= 120 * 1024**3,
        "Whole H200 evidence is missing",
    )
    expected_cells = {cell_name(*k): k for k in cells(spec)}
    records = study.get("cells", {})
    require(set(records) == set(expected_cells), "Missing or unexpected formal visual-learning cell")
    require(study.get("cell_order") == list(expected_cells), "Study cell order differs from registration")
    starts, visual_starts, event_counts = {}, {}, None
    checkpoints = {}
    for name, key in expected_cells.items():
        cell = records[name]
        lr, weight, initialization = key
        require(
            cell.get("status") == "completed"
            and cell.get("learning_rate") == lr
            and cell.get("vision_loss_weight") == weight
            and cell.get("initialization") == initialization
            and cell.get("parameters") == PARAMETERS
            and cell.get("visual_encoder_parameters") == VISUAL_PARAMETERS,
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
                "Initial weights differ across controlled cells",
            )
        if event_counts is None:
            event_counts = metrics["event_counts_seen"]
        require(
            event_counts == metrics["event_counts_seen"], "Training event cadence differs across study cells"
        )
        relative = f"cells/{name}/model.pt"
        require(cell.get("checkpoint") == relative, "Unexpected selected checkpoint path")
        _checkpoint(root, f"study/{relative}", cell.get("checkpoint_sha256", ""), files)
        checkpoints[name] = {
            "path": f"study/{relative}",
            "sha256": cell["checkpoint_sha256"],
            "weights_included": False,
        }
    require(
        starts[7] != starts[11] and visual_starts[7] != visual_starts[11],
        "Different initialization seeds have identical weights",
    )
    expected_summary = summarize(records, spec)
    require(study.get("summary") == expected_summary, "Study summary differs from all eight cell records")
    files["job_status_at_packaging.json"] = files.pop("status.json")
    readout = {
        "schema_version": 1,
        "phase": "A",
        "git_commit": revision,
        "slurm_job_id": job,
        "fresh_test_consumed": False,
        "design": spec,
        "summary": expected_summary,
        "scope": "Learning-rate and visual-supervision pilot; unchanged architecture; descriptive training/validation results only",
    }
    files["readout.json"] = (json.dumps(readout, indent=2, allow_nan=False) + "\n").encode()
    bundle = {
        "schema_version": 1,
        "kind": "isac-vision-learning-phase-a",
        "status": "complete_scientific_stage",
        "completion_scope": "All eight Phase A training/validation cells verified; terminal scheduler state is supplied separately",
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
