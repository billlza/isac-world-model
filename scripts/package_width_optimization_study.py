"""Verify formal Phase D training and checkpoint replays; package evidence only.

Selected model files keep their 256 MiB bound. Final continuation files use the
explicit 2 GiB resume reader, and their CURRENT state is bound to the final replay.
The earlier Phase C run is read-only. The external terminal status is retrieved
separately because evidence packaging precedes scheduler completion.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import normalized_capacity_study as phase_c
import width_optimization_study as study
from package_capacity_study import MAX_BUNDLE_INPUT_BYTES, MAX_JSON_BYTES, _finite, _path, require
from package_normalized_capacity_study import SCHOOL_RUNTIME, require_runtime

BEST_BYTES = 256 * 1024**2
RESUME_BYTES = 2 * 1024**3
MAX_MEMBERS = 4096
HISTORICAL_FILES = ("study", "status", "manifest", "history", "epoch_history", "visual_diagnostics")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _keep(files, name, raw):
    previous = len(files.get(name, b""))
    require(
        sum(map(len, files.values())) - previous + len(raw) <= MAX_BUNDLE_INPUT_BYTES,
        "Evidence bundle is too large",
    )
    require(name in files or len(files) < MAX_MEMBERS, "Too many evidence members")
    files[name] = raw


def _read(root, relative, budget):
    path = _path(root, relative, budget)
    with path.open("rb") as stream:
        raw = stream.read(budget + 1)
    require(0 < len(raw) <= budget, f"Evidence grew beyond size budget: {relative}")
    return raw


def _json(root, relative, files):
    raw = _read(root, relative, MAX_JSON_BYTES)
    value = json.loads(raw)
    _finite(value)
    _keep(files, relative, raw)
    return value


def _hash_file(path, budget):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(min(1024**2, budget + 1 - count))
            if not block:
                break
            count += len(block)
            require(count <= budget, "Checkpoint grew beyond its bounded resource budget")
            digest.update(block)
    require(count > 0, "Checkpoint is empty")
    return digest.hexdigest()


def _checkpoint(root, relative, expected, kind, files, *, archive_sidecar=None):
    from isac_wm.artifacts import read_checkpoint, read_resume_checkpoint

    require(kind in {"best", "final"}, "Unknown checkpoint kind")
    require(
        isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid checkpoint digest"
    )
    budget = BEST_BYTES if kind == "best" else RESUME_BYTES
    path = _path(root, relative, budget)
    require(_hash_file(path, budget) == expected, f"Checkpoint changed: {relative}")
    raw = _read(root, relative + ".sha256", 128)
    require(raw.decode("ascii").strip() == expected, f"Checkpoint sidecar mismatch: {relative}")
    _keep(files, archive_sidecar or relative + ".sha256", raw)
    # These readers deserialize the exact bounded bytes whose sidecar they verify,
    # check ZIP expansion, and use weights_only=True. The report's tensor fingerprint
    # below separately binds the loaded current/selected state to the replay.
    reader = read_checkpoint if kind == "best" else read_resume_checkpoint
    return reader(path)


def _validate_final(payload, manifest, evidence, spec, key, *, historical=False):
    expected = {
        "schema_version": 2,
        "optimizer_updates": manifest["optimizer_updates"],
        "completed_epochs": manifest["completed_epochs"],
        "model_config": manifest["model"],
        "scene_config": manifest["scene"],
        "seed": manifest["seed"],
        "episodes": manifest["episodes"],
        "batch_size": manifest["batch_size"],
        "validation_episodes": manifest["validation_episodes"],
        "validation_every_updates": manifest["validation_every_updates"],
        "learning_rate": manifest["learning_rate"],
        "vision_loss_weight": manifest["vision_loss_weight"],
        "vision_supervision_protocol": manifest["vision_supervision_protocol"],
        "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
        "diagnostic_protocol": manifest["diagnostic_protocol"],
        "seed_plan": manifest["seed_plan"],
        "history": evidence["history"],
        "epoch_history": evidence["epoch_history"],
        "best_rmse": manifest["best_validation_rmse_m"],
        "event_counts_seen": manifest["event_counts_seen"],
        "visual_diagnostics_snapshots": evidence["visual_diagnostics"]["snapshots"],
    }
    require(isinstance(payload, dict), "Final checkpoint is not a mapping")
    for name, value in expected.items():
        require(payload.get(name) == value, f"Final checkpoint differs from completed evidence: {name}")
    require(isinstance(payload.get("best_checkpoint"), dict), "Final checkpoint lacks its selected state")
    if historical:
        phase_c.validate_selected_checkpoint(payload["best_checkpoint"], manifest, spec, key)
        phase_c.validate_checkpoint_weights(payload, key)
    else:
        for field in ("width_lr_scale", "optimizer_policy"):
            require(payload.get(field) == manifest[field], f"Final optimizer policy differs: {field}")
        study.validate_selected_checkpoint(payload["best_checkpoint"], manifest, spec, key)
        study.validate_checkpoint_weights(payload, key)


def _replays(root, links, manifest, revision, manifest_sha, checkpoints, files, *, prefix):
    from isac_wm.checkpoint_replay import validate_replay_report

    require(
        isinstance(links, dict) and set(links) == {"best", "final"}, "Both checkpoint replays are required"
    )
    reports = {}
    for kind in ("best", "final"):
        link = links[kind]
        filename = f"{kind}.json" if prefix == "historical-replay" else f"replay-{kind}.json"
        relative = f"{prefix}/{filename}"
        require(isinstance(link, dict) and link.get("path") == relative, "Unexpected checkpoint replay path")
        archive_relative = f"study/{relative}"
        replay = _json(root, archive_relative, files)
        require(_digest(files[archive_relative]) == link.get("sha256"), "Checkpoint replay evidence changed")
        validate_replay_report(
            replay,
            manifest,
            kind=kind,
            device_name="cuda",
            profile="formal",
            source_commit=revision,
            expected_checkpoint_sha256=checkpoints[kind],
            expected_manifest_sha256=manifest_sha,
        )
        reports[kind] = replay
    return reports


def _historical(root, record, revision, files):
    from isac_wm.checkpoint_replay import state_dict_fingerprint

    require(
        isinstance(record, dict)
        and record.get("status") == "completed"
        and record.get("source_commit") == study.HISTORICAL_COMMIT
        and record.get("source_job_id") == study.HISTORICAL_JOB
        and record.get("source_cell") == study.HISTORICAL_CELL
        and record.get("evaluator_git_commit") == revision,
        "Historical replay provenance differs from the registered Phase C source",
    )
    source_run = record.get("source_run")
    require(isinstance(source_run, str) and Path(source_run).is_absolute(), "Invalid historical run path")
    source = Path(source_run)
    require(source.is_dir() and not source.is_symlink(), "Historical source run is missing or non-regular")
    source = source.resolve(strict=True)
    links = record.get("source_files")
    require(
        isinstance(links, dict) and set(links) == set(HISTORICAL_FILES),
        "Historical source evidence is incomplete",
    )
    copied = {}
    for name in HISTORICAL_FILES:
        relative = f"historical-replay/source-{name}.json"
        require(links[name].get("path") == relative, "Unexpected historical source-copy path")
        copied[name] = _json(root, f"study/{relative}", files)
        raw = files[f"study/{relative}"]
        require(_digest(raw) == links[name].get("sha256"), f"Historical copied {name} digest differs")
        old_relative = (
            "study/study.json"
            if name == "study"
            else "status.json"
            if name == "status"
            else f"study/cells/{study.HISTORICAL_CELL}/{name}.json"
        )
        require(_read(source, old_relative, MAX_JSON_BYTES) == raw, f"Historical source changed: {name}")
    require(
        links["study"]["sha256"] == study.HISTORICAL_STUDY_SHA256,
        "Historical study differs from audited archive",
    )
    require(
        links["manifest"]["sha256"] == study.HISTORICAL_MANIFEST_SHA256, "Historical manifest anchor differs"
    )
    old_study, status = copied["study"], copied["status"]
    old_spec = phase_c.design("formal")
    require(
        old_study.get("status") == "completed"
        and old_study.get("profile") == "formal"
        and old_study.get("git_commit") == study.HISTORICAL_COMMIT
        and old_study.get("slurm_job_id") == study.HISTORICAL_JOB
        and old_study.get("design") == old_spec
        and old_study.get("fresh_test_consumed") is False
        and old_study.get("test_episode_seeds") == [],
        "Historical Phase C study identity or protocol differs",
    )
    require(
        status.get("status") == "completed"
        and status.get("stage") == "done"
        and status.get("exit_code") == 0
        and status.get("git_commit") == study.HISTORICAL_COMMIT
        and status.get("slurm_job_id") == study.HISTORICAL_JOB,
        "Historical source has not completed with the registered identity",
    )
    old_cell = old_study["cells"][study.HISTORICAL_CELL]
    for item in study.EVIDENCE_FILES:
        require(links[item]["sha256"] == old_cell[f"{item}_sha256"], "Historical training evidence changed")
    manifest = copied["manifest"]
    metrics = phase_c.validate_training_evidence(
        manifest,
        copied["history"],
        copied["epoch_history"],
        copied["visual_diagnostics"],
        old_spec,
        (512, 11),
        "cuda",
    )
    require(metrics == old_cell["metrics"], "Historical metrics do not reproduce")
    source_evidence = record.get("source_evidence", {})
    for field, expected in {
        "study_sha256": study.HISTORICAL_STUDY_SHA256,
        "manifest_sha256": study.HISTORICAL_MANIFEST_SHA256,
        "checkpoint_sha256": study.HISTORICAL_CHECKPOINT_SHA256,
    }.items():
        require(source_evidence.get(field) == expected, f"Historical source digest differs: {field}")
    checkpoints = {
        "best": source_evidence["checkpoint_sha256"],
        "final": source_evidence.get("resume_sha256"),
    }
    reports = _replays(
        root,
        record.get("reports"),
        manifest,
        study.HISTORICAL_COMMIT,
        study.HISTORICAL_MANIFEST_SHA256,
        checkpoints,
        files,
        prefix="historical-replay",
    )
    require(
        reports["final"]["recorded_score"]["expected_rmse_m"]
        == copied["history"][-1]["validation_position_rmse_m"],
        "Historical final replay score differs from the retained final validation",
    )
    selected_fingerprint = None
    for kind, filename in (("best", "model.pt"), ("final", "resume.pt")):
        payload = _checkpoint(
            source,
            f"study/cells/{study.HISTORICAL_CELL}/{filename}",
            checkpoints[kind],
            kind,
            files,
            archive_sidecar=f"study/historical-replay/source-{filename}.sha256",
        )
        if kind == "best":
            phase_c.validate_selected_checkpoint(payload, manifest, old_spec, (512, 11))
            phase_c.validate_checkpoint_weights(payload, (512, 11))
            selected_fingerprint = state_dict_fingerprint(payload["state_dict"])
        else:
            _validate_final(payload, manifest, copied, old_spec, (512, 11), historical=True)
            require(
                state_dict_fingerprint(payload["best_checkpoint"]["state_dict"]) == selected_fingerprint,
                "Historical final checkpoint embeds a different selected model",
            )
        require(
            state_dict_fingerprint(payload["state_dict"]) == reports[kind]["state_dict_fingerprint"],
            f"Historical {kind} replay does not match the actual checkpoint state",
        )
        del payload
    verification = record.get("source_verification", {})
    require(
        verification.get("source_mutated") is False
        and verification.get("optimizer_steps") == 0
        and verification.get("fresh_test_consumed") is False
        and verification.get("final_protocol_and_weights_validated") is True,
        "Historical replay must remain read-only and use the retained final current state",
    )
    require(
        verification.get("best_digest_origin") == "previously audited Phase C archive"
        and verification.get("final_digest_origin")
        == ("retained Hamilton sidecar first recorded by Phase D; no resume digest in Phase C archive"),
        "Historical digest provenance is incomplete or overstated",
    )
    compact = {}
    for kind, replay in reports.items():
        compact[kind] = {
            "weights_update": replay["weights_update"],
            "checkpoint_sha256": replay["checkpoint_sha256"],
            "state_dict_fingerprint": replay["state_dict_fingerprint"],
            "paths": {
                name: {
                    field: data[field]
                    for field in ("position_error", "trajectory_rmse", "visual_capture_error")
                }
                for name, data in replay["paths"].items()
            },
            "recorded_score": replay["recorded_score"],
            "comparisons": {
                name: {field: value for field, value in data.items() if field != "trajectories"}
                for name, data in replay["comparisons"].items()
            },
        }
    return {"record": record, "replays": compact, "checkpoint_weights_revalidated": True}


def package(run_dir, destination):
    from isac_wm.artifacts import atomic_write
    from isac_wm.checkpoint_replay import state_dict_fingerprint

    source = Path(run_dir)
    require(source.is_dir() and not source.is_symlink(), "Expected a regular completed run directory")
    root, out = source.resolve(strict=True), Path(destination).absolute()
    staging = out.with_suffix(out.suffix + ".writing")
    require(not out.exists() and not out.is_symlink(), "Evidence archive already exists")
    require(not staging.exists() and not staging.is_symlink(), "Evidence archive staging exists")
    files = {}
    preflight = _json(root, "preflight.json", files)
    report = _json(root, "study/study.json", files)
    status = _json(root, "status.json", files)
    revision, job = report.get("git_commit", ""), report.get("slurm_job_id", "")
    require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
        and isinstance(job, str)
        and re.fullmatch(r"[0-9]+", job),
        "Invalid pinned source or Slurm identity",
    )
    spec = study.design("formal")
    require(
        report.get("status") == "completed"
        and report.get("profile") == "formal"
        and report.get("design") == spec
        and report.get("scene") == spec["scene"]
        and report.get("fresh_test_consumed") is False
        and report.get("test_episode_seeds") == []
        and report.get("deployment_validated") is False,
        "Expected complete registered formal Phase D study without fresh test or deployment claims",
    )
    require_runtime(report.get("runtime"), "study")
    settings = report.get("numerical_settings")
    study.validate_numerical_settings(settings)
    require(settings["cuda_runtime"] == "12.8", "Pinned study CUDA runtime differs")
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
    runtime, gpu = preflight.get("runtime", {}), preflight.get("gpu", {})
    require(
        runtime.get("python") == SCHOOL_RUNTIME["python"]
        and runtime.get("torch") == SCHOOL_RUNTIME["torch"]
        and runtime.get("torch_cuda") == "12.8",
        "Pinned Hamilton preflight runtime differs",
    )
    require(
        "H200" in gpu.get("name", "")
        and gpu.get("mig_mode") == "Disabled"
        and gpu.get("total_memory_bytes", 0) >= 120 * 1024**3,
        "Whole H200 evidence is missing",
    )
    historical = _historical(root, report.get("historical_replay"), revision, files)
    keys = {study.cell_name(*key): key for key in study.cells(spec)}
    records = report.get("cells", {})
    require(
        isinstance(records, dict) and set(records) == set(keys), "Missing or unexpected formal Phase D cell"
    )
    require(report.get("cell_order") == list(keys), "Study cell order differs from registration")
    checkpoint_records, all_replays = {}, {}
    for name, key in keys.items():
        cell = records[name]
        width, scale, seed = key
        for field, value in {
            "status": "completed",
            "hidden_width": width,
            "width_lr_scale": scale,
            "initialization": seed,
            "latent_state_norm": "layernorm",
            "learning_rate": study.LEARNING_RATE,
            "vision_loss_weight": study.VISION_WEIGHT,
            "parameters": study.PARAMETERS_BY_WIDTH[width],
            "visual_encoder_parameters": study.VISUAL_PARAMETERS,
            "checkpoint_metadata_validated": True,
            "checkpoint_weights_validated": True,
            "final_checkpoint_metadata_validated": True,
            "final_checkpoint_weights_validated": True,
        }.items():
            require(cell.get(field) == value, f"Cell protocol differs: {name}/{field}")
        require(cell.get("numerical_settings") == settings, f"Cell numerical settings differ: {name}")
        evidence, prefix = {}, f"study/cells/{name}"
        for item in study.EVIDENCE_FILES:
            relative = f"{prefix}/{item}.json"
            evidence[item] = _json(root, relative, files)
            require(
                _digest(files[relative]) == cell.get(f"{item}_sha256"), f"Changed {item} evidence: {name}"
            )
        manifest = evidence["manifest"]
        require_runtime(manifest.get("runtime"), name)
        metrics = study.validate_training_evidence(
            manifest,
            evidence["history"],
            evidence["epoch_history"],
            evidence["visual_diagnostics"],
            spec,
            key,
            "cuda",
        )
        require(cell.get("metrics") == metrics, f"Cell metrics differ from training evidence: {name}")
        for field in ("initial_parameters_sha256", "initial_visual_parameters_sha256", "optimizer_policy"):
            require(cell.get(field) == manifest[field], f"Cell manifest identity differs: {name}/{field}")
        require(cell.get("checkpoint") == f"cells/{name}/model.pt", "Unexpected selected checkpoint path")
        require(
            cell.get("resume_checkpoint") == f"cells/{name}/resume.pt", "Unexpected final checkpoint path"
        )
        checksums = {"best": cell.get("checkpoint_sha256"), "final": cell.get("resume_checkpoint_sha256")}
        replays = _replays(
            root,
            cell.get("replays"),
            manifest,
            revision,
            cell["manifest_sha256"],
            checksums,
            files,
            prefix=f"cells/{name}",
        )
        require(
            replays["final"]["recorded_score"]["expected_rmse_m"]
            == evidence["history"][-1]["validation_position_rmse_m"],
            f"Final replay score differs from the recorded endpoint: {name}",
        )
        all_replays[name], selected_fingerprint = replays, None
        checkpoint_records[name] = {}
        for kind, filename in (("best", "model.pt"), ("final", "resume.pt")):
            relative = f"{prefix}/{filename}"
            payload = _checkpoint(root, relative, checksums[kind], kind, files)
            if kind == "best":
                study.validate_selected_checkpoint(payload, manifest, spec, key)
                study.validate_checkpoint_weights(payload, key)
                selected_fingerprint = state_dict_fingerprint(payload["state_dict"])
            else:
                _validate_final(payload, manifest, evidence, spec, key)
                require(
                    state_dict_fingerprint(payload["best_checkpoint"]["state_dict"]) == selected_fingerprint,
                    "Final checkpoint embeds a different selected model",
                )
            fingerprint = state_dict_fingerprint(payload["state_dict"])
            require(
                fingerprint == replays[kind]["state_dict_fingerprint"],
                f"Replay uses different {kind} weights: {name}",
            )
            checkpoint_records[name][kind] = {
                "path": relative,
                "sha256": checksums[kind],
                "state_dict_fingerprint": fingerprint,
                "weights_update": replays[kind]["weights_update"],
                "hidden_width": width,
                "parameters": study.PARAMETERS_BY_WIDTH[width],
                "width_lr_scale": scale,
                "checkpoint_metadata_validated": True,
                "checkpoint_weights_validated": True,
                "weights_included": False,
            }
            del payload
    study.validate_pairing(records, spec)
    summary, replay_summary = (
        study.summarize(records, spec),
        study.summarize_replays(records, all_replays, spec),
    )
    require(report.get("summary") == summary, "Study summary differs from completed cells")
    require(
        report.get("replay_summary") == replay_summary, "Replay summary differs from checkpoint-bound reports"
    )
    # Preserve all additional finite JSON reports in this run, with bounded bytes
    # and member counts. Every registered report above still has strict provenance.
    for path in sorted((root / "study").rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if relative not in files:
            _json(root, relative, files)
    readout = {
        "schema_version": 1,
        "phase": "D",
        "git_commit": revision,
        "slurm_job_id": job,
        "fresh_test_consumed": False,
        "deployment_validated": False,
        "design": spec,
        "numerical_settings": settings,
        "summary": summary,
        "replay_summary": replay_summary,
        "historical_replay": historical,
        "scope": "Paired optimizer-step intervention and unchanged-checkpoint validation replay; no fresh test or deployment claim",
    }
    raw_readout = (json.dumps(readout, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw_readout) <= MAX_JSON_BYTES, "Readout exceeds the JSON byte budget")
    readout_path = root / "readout.json"
    if readout_path.exists() or readout_path.is_symlink():
        require(
            _read(root, "readout.json", MAX_JSON_BYTES) == raw_readout,
            "Existing readout differs; preserve and inspect it",
        )
    else:
        atomic_write(readout_path, raw_readout)
    _keep(files, "readout.json", raw_readout)
    files["job_status_at_packaging.json"] = files.pop("status.json")
    bundle = {
        "schema_version": 1,
        "kind": "isac-width-optimization-phase-d",
        "status": "complete_scientific_stage",
        "completion_scope": "Historical read-only replay plus six fresh controlled cells and both checkpoint replays verified; terminal scheduler state is supplied separately",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "slurm_job_id": job,
        "profile": "formal",
        "fresh_test_consumed": False,
        "deployment_validated": False,
        "weights_included": False,
        "checkpoints": checkpoint_records,
        "historical_source": report["historical_replay"]["source_evidence"],
        "historical_checkpoint_weights_revalidated": True,
        "artifacts": {
            name: {"sha256": _digest(raw), "bytes": len(raw)} for name, raw in sorted(files.items())
        },
    }
    _keep(files, "bundle_manifest.json", (json.dumps(bundle, indent=2, allow_nan=False) + "\n").encode())
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
    return {
        "status": "packaged",
        "archive": str(out),
        "sha256": _hash_file(out, MAX_BUNDLE_INPUT_BYTES),
        "bytes": out.stat().st_size,
        "weights_included": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.run_dir, args.out)), flush=True)


if __name__ == "__main__":
    main()
