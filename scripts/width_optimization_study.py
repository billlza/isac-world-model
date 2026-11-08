"""Phase D: paired large-model optimizer intervention and checkpoint error replay.

The changed factor is the learning rate of width-dependent modules. Every cell
starts fresh; old Phase C checkpoints are replayed read-only on the old validation
split. Both selected and final weights receive the same tail-error measurements.
No fresh test, pretrained weights, deployment controller or safety gate is added.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import normalized_capacity_study as phase_c
from normalized_capacity_study import (
    EVIDENCE_FILES,
    INITIALIZATIONS,
    LEARNING_RATE,
    NORM_EPS,
    ORDER_SEED,
    VISION_WEIGHT,
    VISUAL_PARAMETERS,
    epoch_budget,
    numerical_settings,
    require,
    require_finite_tree,
    validate_numerical_settings,
)
from normalized_capacity_study import registered_seeds as registered_seeds
from normalized_capacity_study import validation_updates as validation_updates

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (512, 2048)
ARMS = ((512, 1.0), (2048, 1.0), (2048, 0.5))
PARAMETERS_BY_WIDTH = {width: phase_c.PARAMETERS_BY_WIDTH[width] for width in WIDTHS}
HISTORICAL_COMMIT = "249407dbbe4039f68d7cd54c6cc336a3f9c96f06"
HISTORICAL_JOB = "19404810"
HISTORICAL_CELL = "a512-layernorm-init-11"
HISTORICAL_STUDY_SHA256 = "b9ecba054a7e2ddb1f26bca8312f751160c38236e7988d871ac305c7679ba2ee"
HISTORICAL_MANIFEST_SHA256 = "bacd15d358823c7655dd7052b5b4cbdb7fff520968fd81be782acedba3211c08"
HISTORICAL_CHECKPOINT_SHA256 = "35a24d691085748867be24763ad3e95f62696e97714bf3cd27dfea0ca9fa766f"


def registered_model(width):
    require(type(width) is int and width in WIDTHS, "Unregistered optimization-study width")
    return phase_c.registered_model(width)


def cell_name(width, width_lr_scale, initialization):
    registered_model(width)
    require(
        not isinstance(width_lr_scale, bool) and (width, width_lr_scale) in ARMS,
        "Unregistered width/learning-rate arm",
    )
    require(type(initialization) is int and initialization in INITIALIZATIONS, "Unregistered initialization")
    rate = "base" if width_lr_scale == 1.0 else "half"
    return f"a{width}-widthlr-{rate}-init-{initialization}"


def design(profile):
    spec = phase_c.design(profile)
    spec.update(
        phase="D",
        hidden_widths=list(WIDTHS),
        models_by_width={str(width): registered_model(width) for width in WIDTHS},
        parameters_by_width={str(width): PARAMETERS_BY_WIDTH[width] for width in WIDTHS},
        optimizer_arms=[{"hidden_width": width, "width_lr_scale": scale} for width, scale in ARMS],
        optimizer_intervention={
            "base_learning_rate": LEARNING_RATE,
            "width_dependent_learning_rate": {"base": LEARNING_RATE, "half": LEARNING_RATE * 0.5},
            "unchanged_learning_rate_modules": ["visual", "radio_encoder", "action_embedding"],
            "unchanged_weight_decay": 1e-4,
            "unchanged_global_clip_norm": 1.0,
            "unchanged_loss": True,
            "unchanged_model_architecture": True,
            "scope": "all remaining named trainable modules; exhaustive policy records in each manifest",
            "interpretation": "registered step-size intervention; not a claim that half is optimal",
        },
        initialization_pairing=(
            "fresh per cell; the two A2048 optimizer arms have identical full initial weights at each seed; "
            "same-seed visual weights match at both widths"
        ),
        reference_phase_c={
            "git_commit": HISTORICAL_COMMIT,
            "slurm_job_id": HISTORICAL_JOB,
            "study_sha256": HISTORICAL_STUDY_SHA256,
            "cell": HISTORICAL_CELL,
            "checkpoints": ["selected best at update 192", "final current state at update 384"],
        },
        checkpoint_replay={
            "selected_and_final": True,
            "same_validation_split": True,
            "full_validation_paths": ["batched encoding batch 8", "streaming encoding batch 8"],
            "paired_path_probe": "first four trajectories: encoding at batch 4, then batch 4 versus batch 8",
            "position_error": "Euclidean xy error in metres, ticks 5..63 inclusive",
            "tails": ["p95_m", "p99_m", "max_m"],
            "exceedance_thresholds_m": [0.1, 0.25, 0.5, 1.0],
            "threshold_scope": "descriptive fixed bins, not calibrated operating or safety limits",
            "selection_unchanged": True,
            "deployment_validated": False,
        },
        primary_contrasts=[
            "a2048_half_minus_base",
            "a2048_half_minus_a512_base",
            "a2048_base_minus_a512_base",
        ],
        deployment_validated=False,
    )
    if profile == "development":
        spec.update(train_seed_base=89150000, validation_seed_base=89160000)
    return spec


def cells(spec):
    require(
        spec.get("optimizer_arms") == [{"hidden_width": w, "width_lr_scale": s} for w, s in ARMS]
        and spec.get("hidden_widths") == list(WIDTHS)
        and spec.get("initializations") == list(INITIALIZATIONS),
        "All registered optimizer arms and initializations must be retained",
    )
    return tuple((width, scale, seed) for width, scale in ARMS for seed in INITIALIZATIONS)


def expected_optimizer_policy(width, scale):
    import torch

    from isac_wm.optimizer_policy import describe_optimizer_policy
    from isac_wm.structured import StructuredConfig, build_model

    # Shapes and names suffice. A meta model consumes no random initialization or real weight storage.
    with torch.device("meta"):
        model = build_model(StructuredConfig(**registered_model(width)))
    return describe_optimizer_policy(model, LEARNING_RATE, scale)


def validate_optimizer_evidence(manifest, key):
    width, scale, _ = key
    cell_name(*key)
    require(
        type(manifest.get("width_lr_scale")) in (int, float) and manifest["width_lr_scale"] == scale,
        "Optimizer width learning-rate scale differs from registration",
    )
    policy = expected_optimizer_policy(width, scale)
    require(manifest.get("optimizer_policy") == policy, "Optimizer parameter partition or step sizes differ")
    return policy


def validate_training_evidence(manifest, history, epoch_history, diagnostics, spec, key, device_name):
    width, scale, seed = key
    metrics = phase_c.validate_training_evidence(
        manifest, history, epoch_history, diagnostics, spec, (width, seed), device_name
    )
    validate_optimizer_evidence(manifest, key)
    metrics["width_lr_scale"] = scale
    return metrics


def validate_selected_checkpoint(payload, manifest, spec, key):
    phase_c.validate_selected_checkpoint(payload, manifest, spec, (key[0], key[2]))
    for field in ("width_lr_scale", "optimizer_policy"):
        require(
            payload.get(field) == manifest[field], f"Selected checkpoint optimizer policy differs: {field}"
        )


def validate_checkpoint_weights(payload, key):
    phase_c.validate_checkpoint_weights(payload, (key[0], key[2]))


def validate_pairing(cell_reports, spec, *, require_complete=True):
    expected = {cell_name(*key) for key in cells(spec)}
    require(set(cell_reports) == expected, "Every registered study cell must be retained")
    starts, visual_starts = {}, {}
    shared_events, shared_diagnostics, shared_numerics = None, None, None
    for key in cells(spec):
        width, scale, seed = key
        row = cell_reports[cell_name(*key)]
        if not require_complete and row.get("status") != "completed":
            continue
        require(row.get("status") == "completed", "Incomplete width-optimization cells")
        for field, value in {
            "hidden_width": width,
            "width_lr_scale": scale,
            "initialization": seed,
            "parameters": PARAMETERS_BY_WIDTH[width],
            "visual_encoder_parameters": VISUAL_PARAMETERS,
            "latent_state_norm": "layernorm",
            "learning_rate": LEARNING_RATE,
            "vision_loss_weight": VISION_WEIGHT,
        }.items():
            require(row.get(field) == value, f"Completed cell identity differs: {field}")
        for field in ("initial_parameters_sha256", "initial_visual_parameters_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", row.get(field, "")) is not None, f"Missing {field}")
        full_hash, visual_hash = row["initial_parameters_sha256"], row["initial_visual_parameters_sha256"]
        require(
            starts.setdefault((width, seed), full_hash) == full_hash,
            "Same-width optimizer intervention changed initial model weights",
        )
        require(
            visual_starts.setdefault(seed, visual_hash) == visual_hash,
            "Same-seed visual initialization differs across widths or policies",
        )
        metrics = row["metrics"]
        if shared_events is None:
            shared_events = metrics["event_counts_seen"]
            shared_diagnostics = metrics["diagnostic_event_counts"]
            shared_numerics = row.get("numerical_settings")
        require(metrics["event_counts_seen"] == shared_events, "Training observation cadence differs")
        require(
            metrics["diagnostic_event_counts"] == shared_diagnostics, "Diagnostic observation cadence differs"
        )
        validate_numerical_settings(row.get("numerical_settings"))
        require(row["numerical_settings"] == shared_numerics, "Numerical settings changed between cells")
        require(
            row.get("optimizer_policy") == expected_optimizer_policy(width, scale),
            "Recorded optimizer differs",
        )
    require(len(set(starts.values())) == len(starts), "Distinct model width/seed initializations coincide")
    require(
        len(set(visual_starts.values())) == len(visual_starts), "Distinct visual initializations coincide"
    )


def summarize(cell_reports, spec):
    validate_pairing(cell_reports, spec)
    keys = cells(spec)
    result = {
        "cells": {cell_name(*key): cell_reports[cell_name(*key)]["metrics"] for key in keys},
        "contrasts": {},
    }
    for metric in spec["primary_comparisons"]:
        values = {key: cell_reports[cell_name(*key)]["metrics"][metric] for key in keys}
        require(
            all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()),
            "Nonfinite contrast",
        )
        comparisons = {
            "a2048_half_minus_base": ((2048, 0.5), (2048, 1.0)),
            "a2048_half_minus_a512_base": ((2048, 0.5), (512, 1.0)),
            "a2048_base_minus_a512_base": ((2048, 1.0), (512, 1.0)),
        }
        panel = {}
        for name, (left, right) in comparisons.items():
            paired = {str(seed): values[(*left, seed)] - values[(*right, seed)] for seed in INITIALIZATIONS}
            panel[name] = {"by_initialization": paired, "mean_over_initializations": sum(paired.values()) / 2}
        result["contrasts"][metric] = panel
    result["interpretation"] = (
        "Paired descriptive validation contrasts at initializations 7 and 11. Negative is lower error or "
        "smaller deterioration. The A2048 optimizer arms share initial weights, data, update budget and "
        "validation opportunities; only width-dependent learning rates differ. Best-checkpoint selection "
        "still uses current-position RMSE. Replay tails are reported separately at BOTH selected and final "
        "weights, with the same validation support and inference path. No fresh test or deployment claim."
    )
    return result


def summarize_replays(cell_reports, replay_reports, spec):
    """Keep tails and visual errors aligned to an actual checkpoint and execution path."""
    result = {"cells": {}, "contrasts": {}, "deployment_validated": False}
    for key in cells(spec):
        name = cell_name(*key)
        by_kind = {}
        for kind in ("best", "final"):
            replay = replay_reports[name][kind]
            by_kind[kind] = {
                "weights_update": replay["weights_update"],
                "checkpoint_sha256": replay["checkpoint_sha256"],
                "paths": {
                    path: {
                        field: data[field]
                        for field in ("position_error", "trajectory_rmse", "visual_capture_error")
                    }
                    for path, data in replay["paths"].items()
                },
                "recorded_score": replay["recorded_score"],
                "comparisons": {
                    comparison: {field: value for field, value in values.items() if field != "trajectories"}
                    for comparison, values in replay["comparisons"].items()
                },
            }
        result["cells"][name] = by_kind
    for kind in ("best", "final"):
        result["contrasts"][kind] = {}
        for path in ("batched", "streaming"):
            panel = {}
            for metric in ("rmse_m", "p95_m", "p99_m", "max_m"):
                paired = {}
                for seed in INITIALIZATIONS:
                    candidate = result["cells"][cell_name(2048, 0.5, seed)][kind]["paths"][path][
                        "position_error"
                    ]
                    control = result["cells"][cell_name(2048, 1.0, seed)][kind]["paths"][path][
                        "position_error"
                    ]
                    paired[str(seed)] = candidate[metric] - control[metric]
                panel[metric] = {
                    "by_initialization": paired,
                    "mean_over_initializations": sum(paired.values()) / 2,
                }
            result["contrasts"][kind][path] = {"a2048_half_minus_base": panel}
    result["interpretation"] = (
        "Same-checkpoint, same-path descriptive validation tails. Pooled tick percentiles are not independent "
        "sample confidence intervals. Threshold exceedance bins are not operating limits. Improvements and "
        "regressions remain visible; no tail-based checkpoint reselection or calibrated safety claim."
    )
    return result


def source_revision(profile):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if profile == "formal":
        require(
            re.fullmatch(r"[0-9a-f]{40}", revision) is not None
            and revision == os.environ.get("ISAC_OPTIM_PIN")
            and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) is not None,
            "Formal study requires the pinned Hamilton launcher",
        )
        changed = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"], cwd=ROOT, text=True
        )
        require(
            not changed.strip() and not untracked.strip(), "Formal study requires unchanged pinned source"
        )
    return revision


def read_json_bounded(path):
    path = Path(path)
    require(
        not path.is_symlink() and path.is_file() and path.stat().st_size <= 32 * 1024**2,
        "Invalid evidence JSON file",
    )
    with path.open("rb") as source:
        raw = source.read(32 * 1024**2 + 1)
    require(len(raw) <= 32 * 1024**2, "Evidence JSON grew beyond byte budget")
    result = json.loads(raw)
    require_finite_tree(result, str(path))
    return result


def historical_source(run_dir):
    """Verify the archived Phase C evidence before reading its retained checkpoint files."""
    from isac_wm.artifacts import sha256_file

    root = Path(run_dir).resolve(strict=True)
    require(root.is_dir(), "Historical Phase C run is missing")
    study_path = root / "study/study.json"
    study = read_json_bounded(study_path)
    require(
        sha256_file(study_path) == HISTORICAL_STUDY_SHA256, "Historical Phase C study changed after audit"
    )
    status = read_json_bounded(root / "status.json")
    require(
        status.get("status") == "completed"
        and status.get("stage") == "done"
        and status.get("exit_code") == 0
        and status.get("git_commit") == HISTORICAL_COMMIT
        and status.get("slurm_job_id") == HISTORICAL_JOB,
        "Historical Phase C job identity or completion status differs",
    )
    require(
        study.get("status") == "completed"
        and study.get("profile") == "formal"
        and study.get("git_commit") == HISTORICAL_COMMIT
        and study.get("slurm_job_id") == HISTORICAL_JOB
        and study.get("design") == phase_c.design("formal")
        and study.get("fresh_test_consumed") is False,
        "Historical Phase C protocol differs",
    )
    cell = study["cells"][HISTORICAL_CELL]
    folder = root / "study/cells" / HISTORICAL_CELL
    evidence = {}
    for name in EVIDENCE_FILES:
        path = folder / f"{name}.json"
        evidence[name] = read_json_bounded(path)
        require(sha256_file(path) == cell[f"{name}_sha256"], f"Historical {name} differs from audited record")
    require(cell["manifest_sha256"] == HISTORICAL_MANIFEST_SHA256, "Historical manifest anchor differs")
    require(
        cell["checkpoint_sha256"] == HISTORICAL_CHECKPOINT_SHA256,
        "Historical selected checkpoint anchor differs",
    )
    require(
        phase_c.validate_training_evidence(
            evidence["manifest"],
            evidence["history"],
            evidence["epoch_history"],
            evidence["visual_diagnostics"],
            phase_c.design("formal"),
            (512, 11),
            "cuda",
        )
        == cell["metrics"],
        "Historical metrics differ from retained evidence",
    )
    source_files = {"study": study_path, "status": root / "status.json"}
    source_files.update({name: folder / f"{name}.json" for name in EVIDENCE_FILES})
    checkpoints = {"best": folder / "model.pt", "final": folder / "resume.pt"}
    digests = {}
    for kind, path in checkpoints.items():
        require(path.is_file() and not path.is_symlink(), f"Historical {kind} weights missing")
        limit = 256 * 1024**2 if kind == "best" else 2 * 1024**3
        require(0 < path.stat().st_size <= limit, "Historical checkpoint exceeds explicit byte budget")
        sidecar = Path(str(path) + ".sha256")
        require(
            sidecar.is_file() and not sidecar.is_symlink() and sidecar.stat().st_size < 1024,
            "Invalid checkpoint sidecar",
        )
        digest = sidecar.read_text().strip()
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "Invalid historical weight digest")
        require(sha256_file(path) == digest, "Historical weights differ from their sidecar")
        if kind == "best":
            require(
                digest == HISTORICAL_CHECKPOINT_SHA256, "Historical best weights differ from audited archive"
            )
        digests[kind] = digest
    return root, evidence["manifest"], source_files, checkpoints, digests


def replay_historical(run_dir, out, revision):
    from isac_wm.artifacts import sha256_file
    from isac_wm.checkpoint_replay import evaluate_checkpoint

    root, manifest, source_files, checkpoints, digests = historical_source(run_dir)
    destination = out / "historical-replay"
    destination.mkdir(exist_ok=False)
    copied = {}
    for name, path in source_files.items():
        copy = destination / f"source-{name}.json"
        # Preserve exact audited bytes rather than reserializing their JSON.
        with copy.open("xb") as handle:
            handle.write(path.read_bytes())
        copied[name] = {"path": str(copy.relative_to(out)), "sha256": sha256_file(copy)}
    reports = {}
    for kind in ("best", "final"):
        output = destination / f"{kind}.json"
        print(
            json.dumps({"stage": "historical_replay", "cell": HISTORICAL_CELL, "checkpoint_kind": kind}),
            flush=True,
        )
        replay = evaluate_checkpoint(
            checkpoints[kind],
            source_files["manifest"],
            kind=kind,
            device_name="cuda",
            profile="formal",
            source_commit=HISTORICAL_COMMIT,
            expected_checkpoint_sha256=digests[kind],
            expected_manifest_sha256=HISTORICAL_MANIFEST_SHA256,
            output=output,
        )
        expected_score = (
            manifest["best_validation_rmse_m"]
            if kind == "best"
            else read_json_bounded(source_files["history"])[-1]["validation_position_rmse_m"]
        )
        require(
            replay["recorded_score"]["expected_rmse_m"] == expected_score,
            "Historical checkpoint history differs from the audited score",
        )
        reports[kind] = {"path": str(output.relative_to(out)), "sha256": sha256_file(output)}
    require(
        sha256_file(source_files["study"]) == HISTORICAL_STUDY_SHA256,
        "Historical source changed during replay",
    )
    return {
        "status": "completed",
        "source_commit": HISTORICAL_COMMIT,
        "source_job_id": HISTORICAL_JOB,
        "source_run": str(root),
        "source_cell": HISTORICAL_CELL,
        "evaluator_git_commit": revision,
        "source_files": copied,
        "reports": reports,
        "source_evidence": {
            "study_sha256": HISTORICAL_STUDY_SHA256,
            "manifest_sha256": HISTORICAL_MANIFEST_SHA256,
            "checkpoint_sha256": digests["best"],
            "resume_sha256": digests["final"],
        },
        "source_verification": {
            "best_digest_origin": "previously audited Phase C archive",
            "final_digest_origin": "retained Hamilton sidecar first recorded by Phase D; no resume digest in Phase C archive",
            "final_protocol_and_weights_validated": True,
            "source_mutated": False,
            "optimizer_steps": 0,
            "fresh_test_consumed": False,
        },
    }


def run(out, device_name, profile, historical_run=None):
    import torch

    from isac_wm.artifacts import read_checkpoint, sha256_file, write_json
    from isac_wm.checkpoint_replay import evaluate_checkpoint
    from isac_wm.simulator import SceneConfig
    from isac_wm.structured import LATENT_STATE_NORM_EPS, StructuredConfig
    from isac_wm.training import resolve_device, runtime_info, train

    spec = design(profile)
    require(device_name == ("cuda" if profile == "formal" else "cpu"), "Profile and registered device differ")
    require(
        (historical_run is not None) == (profile == "formal"),
        "Formal historical replay requires the original Phase C run",
    )
    require(LATENT_STATE_NORM_EPS == NORM_EPS, "Normalization differs from registration")
    revision = source_revision(profile)
    device = resolve_device(device_name)
    torch.set_num_threads(2)
    settings = numerical_settings(torch)
    cfg = SceneConfig(**spec["scene"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "profile": profile,
        "git_commit": revision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": runtime_info(device),
        "numerical_settings": settings,
        "design": spec,
        "scene": cfg.to_dict(),
        "fresh_test_consumed": False,
        "test_episode_seeds": [],
        "deployment_validated": False,
        "historical_replay": {"status": "not_started" if profile == "formal" else "not_applicable"},
        "cell_order": [cell_name(*key) for key in cells(spec)],
        "cells": {cell_name(*key): {"status": "not_started"} for key in cells(spec)},
    }
    destination = out / "study.json"
    write_json(destination, report)
    started, active_name, active_stage = time.perf_counter(), None, "historical_replay"
    replays = {}
    try:
        if profile == "formal":
            report["historical_replay"] = replay_historical(historical_run, out, revision)
            write_json(destination, report)
        for key in cells(spec):
            width, scale, initialization = key
            active_name, active_stage = cell_name(*key), "training"
            folder = out / "cells" / active_name
            stage = {
                "status": "running",
                "hidden_width": width,
                "width_lr_scale": scale,
                "latent_state_norm": "layernorm",
                "learning_rate": LEARNING_RATE,
                "vision_loss_weight": VISION_WEIGHT,
                "initialization": initialization,
                "numerical_settings": numerical_settings(torch),
            }
            require(stage["numerical_settings"] == settings, "Numerical settings changed before cell")
            report["cells"][active_name] = stage
            write_json(destination, report)
            print(
                json.dumps(
                    {
                        "stage": active_stage,
                        "cell": active_name,
                        "optimizer_updates": spec["optimizer_updates"],
                    }
                ),
                flush=True,
            )
            cell_started = time.perf_counter()
            manifest = train(
                folder,
                epochs=epoch_budget(spec),
                episodes=spec["training_episodes"],
                batch_size=spec["batch_size"],
                device_name=device_name,
                seed=initialization,
                scene=cfg,
                model_config=StructuredConfig(**registered_model(width)),
                learning_rate=LEARNING_RATE,
                width_lr_scale=scale,
                train_seed_base=spec["train_seed_base"],
                validation_seed_base=spec["validation_seed_base"],
                batch_order_seed=ORDER_SEED,
                validation_episodes=spec["validation_episodes"],
                validation_every_updates=spec["validation_every_updates"],
                vision_loss_weight=VISION_WEIGHT,
                diagnostic_updates=tuple(spec["diagnostic_updates"]),
            )
            require(numerical_settings(torch) == settings, "Numerical settings changed during training")
            evidence = {name: read_json_bounded(folder / f"{name}.json") for name in EVIDENCE_FILES[1:]}
            metrics = validate_training_evidence(
                manifest,
                evidence["history"],
                evidence["epoch_history"],
                evidence["visual_diagnostics"],
                spec,
                key,
                device_name,
            )
            checkpoint, resume = folder / "model.pt", folder / "resume.pt"
            checksums = {"best": sha256_file(checkpoint), "final": sha256_file(resume)}
            payload = read_checkpoint(checkpoint)
            validate_selected_checkpoint(payload, manifest, spec, key)
            validate_checkpoint_weights(payload, key)
            del payload
            manifest_sha = sha256_file(folder / "manifest.json")
            replays[active_name], replay_links = {}, {}
            for kind, weights in (("best", checkpoint), ("final", resume)):
                active_stage = f"replay_{kind}"
                path = folder / f"replay-{kind}.json"
                print(json.dumps({"stage": active_stage, "cell": active_name}), flush=True)
                replays[active_name][kind] = evaluate_checkpoint(
                    weights,
                    folder / "manifest.json",
                    kind=kind,
                    device_name=device_name,
                    profile=profile,
                    source_commit=revision,
                    expected_checkpoint_sha256=checksums[kind],
                    expected_manifest_sha256=manifest_sha,
                    output=path,
                )
                require(
                    replays[active_name][kind]["recorded_score"]["expected_rmse_m"]
                    == metrics["best_validation_rmse_m" if kind == "best" else "last_validation_rmse_m"],
                    "Replayed checkpoint history differs from retained training history",
                )
                replay_links[kind] = {"path": str(path.relative_to(out)), "sha256": sha256_file(path)}
            require(numerical_settings(torch) == settings, "Replay changed numerical execution settings")
            stage.update(
                status="completed",
                wall_seconds=time.perf_counter() - cell_started,
                parameters=manifest["parameters"],
                visual_encoder_parameters=manifest["visual_encoder_parameters"],
                initial_parameters_sha256=manifest["initial_parameters_sha256"],
                initial_visual_parameters_sha256=manifest["initial_visual_parameters_sha256"],
                optimizer_policy=manifest["optimizer_policy"],
                checkpoint=str(checkpoint.relative_to(out)),
                checkpoint_sha256=checksums["best"],
                resume_checkpoint=str(resume.relative_to(out)),
                resume_checkpoint_sha256=checksums["final"],
                metrics=metrics,
                replays=replay_links,
                checkpoint_metadata_validated=True,
                checkpoint_weights_validated=True,
                final_checkpoint_metadata_validated=True,
                final_checkpoint_weights_validated=True,
                **{f"{name}_sha256": sha256_file(folder / f"{name}.json") for name in EVIDENCE_FILES},
            )
            validate_pairing(report["cells"], spec, require_complete=False)
            write_json(destination, report)
            active_name = None
        active_stage = "summary"
        require(source_revision(profile) == revision, "Source revision changed during study")
        report.update(
            status="completed",
            summary=summarize(report["cells"], spec),
            replay_summary=summarize_replays(report["cells"], replays, spec),
            wall_seconds=time.perf_counter() - started,
        )
        write_json(destination, report)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "phase": "D",
                    "fresh_test_consumed": False,
                    "evidence": str(destination),
                }
            ),
            flush=True,
        )
        return report
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure), "stage": active_stage}
        if active_name is not None:
            report["cells"][active_name].update(status="failed", error=error)
        elif active_stage == "historical_replay":
            report["historical_replay"].update(status="failed", error=error)
        report.update(status="failed", error=error, wall_seconds=time.perf_counter() - started)
        write_json(destination, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--profile", choices=("formal", "development"), required=True)
    parser.add_argument("--historical-run", type=Path)
    args = parser.parse_args()
    run(args.out, args.device, args.profile, args.historical_run)


if __name__ == "__main__":
    main()
