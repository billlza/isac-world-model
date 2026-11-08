"""Phase B: paired recurrent-state normalization, with fixed visual supervision.

Four independently trained cells retain the Phase A data and selection budgets.
Development trajectories are disjoint; formal execution requires the pinned
Hamilton launcher and CUDA. No fresh test trajectories are generated.
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

import vision_learning_study as phase_a
from vision_learning_study import (
    MODEL,
    ORDER_SEED,
    epoch_budget,
    finite_nonnegative,
    require,
    validation_updates,
)
from vision_learning_study import (
    PARAMETERS as PARAMETERS,
)
from vision_learning_study import (
    SUPERVISION_PROTOCOL as SUPERVISION_PROTOCOL,
)
from vision_learning_study import (
    VISUAL_PARAMETERS as VISUAL_PARAMETERS,
)
from vision_learning_study import (
    registered_seeds as registered_seeds,
)

ROOT = Path(__file__).resolve().parents[1]
NORMS = ("none", "layernorm")
INITIALIZATIONS = (7, 11)
LEARNING_RATE = 0.0003
VISION_WEIGHT = 1.0
NORM_EPS = 1e-5
PHASE_A_COMMIT = "0656dd4051001edbc08a8468375271e378cc9002"
DIAGNOSTIC_UPDATES = (0, 1, 2, 4, 8, 16, 32, 64, 96, 192, 288, 384)


def registered_model(norm):
    require(norm in NORMS, "Unregistered latent-state normalization")
    # The default serialization is deliberately identical to older checkpoints.
    return MODEL.copy() if norm == "none" else MODEL | {"latent_state_norm": norm}


def design(profile):
    require(profile in ("formal", "development"), "Unknown recurrent-state study profile")
    spec = phase_a.design("formal")
    del spec["learning_rates"], spec["vision_loss_weights"]
    spec.update(
        phase="B",
        latent_state_norms=list(NORMS),
        learning_rate=LEARNING_RATE,
        vision_loss_weight=VISION_WEIGHT,
        reference_phase_a={"git_commit": PHASE_A_COMMIT, "slurm_job_id": "19373715"},
        intervention={
            "location": "complete hidden update; before physical readouts and memory writeback",
            "elementwise_affine": False,
            "eps": NORM_EPS,
            "physical_state_normalized": False,
        },
        weight_interpretation="fixed at Phase A weight 1.0; no weight search in Phase B",
        primary_comparisons=[
            "best_validation_rmse_m",
            "last_validation_rmse_m",
            "last_visual_position_rmse_m",
            "final_minus_best_validation_rmse_m",
            "late_validation_change_m",
        ],
    )
    if profile == "development":
        spec.update(
            training_episodes=16,
            train_seed_base=89110000,
            batch_size=8,
            optimizer_updates=8,
            validation_episodes=8,
            validation_seed_base=89120000,
            validation_every_updates=2,
        )
    spec["diagnostic_updates"] = sorted(
        {u for u in DIAGNOSTIC_UPDATES if u <= spec["optimizer_updates"]} | set(validation_updates(spec))
    )
    return spec


def cells(spec):
    return tuple((norm, seed) for norm in spec["latent_state_norms"] for seed in spec["initializations"])


def cell_name(norm, initialization):
    require(norm in NORMS and initialization in INITIALIZATIONS, "Unregistered recurrent-state cell")
    return f"a512-norm-{norm}-init-{initialization}"


def validate_training_evidence(manifest, history, epoch_history, diagnostics, spec, key, device_name):
    norm, initialization = key
    cell_name(*key)
    # Reuse the exact Phase A data, update, selection and supervision checks.
    # Only the explicitly registered model configuration differs across conditions.
    metrics = phase_a.validate_training_evidence(
        manifest,
        history,
        epoch_history,
        diagnostics,
        spec,
        (LEARNING_RATE, VISION_WEIGHT, initialization),
        device_name,
        expected_model=registered_model(norm),
    )
    snapshots = diagnostics["snapshots"]
    cadence = None
    for snapshot in snapshots:
        measured = snapshot["validation"]
        network = measured.get("network", {})
        for field in ("hidden_pre_norm_rms", "hidden_post_norm_rms", "hidden_rms"):
            require(
                finite_nonnegative(network.get(field)), f"Missing finite recurrent-state diagnostic: {field}"
            )
        require(
            network["hidden_rms"] == network["hidden_post_norm_rms"],
            "Reported hidden must be the state used for readout and recurrence",
        )
        if norm == "none":
            require(
                network["hidden_pre_norm_rms"] == network["hidden_post_norm_rms"],
                "Legacy condition unexpectedly changed the recurrent hidden",
            )
        else:
            require(
                network["hidden_post_norm_rms"] <= 1.00001, "Affine-free hidden normalization bound failed"
            )
        counts = measured["event_counts"]
        if cadence is None:
            cadence = counts
        require(counts == cadence, "Diagnostic observation cadence changed within a fixed subset")
    midpoint = spec["optimizer_updates"] // 2
    middle = next((r for r in history if r["optimizer_updates"] == midpoint), None)
    require(middle is not None, "Missing registered midpoint validation")
    metrics.update(
        final_minus_best_validation_rmse_m=metrics["last_validation_rmse_m"]
        - metrics["best_validation_rmse_m"],
        late_validation_change_m=metrics["last_validation_rmse_m"] - middle["validation_position_rmse_m"],
        midpoint_update=midpoint,
        midpoint_validation_rmse_m=middle["validation_position_rmse_m"],
        latent_state_norm=norm,
    )
    return metrics


def summarize(cell_reports, spec):
    keys = cells(spec)
    require(
        set(cell_reports) == {cell_name(*k) for k in keys}, "Every registered study cell must be retained"
    )
    require(
        all(cell_reports[cell_name(*k)].get("status") == "completed" for k in keys), "Incomplete study cells"
    )
    result = {"cells": {cell_name(*k): cell_reports[cell_name(*k)]["metrics"] for k in keys}, "contrasts": {}}
    for metric in spec["primary_comparisons"]:
        panel = {k: cell_reports[cell_name(*k)]["metrics"][metric] for k in keys}
        require(
            all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in panel.values()
            ),
            "Non-finite cell contrast metric",
        )
        if metric != "late_validation_change_m":
            require(all(v >= 0 for v in panel.values()), "Negative error or best-checkpoint gap")
        differences = {
            f"layernorm_minus_none_init_{seed}": panel["layernorm", seed] - panel["none", seed]
            for seed in INITIALIZATIONS
        }
        differences["layernorm_minus_none_mean_over_inits"] = sum(differences.values()) / len(INITIALIZATIONS)
        result["contrasts"][metric] = differences
    result["interpretation"] = (
        "Paired descriptive validation contrasts conditional on initializations 7 and 11; "
        "negative means lower error, a smaller final-minus-best gap, or a smaller late change. "
        "Best and fixed-endpoint metrics are both retained. A bounded hidden RMS is a property "
        "of normalization, not evidence by itself that task performance improved. No fresh test is consumed."
    )
    return result


def validate_selected_checkpoint(payload, manifest, spec, key):
    """Bind selected checkpoint metadata to the already validated training record.

    Best checkpoints record their update, not a second copy of validation RMSE.
    The manifest/history validation establishes the score at that selected update.
    """
    norm, initialization = key
    cell_name(*key)
    expected = {
        "schema_version": 2,
        "model_config": registered_model(norm),
        "scene_config": spec["scene"],
        "seed": initialization,
        "best_update": manifest["best_update"],
        "best_epoch": math.ceil(manifest["best_update"] / (spec["training_episodes"] / spec["batch_size"])),
        "train_episode_seeds": manifest["train_episode_seeds"],
        "validation_episode_seeds": manifest["validation_episode_seeds"],
        "validation_episodes": spec["validation_episodes"],
        "validation_every_updates": spec["validation_every_updates"],
        "vision_loss_weight": VISION_WEIGHT,
        "vision_supervision_protocol": manifest["vision_supervision_protocol"],
        "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
        "diagnostic_protocol": manifest["diagnostic_protocol"],
        "seed_plan": manifest["seed_plan"],
    }
    require(isinstance(payload, dict), "Best checkpoint is not a mapping")
    for field, value in expected.items():
        require(payload.get(field) == value, f"Best checkpoint differs from validated evidence: {field}")
    require(
        isinstance(payload.get("state_dict"), dict) and payload["state_dict"],
        "Best checkpoint has no model weights",
    )


def run(out, device_name, profile):
    import torch

    from isac_wm.artifacts import read_checkpoint, sha256_file, write_json
    from isac_wm.simulator import SceneConfig
    from isac_wm.structured import LATENT_STATE_NORM_EPS, StructuredConfig
    from isac_wm.training import resolve_device, runtime_info, train

    spec = design(profile)
    expected_device = "cuda" if profile == "formal" else "cpu"
    require(device_name == expected_device, f"{profile} requires --device {expected_device}")
    require(LATENT_STATE_NORM_EPS == NORM_EPS, "Normalization epsilon differs from registered intervention")
    device = resolve_device(device_name)
    torch.set_num_threads(2)
    cfg = SceneConfig(**spec["scene"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if profile == "formal":
        require(
            revision == os.environ.get("ISAC_STATE_PIN")
            and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) is not None,
            "Formal study requires the pinned Hamilton launcher",
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "profile": profile,
        "git_commit": revision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": runtime_info(device),
        "design": spec,
        "scene": cfg.to_dict(),
        "fresh_test_consumed": False,
        "test_episode_seeds": [],
        "cell_order": [cell_name(*k) for k in cells(spec)],
        "cells": {cell_name(*k): {"status": "not_started"} for k in cells(spec)},
    }
    destination = out / "study.json"
    write_json(destination, report)
    started = time.perf_counter()
    starts, visual_starts, event_counts, diagnostic_counts = {}, {}, None, None
    try:
        for key in cells(spec):
            norm, initialization = key
            name = cell_name(*key)
            folder = out / "cells" / name
            stage = {
                "status": "running",
                "latent_state_norm": norm,
                "learning_rate": LEARNING_RATE,
                "vision_loss_weight": VISION_WEIGHT,
                "initialization": initialization,
            }
            report["cells"][name] = stage
            write_json(destination, report)
            print(
                json.dumps(
                    {"stage": "training", "cell": name, "optimizer_updates": spec["optimizer_updates"]}
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
                model_config=StructuredConfig(**registered_model(norm)),
                learning_rate=LEARNING_RATE,
                train_seed_base=spec["train_seed_base"],
                validation_seed_base=spec["validation_seed_base"],
                batch_order_seed=ORDER_SEED,
                validation_episodes=spec["validation_episodes"],
                validation_every_updates=spec["validation_every_updates"],
                vision_loss_weight=VISION_WEIGHT,
                diagnostic_updates=tuple(spec["diagnostic_updates"]),
            )
            evidence = {
                n: json.loads((folder / f"{n}.json").read_text())
                for n in ("history", "epoch_history", "visual_diagnostics")
            }
            metrics = validate_training_evidence(
                manifest,
                evidence["history"],
                evidence["epoch_history"],
                evidence["visual_diagnostics"],
                spec,
                key,
                device_name,
            )
            for cache, field in (
                (starts, "initial_parameters_sha256"),
                (visual_starts, "initial_visual_parameters_sha256"),
            ):
                require(
                    cache.setdefault(initialization, manifest[field]) == manifest[field],
                    "Initial weights differ across normalization conditions",
                )
            if event_counts is None:
                event_counts = metrics["event_counts_seen"]
                diagnostic_counts = metrics["diagnostic_event_counts"]
            require(
                event_counts == metrics["event_counts_seen"],
                "Training event cadence differs across paired cells",
            )
            require(
                diagnostic_counts == metrics["diagnostic_event_counts"],
                "Diagnostic event cadence differs across paired cells",
            )
            checkpoint = folder / "model.pt"
            checksum = sha256_file(checkpoint)
            require(
                Path(str(checkpoint) + ".sha256").read_text().strip() == checksum,
                "Best checkpoint digest differs from sidecar",
            )
            payload = read_checkpoint(checkpoint)
            validate_selected_checkpoint(payload, manifest, spec, key)
            require(
                all(
                    isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                    for value in payload["state_dict"].values()
                ),
                "Best checkpoint contains invalid or non-finite model weights",
            )
            del payload
            stage.update(
                status="completed",
                wall_seconds=time.perf_counter() - cell_started,
                parameters=manifest["parameters"],
                visual_encoder_parameters=manifest["visual_encoder_parameters"],
                initial_parameters_sha256=manifest["initial_parameters_sha256"],
                initial_visual_parameters_sha256=manifest["initial_visual_parameters_sha256"],
                checkpoint=str(checkpoint.relative_to(out)),
                checkpoint_sha256=checksum,
                metrics=metrics,
                checkpoint_metadata_validated=True,
                **{
                    f"{n}_sha256": sha256_file(folder / f"{n}.json")
                    for n in ("manifest", "history", "epoch_history", "visual_diagnostics")
                },
            )
            write_json(destination, report)
        require(
            starts[7] != starts[11] and visual_starts[7] != visual_starts[11],
            "Different initializations produced identical fingerprints",
        )
        report.update(
            summary=summarize(report["cells"], spec),
            status="completed",
            wall_seconds=time.perf_counter() - started,
        )
        write_json(destination, report)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "phase": "B",
                    "fresh_test_consumed": False,
                    "evidence": str(destination),
                }
            ),
            flush=True,
        )
        return report
    except Exception as failure:
        report.update(
            status="failed",
            error={"type": type(failure).__name__, "message": str(failure)},
            wall_seconds=time.perf_counter() - started,
        )
        write_json(destination, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--profile", required=True, choices=("development", "formal"))
    args = parser.parse_args()
    run(args.out, args.device, args.profile)


if __name__ == "__main__":
    main()
