"""Phase C: fresh, paired capacity training with a normalized recurrent state.

The six cells use the same data, updates, observation cadence and validation
opportunities. Only hidden width and initialization vary. The visual encoder
remains fixed. Development runs use every registered width on disjoint data;
formal runs require the pinned Hamilton CUDA launcher. No fresh test is read.
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
from vision_learning_study import SUPERVISION_PROTOCOL as SUPERVISION_PROTOCOL
from vision_learning_study import VISUAL_PARAMETERS as VISUAL_PARAMETERS
from vision_learning_study import registered_seeds as registered_seeds

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (512, 1024, 2048)
INITIALIZATIONS = (7, 11)
PARAMETERS_BY_WIDTH = {512: 3_883_798, 1024: 15_103_766, 2048: 59_563_798}
LEARNING_RATE = 0.0003
VISION_WEIGHT = 1.0
NORM_EPS = 1e-5
PHASE_B_COMMIT = "841142a8bb59fd0bb8c67156fd74ec125955e2f3"
DIAGNOSTIC_UPDATES = (0, 1, 2, 4, 8, 16, 32, 64, 96, 192, 288, 384)
EVIDENCE_FILES = ("manifest", "history", "epoch_history", "visual_diagnostics")
EVENT_FIELDS = (
    "snapshots",
    "vision_new_valid",
    "radio_new_valid",
    "evidence_update_ticks",
    "prediction_only_ticks",
    "vision_visible_captures",
)


def registered_model(width):
    require(type(width) is int and width in WIDTHS, "Unregistered normalized-capacity width")
    return MODEL | {"hidden_dim": width, "latent_state_norm": "layernorm"}


def design(profile):
    require(profile in ("formal", "development"), "Unknown normalized-capacity profile")
    spec = phase_a.design("formal")
    for field in ("learning_rates", "vision_loss_weights", "model", "parameters"):
        del spec[field]
    spec.update(
        phase="C",
        hidden_widths=list(WIDTHS),
        models_by_width={str(width): registered_model(width) for width in WIDTHS},
        parameters_by_width={str(width): PARAMETERS_BY_WIDTH[width] for width in WIDTHS},
        parameter_formula="14*h*h + 410*h + 3862",
        latent_state_norm="layernorm",
        learning_rate=LEARNING_RATE,
        vision_loss_weight=VISION_WEIGHT,
        reference_phase_b={"git_commit": PHASE_B_COMMIT, "slurm_job_id": "19404348"},
        normalization={
            "location": "complete hidden update; before physical readouts and memory writeback",
            "elementwise_affine": False,
            "eps": NORM_EPS,
            "physical_state_normalized": False,
        },
        capacity_scope="recurrent, attention and fusion capacity; unchanged visual encoder and inputs",
        weight_interpretation="fixed at 1.0; no supervision-weight search in Phase C",
        initialization_pairing=(
            "fresh initialization per cell; same-seed visual weights match across widths; "
            "full-model weights differ because their shapes differ"
        ),
        numerical_policy=(
            "record actual numerical execution settings without modifying them; "
            "same seeds do not imply bitwise reproducibility across hardware or kernels"
        ),
        compute_policy="equal data and optimizer updates, not equal FLOPs or wall time",
        primary_comparisons=[
            "best_validation_rmse_m",
            "last_validation_rmse_m",
            "last_visual_position_rmse_m",
            "final_minus_best_validation_rmse_m",
            "late_validation_change_m",
        ],
        primary_contrasts=["1024_minus_512", "2048_minus_512"],
    )
    if profile == "development":
        spec.update(
            training_episodes=8,
            train_seed_base=89130000,
            batch_size=8,
            optimizer_updates=4,
            validation_episodes=8,
            validation_seed_base=89140000,
            validation_every_updates=1,
        )
    spec["diagnostic_updates"] = sorted(
        {u for u in DIAGNOSTIC_UPDATES if u <= spec["optimizer_updates"]} | set(validation_updates(spec))
    )
    return spec


def cells(spec):
    require(
        spec.get("hidden_widths") == list(WIDTHS) and spec.get("initializations") == list(INITIALIZATIONS),
        "All registered widths and initializations must be retained",
    )
    return tuple((width, seed) for width in WIDTHS for seed in INITIALIZATIONS)


def cell_name(width, initialization):
    registered_model(width)
    require(type(initialization) is int and initialization in INITIALIZATIONS, "Unregistered initialization")
    return f"a{width}-layernorm-init-{initialization}"


def require_finite_tree(value, label):
    """Reject non-finite scalar diagnostics without rejecting declared empty supports."""
    if isinstance(value, dict):
        for name, child in value.items():
            require_finite_tree(child, f"{label}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f"Non-finite diagnostic: {label}")


def validate_event_counts(counts, snapshots, label):
    require(
        isinstance(counts, dict)
        and set(counts) == set(EVENT_FIELDS)
        and all(type(counts[field]) is int and counts[field] >= 0 for field in EVENT_FIELDS),
        f"Invalid {label} event-count schema",
    )
    require(counts["snapshots"] == snapshots, f"Wrong {label} snapshot budget")
    require(
        counts["evidence_update_ticks"] + counts["prediction_only_ticks"] == snapshots
        and counts["vision_visible_captures"] <= counts["vision_new_valid"] <= snapshots
        and counts["radio_new_valid"] <= snapshots,
        f"Inconsistent {label} event counts",
    )


def validate_training_evidence(manifest, history, epoch_history, diagnostics, spec, key, device_name):
    width, initialization = key
    cell_name(*key)
    metrics = phase_a.validate_training_evidence(
        manifest,
        history,
        epoch_history,
        diagnostics,
        spec,
        (LEARNING_RATE, VISION_WEIGHT, initialization),
        device_name,
        expected_model=registered_model(width),
        expected_parameters=PARAMETERS_BY_WIDTH[width],
    )
    for field, expected in {
        "initialization": "random; no pretrained weights",
        "planned_optimizer_updates": spec["optimizer_updates"],
        "planned_validation_opportunities": 4,
        "validation_mode": "fixed_optimizer_updates",
        "diagnostic_observed_updates": spec["diagnostic_updates"],
        "optimizer_updates_this_call": spec["optimizer_updates"],
        "completed_epochs_this_call": epoch_budget(spec),
    }.items():
        require(manifest.get(field) == expected, f"Fresh capacity-training contract differs: {field}")
    require_finite_tree(history, "history")
    require_finite_tree(epoch_history, "epoch_history")
    require_finite_tree(diagnostics, "diagnostics")
    for rows in (history, epoch_history):
        require(
            all(
                isinstance(row.get("train_loss"), (int, float)) and not isinstance(row["train_loss"], bool)
                for row in rows
            ),
            "Missing finite training loss history",
        )
    cadence = None
    for snapshot in diagnostics["snapshots"]:
        measured = snapshot["validation"]
        network = measured.get("network", {})
        for field in ("hidden_pre_norm_rms", "hidden_post_norm_rms", "hidden_rms"):
            require(finite_nonnegative(network.get(field)), f"Missing finite recurrent diagnostic: {field}")
        require(
            network["hidden_rms"] == network["hidden_post_norm_rms"],
            "Reported hidden differs from the recurrent state used for readouts",
        )
        require(network["hidden_post_norm_rms"] <= 1.00001, "Affine-free hidden normalization bound failed")
        counts = measured["event_counts"]
        validate_event_counts(counts, min(spec["validation_episodes"], 4) * spec["steps"], "diagnostic")
        require(
            measured["visual_visible_capture_count"] == counts["vision_visible_captures"],
            "Visual diagnostic support differs from delivered visible captures",
        )
        if cadence is None:
            cadence = counts
        require(counts == cadence, "Diagnostic observation cadence changed within the fixed subset")
        training = snapshot.get("training")
        if snapshot["optimizer_updates"] == 0:
            require(training is None, "Update-zero diagnostic must precede training")
        else:
            require(isinstance(training, dict), "Missing scheduled optimizer-step diagnostic")
            validate_event_counts(
                training.get("event_counts"), spec["batch_size"] * spec["steps"], "minibatch"
            )
            branches = training.get("branches", {})
            require(isinstance(branches, dict) and branches, "Missing branch gradient/update diagnostics")
            for branch in branches.values():
                require(isinstance(branch, dict), "Invalid branch diagnostic record")
                require(
                    type(branch.get("parameters")) is int and branch["parameters"] > 0,
                    "Missing branch parameter count",
                )
                for field in (
                    "parameter_norm",
                    "gradient_norm",
                    "clipped_gradient_norm",
                    "update_norm",
                    "relative_update",
                ):
                    require(finite_nonnegative(branch.get(field)), f"Invalid branch diagnostic: {field}")
            require(
                sum(branch["parameters"] for branch in branches.values()) == PARAMETERS_BY_WIDTH[width],
                "Diagnostic branch parameter count differs from registered capacity",
            )
    midpoint = spec["optimizer_updates"] // 2
    middle = next((row for row in history if row["optimizer_updates"] == midpoint), None)
    require(middle is not None, "Missing registered midpoint validation")
    resource = {
        "training_seconds": manifest["training_seconds"],
        "dataset_estimated_bytes": manifest.get("dataset_estimated_bytes"),
        "diagnostic_overhead_seconds": manifest.get("diagnostic_overhead_seconds"),
        "cuda_peak_allocated_bytes": manifest.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": manifest.get("cuda_peak_reserved_bytes"),
        "cuda_memory_scope": manifest.get("cuda_memory_scope"),
        "timing_scope": manifest.get("training_timing_scope"),
    }
    require(
        type(resource["dataset_estimated_bytes"]) is int and resource["dataset_estimated_bytes"] > 0,
        "Missing measured dataset resource budget",
    )
    require(finite_nonnegative(resource["diagnostic_overhead_seconds"]), "Missing diagnostic duration")
    require(isinstance(resource["timing_scope"], str) and resource["timing_scope"], "Missing timing scope")
    if device_name == "cuda":
        for field in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            require(
                type(resource[field]) is int and resource[field] > 0,
                f"Missing measured CUDA resource: {field}",
            )
        require(
            resource["cuda_peak_reserved_bytes"] >= resource["cuda_peak_allocated_bytes"]
            and isinstance(resource["cuda_memory_scope"], str)
            and resource["cuda_memory_scope"],
            "Invalid CUDA allocator scope or peak accounting",
        )
    else:
        require(
            all(
                resource[field] is None
                for field in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "cuda_memory_scope")
            ),
            "CPU development evidence must not claim CUDA measurements",
        )
    metrics.update(
        hidden_width=width,
        latent_state_norm="layernorm",
        final_minus_best_validation_rmse_m=metrics["last_validation_rmse_m"]
        - metrics["best_validation_rmse_m"],
        late_validation_change_m=metrics["last_validation_rmse_m"] - middle["validation_position_rmse_m"],
        midpoint_update=midpoint,
        midpoint_validation_rmse_m=middle["validation_position_rmse_m"],
        resources=resource,
    )
    return metrics


def numerical_settings(torch):
    """Read actual settings; do not introduce a numerical intervention in Phase C."""
    result = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    validate_numerical_settings(result)
    return result


def validate_numerical_settings(value):
    require(isinstance(value, dict), "Missing actual numerical settings")
    for field in (
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cuda_matmul_allow_tf32",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
    ):
        require(type(value.get(field)) is bool, f"Missing numerical execution setting: {field}")
    require(
        value.get("float32_matmul_precision") in ("highest", "high", "medium"),
        "Missing actual float32 matmul precision",
    )
    for field in ("cublas_workspace_config", "nvidia_tf32_override", "cuda_runtime"):
        require(
            field in value and (value[field] is None or isinstance(value[field], str)), f"Missing {field}"
        )
    require(
        "cudnn_version" in value
        and (
            value["cudnn_version"] is None
            or type(value["cudnn_version"]) is int
            and value["cudnn_version"] > 0
        ),
        "Missing actual cuDNN version",
    )


def validate_pairing(cell_reports, spec, *, require_complete=True):
    expected = {cell_name(*key) for key in cells(spec)}
    require(set(cell_reports) == expected, "Every registered study cell must be retained")
    starts, visual_starts, shared_events, shared_diagnostics, shared_numerics = set(), {}, None, None, None
    for key in cells(spec):
        width, initialization = key
        row = cell_reports[cell_name(*key)]
        if not require_complete and row.get("status") != "completed":
            continue
        require(row.get("status") == "completed", "Incomplete capacity-study cells")
        require(
            row.get("hidden_width") == width
            and row.get("initialization") == initialization
            and row.get("parameters") == PARAMETERS_BY_WIDTH[width]
            and row.get("visual_encoder_parameters") == VISUAL_PARAMETERS
            and row.get("latent_state_norm") == "layernorm"
            and row.get("learning_rate") == LEARNING_RATE
            and row.get("vision_loss_weight") == VISION_WEIGHT,
            "Completed cell identity or capacity differs from registration",
        )
        for field in ("initial_parameters_sha256", "initial_visual_parameters_sha256"):
            require(
                re.fullmatch(r"[0-9a-f]{64}", row.get(field, "")) is not None, f"Missing fingerprint: {field}"
            )
        full_hash, visual_hash = row["initial_parameters_sha256"], row["initial_visual_parameters_sha256"]
        require(
            full_hash not in starts, "Different widths or seeds produced identical full-model fingerprints"
        )
        starts.add(full_hash)
        require(
            visual_starts.setdefault(initialization, visual_hash) == visual_hash,
            "Same-seed visual initialization differs across widths",
        )
        metrics = row["metrics"]
        if shared_events is None:
            shared_events = metrics["event_counts_seen"]
            shared_diagnostics = metrics["diagnostic_event_counts"]
            shared_numerics = row.get("numerical_settings")
        require(
            metrics["event_counts_seen"] == shared_events,
            "Training event cadence differs across paired cells",
        )
        require(
            metrics["diagnostic_event_counts"] == shared_diagnostics,
            "Diagnostic event cadence differs across paired cells",
        )
        validate_numerical_settings(row.get("numerical_settings"))
        require(
            row["numerical_settings"] == shared_numerics, "Numerical settings changed between capacity cells"
        )
    if len(visual_starts) == len(INITIALIZATIONS):
        require(
            len(set(visual_starts.values())) == len(INITIALIZATIONS),
            "Different visual initializations coincide",
        )


def summarize(cell_reports, spec):
    validate_pairing(cell_reports, spec)
    keys = cells(spec)
    result = {
        "cells": {cell_name(*key): cell_reports[cell_name(*key)]["metrics"] for key in keys},
        "contrasts": {},
    }
    for metric in spec["primary_comparisons"]:
        panel = {key: cell_reports[cell_name(*key)]["metrics"][metric] for key in keys}
        require(
            all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in panel.values()
            ),
            "Non-finite cell contrast metric",
        )
        if metric != "late_validation_change_m":
            require(all(v >= 0 for v in panel.values()), "Negative error or best-checkpoint gap")
        differences = {}
        for width in WIDTHS[1:]:
            for seed in INITIALIZATIONS:
                differences[f"{width}_minus_512_init_{seed}"] = panel[width, seed] - panel[512, seed]
            differences[f"{width}_minus_512_mean_over_inits"] = sum(
                panel[width, seed] - panel[512, seed] for seed in INITIALIZATIONS
            ) / len(INITIALIZATIONS)
        result["contrasts"][metric] = differences
    resource_summary = {}
    for width in WIDTHS:
        rows = [cell_reports[cell_name(width, seed)] for seed in INITIALIZATIONS]
        resources = [row["metrics"]["resources"] for row in rows]
        width_summary = {
            "parameters": PARAMETERS_BY_WIDTH[width],
            "visual_encoder_parameters": VISUAL_PARAMETERS,
        }
        for field in (
            "training_seconds",
            "diagnostic_overhead_seconds",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            values = [resource[field] for resource in resources]
            require(
                all(value is None or finite_nonnegative(value) for value in values),
                "Invalid resource summary",
            )
            require(
                all(value is None for value in values) or all(value is not None for value in values),
                "Mixed resource coverage",
            )
            width_summary[f"{field}_by_initialization"] = dict(zip(map(str, INITIALIZATIONS), values))
            width_summary[f"mean_{field}"] = None if values[0] is None else sum(values) / len(values)
        resource_summary[str(width)] = width_summary
    result["resources_by_width"] = resource_summary
    result["interpretation"] = (
        "Paired descriptive validation contrasts conditional on initializations 7 and 11; negative means "
        "lower error, a smaller final-minus-best gap, or a smaller late change. All widths start fresh and "
        "retain best and fixed-endpoint metrics. Data, updates and validation opportunities are matched; "
        "FLOPs and wall time are not matched. Resource figures include scheduled validation and diagnostics, "
        "not online decision latency or total process memory. No fresh test is consumed."
    )
    return result


def validate_selected_checkpoint(payload, manifest, spec, key):
    width, initialization = key
    cell_name(*key)
    expected = {
        "schema_version": 2,
        "model_config": registered_model(width),
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


def validate_checkpoint_weights(payload, key):
    """Bind real tensors to the registered architecture, without allocating another full model."""
    import torch

    from isac_wm.structured import StructuredConfig, build_model

    cell_name(*key)
    with torch.device("meta"):
        expected = build_model(StructuredConfig(**registered_model(key[0]))).state_dict()
    actual = payload["state_dict"]
    require(set(actual) == set(expected), "Best checkpoint weight keys differ from registered model")
    for name, value in actual.items():
        require(
            isinstance(value, torch.Tensor)
            and value.shape == expected[name].shape
            and value.dtype == expected[name].dtype
            and value.device.type == "cpu"
            and bool(torch.isfinite(value).all()),
            f"Best checkpoint contains invalid shape, dtype or non-finite weights: {name}",
        )
    require(
        sum(value.numel() for value in actual.values()) == PARAMETERS_BY_WIDTH[key[0]],
        "Checkpoint tensor budget differs",
    )


def source_revision(profile):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if profile == "formal":
        require(
            re.fullmatch(r"[0-9a-f]{40}", revision) is not None
            and revision == os.environ.get("ISAC_SCALE_PIN")
            and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) is not None,
            "Formal study requires the pinned Hamilton launcher",
        )
        changed = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        )
        untracked_code = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"], cwd=ROOT, text=True
        )
        require(
            not changed.strip() and not untracked_code.strip(),
            "Formal study requires unchanged pinned source",
        )
    return revision


def run(out, device_name, profile):
    import torch

    from isac_wm.artifacts import read_checkpoint, sha256_file, write_json
    from isac_wm.simulator import SceneConfig
    from isac_wm.structured import LATENT_STATE_NORM_EPS, StructuredConfig
    from isac_wm.training import resolve_device, runtime_info, train

    spec = design(profile)
    require(device_name == ("cuda" if profile == "formal" else "cpu"), "Profile and registered device differ")
    require(LATENT_STATE_NORM_EPS == NORM_EPS, "Normalization epsilon differs from registration")
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
        "cell_order": [cell_name(*key) for key in cells(spec)],
        "cells": {cell_name(*key): {"status": "not_started"} for key in cells(spec)},
    }
    destination = out / "study.json"
    write_json(destination, report)
    started, active_name = time.perf_counter(), None
    try:
        for key in cells(spec):
            width, initialization = key
            active_name = cell_name(*key)
            folder = out / "cells" / active_name
            stage = {
                "status": "running",
                "hidden_width": width,
                "latent_state_norm": "layernorm",
                "learning_rate": LEARNING_RATE,
                "vision_loss_weight": VISION_WEIGHT,
                "initialization": initialization,
                "numerical_settings": numerical_settings(torch),
            }
            require(
                stage["numerical_settings"] == settings, "Numerical settings changed before a capacity cell"
            )
            report["cells"][active_name] = stage
            write_json(destination, report)
            print(
                json.dumps(
                    {"stage": "training", "cell": active_name, "optimizer_updates": spec["optimizer_updates"]}
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
                train_seed_base=spec["train_seed_base"],
                validation_seed_base=spec["validation_seed_base"],
                batch_order_seed=ORDER_SEED,
                validation_episodes=spec["validation_episodes"],
                validation_every_updates=spec["validation_every_updates"],
                vision_loss_weight=VISION_WEIGHT,
                diagnostic_updates=tuple(spec["diagnostic_updates"]),
            )
            require(numerical_settings(torch) == settings, "Numerical settings changed during training")
            evidence = {
                name: json.loads((folder / f"{name}.json").read_text()) for name in EVIDENCE_FILES[1:]
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
            checkpoint = folder / "model.pt"
            checksum = sha256_file(checkpoint)
            require(
                Path(str(checkpoint) + ".sha256").read_text().strip() == checksum,
                "Best checkpoint digest differs from sidecar",
            )
            payload = read_checkpoint(checkpoint)
            validate_selected_checkpoint(payload, manifest, spec, key)
            validate_checkpoint_weights(payload, key)
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
                checkpoint_weights_validated=True,
                **{f"{name}_sha256": sha256_file(folder / f"{name}.json") for name in EVIDENCE_FILES},
            )
            validate_pairing(report["cells"], spec, require_complete=False)
            write_json(destination, report)
            active_name = None
        require(source_revision(profile) == revision, "Source revision changed while training")
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
                    "phase": "C",
                    "fresh_test_consumed": False,
                    "evidence": str(destination),
                }
            ),
            flush=True,
        )
        return report
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        if active_name is not None:
            report["cells"][active_name].update(status="failed", error=error)
        report.update(status="failed", error=error, wall_seconds=time.perf_counter() - started)
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
