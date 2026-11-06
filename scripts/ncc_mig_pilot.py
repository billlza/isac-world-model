"""Run the real A512 working set on one NCC 1g.10gb instance.

This bounded development check does not run, select, or resume a formal Phase A
cell. It deliberately retains the formal model, batch size and sequence length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from vision_learning_study import MODEL, PARAMETERS, SCENE, VISUAL_PARAMETERS

DEVELOPMENT_TRAIN_BASE = 4_080_000_000
DEVELOPMENT_VALIDATION_BASE = 4_080_100_000
DIAGNOSTIC_UPDATES = (0, 1, 2, 4)
WEIGHTS = (0.0, 1.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_arguments(weight):
    require(weight in WEIGHTS, "Unregistered visual-supervision variant")
    return {
        "epochs": 2,
        "episodes": 16,
        "batch_size": 8,
        "seed": 97,
        "learning_rate": 0.001,
        "patience": 0,
        "train_seed_base": DEVELOPMENT_TRAIN_BASE,
        "validation_seed_base": DEVELOPMENT_VALIDATION_BASE,
        "batch_order_seed": 37,
        "validation_episodes": 8,
        "validation_every_updates": 2,
        "vision_loss_weight": weight,
        "diagnostic_updates": DIAGNOSTIC_UPDATES,
    }


def finite(value, *, nonnegative=False):
    return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)


def validate_cell(directory, manifest, weight, *, expected_device="cuda"):
    """Check actual training evidence; CPU is used only by the compatibility CI."""
    expected = {
        "parameters": PARAMETERS,
        "visual_encoder_parameters": VISUAL_PARAMETERS,
        "model": MODEL,
        "scene": SCENE,
        "epochs": 2,
        "completed_epochs": 2,
        "episodes": 16,
        "batch_size": 8,
        "seed": 97,
        "learning_rate": 0.001,
        "early_stopping_patience": 0,
        "optimizer_updates": 4,
        "optimizer_updates_this_call": 4,
        "validation_updates": [2, 4],
        "validation_opportunities": 2,
        "validation_episodes": 8,
        "vision_loss_weight": weight,
        "resume_source": None,
        "model_selection_status": "selected",
        "diagnostic_observed_updates": list(DIAGNOSTIC_UPDATES),
        "train_episode_seeds": list(range(DEVELOPMENT_TRAIN_BASE, DEVELOPMENT_TRAIN_BASE + 16)),
        "validation_episode_seeds": list(range(DEVELOPMENT_VALIDATION_BASE, DEVELOPMENT_VALIDATION_BASE + 8)),
    }
    for field, value in expected.items():
        require(manifest.get(field) == value, f"A512 development protocol mismatch: {field}")
    require(manifest.get("runtime", {}).get("device") == expected_device, "Wrong training device")
    require(finite(manifest.get("best_validation_rmse_m"), nonnegative=True), "Non-finite validation RMSE")
    require(finite(manifest.get("training_seconds"), nonnegative=True), "Missing synchronized duration")
    require(manifest.get("event_counts_seen", {}).get("snapshots") == 4 * 8 * 64, "Wrong snapshot budget")
    for field in ("initial_parameters_sha256", "initial_visual_parameters_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", manifest.get(field, "")) is not None, "Missing initial hash")
    history = json.loads((directory / "history.json").read_text())
    require([row.get("optimizer_updates") for row in history] == [2, 4], "Missing validation steps")
    require(
        all(
            finite(row.get("train_loss")) and finite(row.get("validation_position_rmse_m"), nonnegative=True)
            for row in history
        ),
        "Non-finite loss/validation history",
    )
    diagnostics = json.loads((directory / "visual_diagnostics.json").read_text())
    snapshots = diagnostics.get("snapshots", [])
    require(
        [row.get("optimizer_updates") for row in snapshots] == list(DIAGNOSTIC_UPDATES),
        "Missing actual gradient/update observations",
    )
    for row in snapshots[1:]:
        branches = row.get("training", {}).get("branches", {})
        require(bool(branches), "Missing parameter branches")
        for branch in branches.values():
            require(
                all(
                    finite(branch.get(field), nonnegative=True)
                    for field in ("gradient_norm", "clipped_gradient_norm", "update_norm")
                ),
                "Invalid observed gradient or parameter update",
            )
        require(any(b["update_norm"] > 0 for b in branches.values()), "No parameters were updated")
        if weight == 1.0:
            visual = branches.get("visual", {})
            require(
                visual.get("gradient_norm", 0) > 0 and visual.get("update_norm", 0) > 0,
                "Directly supervised visual branch did not receive a finite update",
            )
    hashes = {}
    for name in ("model.pt", "resume.pt"):
        path = directory / name
        require(path.is_file() and 0 < path.stat().st_size <= 256 * 1024**2, "Missing/bad checkpoint")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(path.with_suffix(path.suffix + ".sha256").read_text().strip() == digest, "Checkpoint changed")
        hashes[name] = digest
    if expected_device == "cuda":
        for field in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            require(type(manifest.get(field)) is int and manifest[field] > 0, "Missing CUDA memory evidence")
        require(bool(manifest.get("cuda_memory_scope")), "Missing memory measurement scope")
    return {
        "parameters": manifest["parameters"],
        "optimizer_updates": manifest["optimizer_updates"],
        "training_seconds": manifest["training_seconds"],
        "training_timing_scope": manifest.get("training_timing_scope"),
        "initial_parameters_sha256": manifest["initial_parameters_sha256"],
        "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
        "cuda_peak_allocated_bytes": manifest.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": manifest.get("cuda_peak_reserved_bytes"),
        "cuda_memory_scope": manifest.get("cuda_memory_scope"),
        "best_validation_rmse_m": manifest["best_validation_rmse_m"],
        "checkpoint_sha256": hashes,
        "gradient_and_update_evidence": "visual_diagnostics.json",
    }


def validate_preflight(preflight, expected_commit, job_id):
    require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "Invalid source pin")
    require(re.fullmatch(r"[0-9]+", job_id) is not None, "Run inside the submitted Slurm job")
    require(preflight.get("status") == "passed", "GPU preflight did not pass")
    require(preflight.get("git_commit") == expected_commit, "Preflight source differs from this pilot")
    require(preflight.get("slurm", {}).get("job_id") == job_id, "Preflight belongs to another job")
    hardware = preflight.get("hardware", {})
    require(hardware.get("profile") == "1g.10gb", "Pilot requires the explicit NCC MIG profile")
    require(
        type(hardware.get("total_memory_bytes")) is int and hardware["total_memory_bytes"] > 0,
        "Missing instance memory",
    )


def run(outdir, preflight_path, expected_commit):
    outdir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "NCC MIG A512 development fit/runtime check; not a formal Phase A result",
        "git_commit": expected_commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "precision": "float32; no autocast; no compile",
        "model": MODEL,
        "scene": SCENE,
        "training_arguments": [train_arguments(weight) for weight in WEIGHTS],
        "fresh_test_consumed": False,
        "formal_training_or_validation_seeds_consumed": [],
        "cells": [],
    }
    output = outdir / "pilot.json"
    write_json(output, report)
    try:
        preflight = json.loads(preflight_path.read_text())
        validate_preflight(preflight, expected_commit, report["slurm_job_id"])
        import torch

        from isac_wm.simulator import SceneConfig
        from isac_wm.structured import StructuredConfig
        from isac_wm.training import resolve_device, runtime_info, train

        device = resolve_device("cuda")
        require(torch.cuda.device_count() == 1, "Exactly one allocated CUDA instance is required")
        torch.set_num_threads(2)
        report["runtime"] = runtime_info(device)
        report["backend"] = {
            "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "matmul_float32_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        report["hardware"] = preflight["hardware"]
        for weight in WEIGHTS:
            name = f"a512-vision-{int(weight)}"
            print(json.dumps({"stage": "model_check", "cell": name}), flush=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            directory = outdir / name
            manifest = train(
                directory,
                device_name="cuda",
                scene=SceneConfig(**SCENE),
                model_config=StructuredConfig(**MODEL),
                **train_arguments(weight),
            )
            torch.cuda.synchronize(device)
            evidence = validate_cell(directory, manifest, weight)
            evidence.update(
                name=name, vision_loss_weight=weight, whole_cell_seconds=time.perf_counter() - started
            )
            report["cells"].append(evidence)
            write_json(output, report)
        require(
            report["cells"][0]["initial_parameters_sha256"]
            == report["cells"][1]["initial_parameters_sha256"],
            "Two visual branches did not start from identical random weights",
        )
        report["status"] = "completed"
        write_json(output, report)
        print(json.dumps({"status": "completed", "evidence": str(output)}), flush=True)
        return report
    except BaseException as failure:
        report["status"] = "failed"
        report["error"] = {"type": type(failure).__name__, "message": str(failure)}
        write_json(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.out, args.preflight, args.expected_commit)


if __name__ == "__main__":
    main()
