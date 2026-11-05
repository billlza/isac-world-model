"""Optional, bounded runtime diagnosis; never a scientific training run.

The unprofiled timing pass and operator-profile pass are separate. This script
does not compile, replace or modify model operations, and saves no checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_SEED_BASE = 4_080_000_000
INITIALIZATION_SEED = 97
SYNC_CANDIDATES = (
    "_local_scalar_dense",
    "aten::item",
    "cudadevicesynchronize",
    "cudastreamsynchronize",
    "cudastreamwaitevent",
    "linalg_solve",
    "linalg_check_errors",
)


def bounded_count(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def plan(*, warmup_updates=2, measured_updates=8, profile_updates=3, batch_size=8):
    """No seed override: this diagnostic uses a dedicated development corpus."""
    bounded_count(warmup_updates, "warmup_updates", 2)
    bounded_count(measured_updates, "measured_updates", 8)
    bounded_count(profile_updates, "profile_updates", 3)
    bounded_count(batch_size, "batch_size", 8)
    total = warmup_updates + measured_updates + profile_updates
    return {
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "profile_updates": profile_updates,
        "total_optimizer_updates_per_variant": total,
        "batch_size": batch_size,
        "initialization_seed": INITIALIZATION_SEED,
        "development_episode_seeds": list(
            range(DEVELOPMENT_SEED_BASE, DEVELOPMENT_SEED_BASE + total * batch_size)
        ),
        "formal_train_validation_test_seeds_consumed": [],
    }


def source_identity():
    """Hash the executable source independently of git availability/dirty state."""
    paths = sorted((ROOT / "src/isac_wm").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {}
    aggregate = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = digest
        aggregate.update(f"{relative}\0{digest}\n".encode())
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        revision = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    return {
        "git_commit": revision,
        "source_sha256": aggregate.hexdigest(),
        "source_file_sha256": hashes,
        "git_commit_is_not_a_clean_tree_claim": True,
    }


def timing_summary(seconds):
    ordered = sorted(seconds)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else sum(ordered[middle - 1 : middle + 1]) / 2
    return {
        "update_seconds": seconds,
        "total_seconds": sum(seconds),
        "median_update_seconds": median,
        "p95_update_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p95_method": "nearest rank; very small descriptive sample, not a service-level bound",
        "updates_per_second": len(seconds) / sum(seconds),
    }


def operator_statistics(profiler):
    rows = []
    for event in profiler.key_averages():
        rows.append(
            {
                "operator": event.key,
                "calls": event.count,
                "self_cpu_us": float(event.self_cpu_time_total),
                "total_cpu_us": float(event.cpu_time_total),
                "self_device_us": float(event.self_device_time_total),
                "total_device_us": float(event.device_time_total),
            }
        )
    return {
        "operator_groups": len(rows),
        "instrumented_scopes": [row for row in rows if row["operator"].startswith("diagnostic.")],
        "top_self_cpu": sorted(rows, key=lambda x: x["self_cpu_us"], reverse=True)[:50],
        "top_self_device": sorted(rows, key=lambda x: x["self_device_us"], reverse=True)[:50],
        "synchronization_candidates": [
            row for row in rows if any(token in row["operator"].lower() for token in SYNC_CANDIDATES)
        ][:40],
        "interpretation": (
            "Candidate operator counts are clues, not proof of a bottleneck. Nested operator counts "
            "and total times overlap. Explicit whole-update timing synchronizations are outside "
            "this operator-profile pass. Profiler overhead is excluded from unprofiled timings."
        ),
    }


def require_device_activity(statistics, device_name):
    """CUPTI/runtime failures can produce an empty CUDA trace despite API support."""
    if device_name == "cuda" and not any(row["self_device_us"] > 0 for row in statistics["top_self_device"]):
        raise RuntimeError(
            "CUDA profiling produced no positive device time; refusing CPU-only profiler evidence"
        )


def run_profile(
    outdir,
    *,
    device_name,
    warmup_updates=2,
    measured_updates=8,
    profile_updates=3,
    batch_size=8,
    threads=2,
    hidden_dims=(96, 512),
    scene=None,
):
    """Internal shape overrides exist for CPU tests; the CLI fixes A96/A512/64 ticks."""
    spec = plan(
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        profile_updates=profile_updates,
        batch_size=batch_size,
    )
    bounded_count(threads, "threads", 16)
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be explicitly cpu or cuda; no automatic fallback")
    if (
        not hidden_dims
        or len(hidden_dims) > 2
        or len(set(hidden_dims)) != len(hidden_dims)
        or any(
            isinstance(h, bool) or not isinstance(h, int) or h < 8 or h > 512 or h % 4 for h in hidden_dims
        )
    ):
        raise ValueError("Use one or two distinct hidden dimensions, multiples of 4 in [8, 512]")
    import torch

    from isac_wm.artifacts import model_fingerprint, write_json
    from isac_wm.model import training_loss
    from isac_wm.simulator import SceneConfig, recorded_episode
    from isac_wm.structured import StructuredConfig, build_model
    from isac_wm.training import _batch_to_device, batchify, resolve_device, runtime_info, seed_all

    device = resolve_device(device_name)
    scene = scene or SceneConfig(steps=64, dt=0.1)
    if scene.steps > 64 or scene.image_size > 24 or scene.dt != 0.1:
        raise ValueError("Diagnostic budget requires steps<=64, image_size<=24 and dt=0.1")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("CUDA profiling was requested but the profiler cannot collect CUDA activity")
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "optional synthetic development runtime diagnosis; no accuracy or scaling claim",
        "plan": spec,
        "scene": scene.to_dict(),
        "variants": [],
        "timing_scope": (
            "Cached CPU minibatch transfer, zero_grad, forward, existing loss/checks, backward, "
            "gradient clipping and AdamW step. Dataset generation, initialization, validation, "
            "checkpoint I/O and profiler collection excluded. CUDA synchronized only at "
            "whole-update timing boundaries; existing model/loss synchronization is preserved."
        ),
        "precision": "float32; no autocast; no compile; backend defaults recorded below",
    }
    write_json(outdir / "profile.json", report)
    try:
        torch.set_num_threads(threads)
        report["source"] = source_identity()
        report["runtime"] = runtime_info(device) | {
            "torch_cuda_build": torch.version.cuda,
            "machine": platform.machine(),
            "cudnn_version": torch.backends.cudnn.version(),
            "matmul_float32_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            report["device"] = {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "visible_device_count": torch.cuda.device_count(),
            }
        else:
            report["device"] = {"name": "cpu", "total_memory_bytes": None}

        # Precompute once, then reuse byte-identical batches and order for both widths.
        batches, batch_hashes = [], []
        seeds = spec["development_episode_seeds"]
        for offset in range(0, len(seeds), batch_size):
            batch = batchify(
                [recorded_episode(s, scene) for s in seeds[offset : offset + batch_size]],
                torch.device("cpu"),
            )
            digest = hashlib.sha256()
            for key, tensor in sorted(batch.items()):
                array = tensor.numpy()
                digest.update(f"{key}\0{array.dtype}\0{array.shape}\0".encode())
                digest.update(array.tobytes())
            batches.append(batch)
            batch_hashes.append(digest.hexdigest())
        report["cpu_minibatch_sha256"] = batch_hashes

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for hidden in hidden_dims:
            seed_all(INITIALIZATION_SEED)
            config = StructuredConfig(hidden_dim=hidden, room_m=scene.room_m)
            model = build_model(config).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            variant = {
                "name": f"A{hidden}",
                "model": config.to_dict(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "initial_parameters_sha256": model_fingerprint(model),
                "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 1e-4},
                "vision_loss_weight": 0.0,
                "counts": {"warmup": 0, "measured": 0, "profiled": 0},
            }
            if device.type == "cuda":
                synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            losses = []

            def update(index, model=model, optimizer=optimizer):
                with torch.profiler.record_function("diagnostic.batch_transfer"):
                    batch = _batch_to_device(batches[index], device)
                optimizer.zero_grad(set_to_none=True)
                with torch.profiler.record_function("diagnostic.forward_and_loss"):
                    output = model(batch, forecast_s=3 * scene.dt, batched_encoding=True)
                    loss, _ = training_loss(output, batch)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite diagnostic loss")
                with torch.profiler.record_function("diagnostic.backward"):
                    loss.backward()
                with torch.profiler.record_function("diagnostic.clip_and_step"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                losses.append(float(loss.detach()))

            offset = 0
            for phase, count in (("warmup", warmup_updates), ("measured", measured_updates)):
                seconds = []
                for index in range(offset, offset + count):
                    synchronize()
                    started = time.perf_counter()
                    update(index)
                    synchronize()
                    seconds.append(time.perf_counter() - started)
                    variant["counts"][phase] += 1
                variant[phase] = timing_summary(seconds)
                offset += count
            if device.type == "cuda":
                variant["unprofiled_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                variant["unprofiled_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                variant["unprofiled_cuda_peak_allocated_bytes"] = None
                variant["unprofiled_cuda_peak_reserved_bytes"] = None
            # Separate pass: CPU/CUDA operator measurements cannot contaminate wall timings above.
            with torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False) as prof:
                for index in range(offset, offset + profile_updates):
                    update(index)
                    prof.step()
                    variant["counts"]["profiled"] += 1
            variant["operators"] = operator_statistics(prof)
            require_device_activity(variant["operators"], device_name)
            variant["profiled_cuda_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
            completed = sum(variant["counts"].values())
            variant["counts"].update(
                optimizer_updates=completed,
                forward_calls=completed,
                batched_time_steps=completed * scene.steps,
                episode_time_steps=completed * batch_size * scene.steps,
                consumed_episodes=completed * batch_size,
            )
            variant["diagnostic_losses_not_accuracy_metrics"] = losses
            report["variants"].append(variant)
            write_json(outdir / "profile.json", report)
            del update, optimizer, model, prof
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["status"] = "completed"
        write_json(outdir / "profile.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(outdir / "profile.json", report)
        raise
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=8)
    parser.add_argument("--profile-updates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run_profile(
        args.out,
        device_name=args.device,
        warmup_updates=args.warmup_updates,
        measured_updates=args.measured_updates,
        profile_updates=args.profile_updates,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.out / "profile.json"),
                "variants": [v["name"] for v in report["variants"]],
            }
        )
    )


if __name__ == "__main__":
    main()
