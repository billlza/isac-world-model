from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import model_fingerprint, read_checkpoint, save_checkpoint, write_json
from .model import EventBeliefModel, ModelConfig, training_loss
from .simulator import SceneConfig, recorded_episode
from .structured import StructuredConfig, build_model, model_config_from_dict
from .validation import finite, integer
from .visual_training_diagnostics import (
    MAX_DIAGNOSTIC_EPISODES,
    MAX_DIAGNOSTIC_STEPS,
    VISION_SUPERVISION_PROTOCOL,
    begin_parameter_update,
    diagnostic_protocol,
    event_counts,
    finish_parameter_update,
    validate_schedule,
    validation_snapshot,
)

INPUT_KEYS = ("image", "radio", "age", "valid", "new", "dt")
PROVENANCE_KEYS = ("vision_capture_s", "vision_event_uid")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Stage a CPU batch while retaining exact provenance-only metadata on CPU.

    Capture timestamps are float64 for provenance, which MPS cannot represent.
    Neither timestamp nor event UID participates in model inputs or losses.
    """
    return {key: value if key in PROVENANCE_KEYS else value.to(device) for key, value in batch.items()}


def batchify(episodes: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    cpu_batch = {key: torch.from_numpy(np.stack([e[key] for e in episodes])) for key in episodes[0]}
    return _batch_to_device(cpu_batch, device)


def runtime_info(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
        "threads": torch.get_num_threads(),
    }


def _validation_position_rmse(model, val, batch_size, device, scene) -> float:
    """Evaluate a fixed corpus without adding any optimizer steps."""
    model.eval()
    sums, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val), batch_size):
            batch = batchify(val[i : i + batch_size], device)
            pred = model(batch, forecast_s=3 * scene.dt, batched_encoding=True)
            mse = (pred["mean"][:, 5:, :2] - batch["state_target"][:, 5:, :2]).square().sum(-1)
            sums += float(mse.sum()) * scene.room_m**2
            count += mse.numel()
    rmse = float(np.sqrt(sums / count))
    if not math.isfinite(rmse):
        raise FloatingPointError("Non-finite validation metric")
    return rmse


def train(
    outdir: Path,
    epochs: int = 10,
    episodes: int = 96,
    batch_size: int = 8,
    device_name: str = "auto",
    seed: int = 7,
    scene: SceneConfig | None = None,
    model_config: ModelConfig | StructuredConfig | None = None,
    *,
    resume: Path | None = None,
    learning_rate: float = 1e-3,
    patience: int = 0,
    max_dataset_bytes: int = 4 * 1024**3,
    train_seed_base: int | None = None,
    validation_seed_base: int | None = None,
    batch_order_seed: int | None = None,
    validation_episodes: int | None = None,
    validation_every_updates: int | None = None,
    vision_loss_weight: float = 0.0,
    diagnostic_updates: tuple[int, ...] | None = None,
) -> dict:
    """Train with epoch-end validation, or an explicit optimizer-update schedule.

    ``epochs`` is the total target, including epochs restored from ``resume``.
    An update schedule never adds an unscheduled final validation. Checkpoints
    for continuation are saved at completed epoch boundaries. A short controlled
    run may therefore have a resume checkpoint before any model is selected;
    its manifest explicitly records ``no_validation_yet`` and a null best score.
    Fixed-update studies disable early stopping so every cell gets its full budget.
    """
    integer(epochs, "epochs", 1, 100_000)
    integer(episodes, "episodes", 1, 100_000)
    integer(batch_size, "batch_size", 1, 100_000)
    integer(seed, "seed", 0, 99)
    integer(patience, "patience", 0, 100_000)
    finite(learning_rate, "learning_rate", positive=True)
    vision_loss_weight = finite(vision_loss_weight, "vision_loss_weight")
    diagnostic_updates = validate_schedule(diagnostic_updates)
    if (vision_loss_weight or diagnostic_updates is not None) and not isinstance(
        model_config, StructuredConfig
    ):
        raise ValueError("Visual supervision and diagnostics require a structured spatial encoder")
    integer(max_dataset_bytes, "max_dataset_bytes", 1)
    n_val = (
        max(8, episodes // 8)
        if validation_episodes is None
        else integer(validation_episodes, "validation_episodes", 1, 100_000)
    )
    if validation_every_updates is not None:
        validation_every_updates = integer(validation_every_updates, "validation_every_updates", 1)
        if patience:
            raise ValueError("Fixed optimizer-update validation requires patience=0")
    updates_per_epoch = math.ceil(episodes / batch_size)
    if min(epochs, episodes, batch_size) <= 0 or seed < 0:
        raise ValueError("Training counts must be positive; seed must be non-negative")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new run directory")
    outdir.mkdir(parents=True, exist_ok=True)
    scene = scene or SceneConfig()
    device = resolve_device(device_name)
    seed_all(seed)
    # Account for the transient second copy while stacking and staging batches.
    # This budgets dataset arrays, not model/optimizer memory or process RSS.
    estimated = int(
        (2 * episodes + n_val + 2 * min(batch_size, episodes))
        * scene.steps
        * (3 * scene.image_size**2 + 40)
        * 4
    )
    if estimated > max_dataset_bytes:
        raise MemoryError(f"Dataset estimate {estimated} exceeds configured RAM budget {max_dataset_bytes}")
    # Disjoint episode seeds, never an adjacent-frame random split.
    custom_seeds = any(x is not None for x in (train_seed_base, validation_seed_base, batch_order_seed))
    data_base = (
        seed * 100000
        if train_seed_base is None
        else integer(train_seed_base, "train_seed_base", 0, 2**32 - episodes)
    )
    val_base = (
        10_000_000 + seed * 100000
        if validation_seed_base is None
        else integer(validation_seed_base, "validation_seed_base", 0, 2**32 - n_val)
    )
    order_seed = (
        seed if batch_order_seed is None else integer(batch_order_seed, "batch_order_seed", 0, 2**32 - 1)
    )
    train_seeds = list(range(data_base, data_base + episodes))
    val_seeds = list(range(val_base, val_base + n_val))
    if set(train_seeds).intersection(val_seeds):
        raise ValueError("Training and validation episode seeds overlap")
    explicit = {}
    if custom_seeds:
        explicit["seed_plan"] = {
            "initialization": seed,
            "train_seed_base": data_base,
            "validation_seed_base": val_base,
            "batch_order_seed": order_seed,
        }
    data = [recorded_episode(s, scene) for s in train_seeds]
    val = [recorded_episode(s, scene) for s in val_seeds]
    if isinstance(model_config, StructuredConfig) and model_config.room_m != scene.room_m:
        raise ValueError("Structured model geometry must match scene room_m")
    model = build_model(model_config).to(device)
    observed_protocol = diagnostic_protocol(diagnostic_updates, val_seeds, scene.steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(order_seed)
    manifest = {
        **explicit,
        "version": "0.4.0",
        "initialization": "random; no pretrained weights",
        "scene": scene.to_dict(),
        "model": model.config.to_dict(),
        "seed": seed,
        "epochs": epochs,
        "episodes": episodes,
        "batch_size": batch_size,
        "validation_episodes": n_val,
        "validation_every_updates": validation_every_updates,
        "validation_mode": "epoch_end" if validation_every_updates is None else "fixed_optimizer_updates",
        "updates_per_epoch": updates_per_epoch,
        "planned_optimizer_updates": epochs * updates_per_epoch,
        "planned_validation_opportunities": (
            epochs
            if validation_every_updates is None
            else epochs * updates_per_epoch // validation_every_updates
        ),
        "history_train_loss_scope": "optimizer steps since previous validation",
        "train_episode_seeds": train_seeds,
        "validation_episode_seeds": val_seeds,
        "parameters": sum(p.numel() for p in model.parameters()),
        "runtime": runtime_info(device),
        "hardware_validated": [str(device)],
        "vision_loss_weight": vision_loss_weight,
        "vision_supervision_protocol": dict(VISION_SUPERVISION_PROTOCOL),
        "visual_encoder_parameters": sum(p.numel() for p in model.visual.parameters()),
        "initial_visual_parameters_sha256": model_fingerprint(model.visual),
        "diagnostic_protocol": observed_protocol,
    }
    if custom_seeds:
        manifest["initial_parameters_sha256"] = model_fingerprint(model)
    manifest.update(
        {
            "learning_rate": learning_rate,
            "early_stopping_patience": patience,
            "dataset_estimated_bytes": estimated,
            "resume_source": str(resume) if resume else None,
        }
    )
    write_json(outdir / "manifest.json", manifest)
    # Cache stacking once; batches remain on CPU until transferred.
    train_tensors = batchify(data, torch.device("cpu"))
    del data
    counts_per_epoch = event_counts(train_tensors)
    counts_seen = {key: 0 for key in counts_per_epoch}
    manifest["training_event_counts_per_epoch"] = counts_per_epoch
    diagnostic_batch = None
    diagnostic_history = []
    diagnostic_overhead_seconds = 0.0
    if diagnostic_updates is not None:
        diagnostic_batch = batchify(val[:MAX_DIAGNOSTIC_EPISODES], device)
        diagnostic_batch = {key: value[:, :MAX_DIAGNOSTIC_STEPS] for key, value in diagnostic_batch.items()}
    best = float("inf")
    history = []
    epoch_history = []
    begin, stale = 0, 0
    optimizer_updates = 0
    pending_loss_sum, pending_loss_count = 0.0, 0
    if resume is not None:
        saved = read_checkpoint(resume)
        expected = {
            "seed": seed,
            "episodes": episodes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "scene_config": scene.to_dict(),
            "model_config": model.config.to_dict(),
        }
        if saved.get("seed_plan") != explicit.get("seed_plan"):
            raise ValueError("Resume seed plan does not match saved run")
        # Optional defaults omitted by old checkpoints and explicitly serialized
        # defaults describe the same model. Parse first so unknown fields/modes
        # cannot disappear during the comparison.
        try:
            saved_model_config = model_config_from_dict(saved["model_config"]).to_dict()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Resume model configuration is invalid") from error
        comparison = saved | {"model_config": saved_model_config}
        if any(comparison.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume configuration does not match saved run")
        if (
            saved.get("validation_episodes", max(8, episodes // 8)) != n_val
            or saved.get("validation_every_updates") != validation_every_updates
        ):
            raise ValueError("Resume validation protocol does not match saved run")
        if saved.get("vision_loss_weight", 0.0) != vision_loss_weight:
            raise ValueError("Resume visual-supervision weight does not match saved run")
        saved_visual_protocol = saved.get("vision_supervision_protocol")
        if saved_visual_protocol is not None and saved_visual_protocol != VISION_SUPERVISION_PROTOCOL:
            raise ValueError("Resume visual-supervision protocol does not match saved run")
        if vision_loss_weight and saved_visual_protocol is None:
            raise ValueError("Resume visual-supervision protocol is missing")
        if saved.get("diagnostic_protocol") != observed_protocol:
            raise ValueError("Resume diagnostic protocol does not match saved run")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = json.loads(saved["numpy_rng_json"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and "torch_cuda_rng" in saved:
            torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
        begin, best, history = saved["completed_epochs"], saved["best_rmse"], saved["history"]
        integer(begin, "saved completed_epochs", 1, 100_000)
        if epochs <= begin:
            raise ValueError("--epochs is the total epoch target and must exceed completed epochs")
        optimizer_updates = saved.get("optimizer_updates", begin * updates_per_epoch)
        if optimizer_updates != begin * updates_per_epoch:
            raise ValueError("Resume update count is not a completed epoch boundary")
        counts_seen = saved.get(
            "event_counts_seen", {key: value * begin for key, value in counts_per_epoch.items()}
        )
        if counts_seen != {key: value * begin for key, value in counts_per_epoch.items()}:
            raise ValueError("Resume delivered-evidence counts do not match completed epochs")
        diagnostic_history = saved.get("visual_diagnostics_snapshots", [])
        expected_observations = (
            [] if diagnostic_updates is None else [u for u in diagnostic_updates if u <= optimizer_updates]
        )
        if [row.get("optimizer_updates") for row in diagnostic_history] != expected_observations:
            raise ValueError("Resume diagnostic history does not match the requested schedule")
        if validation_every_updates is None:
            valid_history = len(history) == begin and all(
                row["epoch"] == i + 1 for i, row in enumerate(history)
            )
        else:
            valid_history = len(history) == optimizer_updates // validation_every_updates and all(
                row["optimizer_updates"] == (i + 1) * validation_every_updates
                for i, row in enumerate(history)
            )
        if not valid_history:
            raise ValueError("Resume validation history does not match the update schedule")
        epoch_history = saved.get(
            "epoch_history",
            [
                {
                    "epoch": row["epoch"],
                    "optimizer_updates": row["epoch"] * updates_per_epoch,
                    "train_loss": row["train_loss"],
                }
                for row in history
            ],
        )
        if len(epoch_history) != begin:
            raise ValueError("Resume epoch history does not match completed epochs")
        pending_loss_sum = saved.get("pending_validation_loss_sum", 0.0)
        pending_loss_count = saved.get("pending_validation_loss_count", 0)
        expected_pending = (
            0 if validation_every_updates is None else optimizer_updates % validation_every_updates
        )
        if pending_loss_count != expected_pending or not math.isfinite(pending_loss_sum):
            raise ValueError("Resume pending validation interval does not match the update schedule")
        stale = saved.get("stale_epochs", 0)
        # Preserve the true best checkpoint even if continuation never improves.
        if saved["best_checkpoint"] is not None:
            save_checkpoint(outdir / "model.pt", saved["best_checkpoint"])
        elif history:
            raise ValueError("Resume has validation history but no selected checkpoint")
    else:
        saved = None
    best_checkpoint = None if saved is None else saved["best_checkpoint"]

    def record_diagnostic(update, training_details=None):
        nonlocal diagnostic_overhead_seconds
        observed = validation_snapshot(
            model, diagnostic_batch, room_m=scene.room_m, dt=scene.dt, vision_loss_weight=vision_loss_weight
        )
        diagnostic_overhead_seconds += observed["observer_seconds"]
        diagnostic_history.append(
            {
                "optimizer_updates": update,
                "validation": observed,
                "training": training_details,
                "event_counts_seen": dict(counts_seen),
            }
        )
        write_json(
            outdir / "visual_diagnostics.json",
            {
                "schema_version": 1,
                "protocol": observed_protocol,
                "snapshots": diagnostic_history,
            },
        )

    if diagnostic_updates is not None:
        if saved is None and 0 in diagnostic_updates:
            record_diagnostic(0)
        else:
            write_json(
                outdir / "visual_diagnostics.json",
                {
                    "schema_version": 1,
                    "protocol": observed_protocol,
                    "snapshots": diagnostic_history,
                },
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        # Release only unused cache from preceding cells; live model/optimizer
        # tensors remain in the measured baseline. Do not claim process RSS.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    initial_updates = optimizer_updates
    completed_epochs = begin
    for epoch in range(begin, epochs):
        model.train()
        losses = []
        for batch_index, indices in enumerate(np.array_split(rng.permutation(episodes), updates_per_epoch)):
            selection = torch.as_tensor(indices, dtype=torch.long)
            cpu_batch = {k: v.index_select(0, selection) for k, v in train_tensors.items()}
            batch = _batch_to_device(cpu_batch, device)
            observed_step = diagnostic_updates is not None and optimizer_updates + 1 in diagnostic_updates
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch, forecast_s=3 * scene.dt, batched_encoding=True, return_visual=bool(vision_loss_weight)
            )
            loss, loss_details = training_loss(output, batch, vision_loss_weight=vision_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            parameter_before = None
            if observed_step:
                observer_started = time.perf_counter()
                parameter_before = begin_parameter_update(model)
                diagnostic_overhead_seconds += time.perf_counter() - observer_started
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            parameter_updates = None
            if observed_step:
                observer_started = time.perf_counter()
                parameter_updates = finish_parameter_update(model, parameter_before)
                diagnostic_overhead_seconds += time.perf_counter() - observer_started
            loss_value = float(loss.detach())
            losses.append(loss_value)
            optimizer_updates += 1
            batch_counts = event_counts(cpu_batch)
            for key, value in batch_counts.items():
                counts_seen[key] += value
            if observed_step:
                record_diagnostic(
                    optimizer_updates,
                    {
                        "losses": loss_details,
                        "branches": parameter_updates,
                        "event_counts": batch_counts,
                    },
                )
            pending_loss_sum += loss_value
            pending_loss_count += 1
            epoch_end = batch_index + 1 == updates_per_epoch
            should_validate = (
                epoch_end
                if validation_every_updates is None
                else optimizer_updates % validation_every_updates == 0
            )
            if should_validate:
                rmse = _validation_position_rmse(model, val, batch_size, device, scene)
                row = {
                    "epoch": epoch + 1,
                    # Keep the legacy epoch history numerically unchanged.
                    "train_loss": (
                        float(np.mean(losses))
                        if validation_every_updates is None
                        else pending_loss_sum / pending_loss_count
                    ),
                    "validation_position_rmse_m": rmse,
                }
                if validation_every_updates is not None:
                    row.update(optimizer_updates=optimizer_updates, completed_epochs=epoch + int(epoch_end))
                history.append(row)
                pending_loss_sum, pending_loss_count = 0.0, 0
                print(json.dumps(row), flush=True)
                if rmse < best:
                    best = rmse
                    stale = 0
                    # CPU tensors + primitive metadata; loader uses weights_only=True.
                    best_checkpoint = {
                        **explicit,
                        "schema_version": 2,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "model_config": model.config.to_dict(),
                        "scene_config": scene.to_dict(),
                        "best_epoch": epoch + 1,
                        "best_update": optimizer_updates,
                        "seed": seed,
                        "train_episode_seeds": train_seeds,
                        "validation_episode_seeds": val_seeds,
                        "validation_episodes": n_val,
                        "validation_every_updates": validation_every_updates,
                        "vision_loss_weight": vision_loss_weight,
                        "vision_supervision_protocol": dict(VISION_SUPERVISION_PROTOCOL),
                        "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
                        "diagnostic_protocol": observed_protocol,
                    }
                    save_checkpoint(outdir / "model.pt", best_checkpoint)
                else:
                    stale += 1
                model.train()
        completed_epochs = epoch + 1
        epoch_history.append(
            {
                "epoch": completed_epochs,
                "optimizer_updates": optimizer_updates,
                "train_loss": float(np.mean(losses)),
            }
        )
        save_checkpoint(
            outdir / "resume.pt",
            {
                **explicit,
                "schema_version": 2,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "numpy_rng_json": json.dumps(rng.bit_generator.state),
                "torch_rng": torch.get_rng_state(),
                **({"torch_cuda_rng": torch.cuda.get_rng_state_all()} if device.type == "cuda" else {}),
                "completed_epochs": completed_epochs,
                "optimizer_updates": optimizer_updates,
                "best_rmse": best,
                "best_checkpoint": best_checkpoint,
                "history": history,
                "epoch_history": epoch_history,
                "pending_validation_loss_sum": pending_loss_sum,
                "pending_validation_loss_count": pending_loss_count,
                "stale_epochs": stale,
                "seed": seed,
                "episodes": episodes,
                "batch_size": batch_size,
                "validation_episodes": n_val,
                "validation_every_updates": validation_every_updates,
                "learning_rate": learning_rate,
                "scene_config": scene.to_dict(),
                "model_config": model.config.to_dict(),
                "vision_loss_weight": vision_loss_weight,
                "vision_supervision_protocol": dict(VISION_SUPERVISION_PROTOCOL),
                "initial_visual_parameters_sha256": manifest["initial_visual_parameters_sha256"],
                "diagnostic_protocol": observed_protocol,
                "visual_diagnostics_snapshots": diagnostic_history,
                "event_counts_seen": counts_seen,
            },
        )
        write_json(outdir / "history.json", history)
        write_json(outdir / "epoch_history.json", epoch_history)
        if patience and stale >= patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        manifest["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        manifest["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        manifest["cuda_memory_scope"] = (
            "This call's training/validation/checkpoint loop, including live model/optimizer at start; "
            "unused allocator cache released before resetting peaks"
        )
    manifest["completed_epochs"] = completed_epochs
    manifest["completed_epochs_this_call"] = completed_epochs - begin
    manifest["optimizer_updates"] = optimizer_updates
    manifest["optimizer_updates_this_call"] = optimizer_updates - initial_updates
    manifest["validation_updates"] = [
        row.get("optimizer_updates", row["epoch"] * updates_per_epoch) for row in history
    ]
    manifest["validation_opportunities"] = len(history)
    manifest["best_update"] = (
        None
        if best_checkpoint is None
        else best_checkpoint.get("best_update", best_checkpoint["best_epoch"] * updates_per_epoch)
    )
    manifest["model_selection_status"] = "no_validation_yet" if best_checkpoint is None else "selected"
    manifest["training_seconds"] = time.perf_counter() - start
    manifest["training_timing_scope"] = "This call's training/validation/checkpoint loop; CUDA synchronized"
    if diagnostic_updates is not None:
        manifest["training_timing_scope"] += "; includes scheduled diagnostic work after update zero"
    manifest["diagnostic_overhead_seconds"] = diagnostic_overhead_seconds
    manifest["event_counts_seen"] = counts_seen
    manifest["diagnostic_observed_updates"] = [row["optimizer_updates"] for row in diagnostic_history]
    manifest["best_validation_rmse_m"] = None if best_checkpoint is None else best
    write_json(outdir / "history.json", history)
    write_json(outdir / "manifest.json", manifest)
    return manifest


def load_model(path: Path, device_name: str = "auto") -> tuple[EventBeliefModel, SceneConfig, torch.device]:
    device = resolve_device(device_name)
    checkpoint = read_checkpoint(path)
    if not {"state_dict", "model_config", "scene_config"}.issubset(checkpoint):
        raise ValueError("Missing checkpoint fields")
    if not isinstance(checkpoint["state_dict"], dict) or not all(
        isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
        for v in checkpoint["state_dict"].values()
    ):
        raise ValueError("Checkpoint contains invalid/non-finite parameters")
    model = build_model(model_config_from_dict(checkpoint["model_config"]))
    if (
        isinstance(model.config, StructuredConfig)
        and model.config.room_m != checkpoint["scene_config"]["room_m"]
    ):
        raise ValueError("Checkpoint model and scene geometry do not match")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    model.provenance_seeds = set(checkpoint.get("train_episode_seeds", [])) | set(
        checkpoint.get("validation_episode_seeds", [])
    )
    return model, SceneConfig(**checkpoint["scene_config"]), device
