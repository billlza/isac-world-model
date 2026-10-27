from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import read_checkpoint, save_checkpoint, write_json
from .model import EventBeliefModel, ModelConfig, training_loss
from .simulator import SceneConfig, recorded_episode
from .structured import StructuredConfig, build_model, model_config_from_dict
from .validation import finite, integer

INPUT_KEYS = ("image", "radio", "age", "valid", "new", "dt")


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


def batchify(episodes: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(np.stack([e[key] for e in episodes])).to(device) for key in episodes[0]}


def runtime_info(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
        "threads": torch.get_num_threads(),
    }


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
) -> dict:
    integer(epochs, "epochs", 1, 100_000)
    integer(episodes, "episodes", 1, 100_000)
    integer(batch_size, "batch_size", 1, 100_000)
    integer(seed, "seed", 0, 99)
    integer(patience, "patience", 0, 100_000)
    finite(learning_rate, "learning_rate", positive=True)
    integer(max_dataset_bytes, "max_dataset_bytes", 1)
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
        (2 * episodes + max(8, episodes // 8) + 2 * min(batch_size, episodes))
        * scene.steps
        * (3 * scene.image_size**2 + 32)
        * 4
    )
    if estimated > max_dataset_bytes:
        raise MemoryError(f"Dataset estimate {estimated} exceeds configured RAM budget {max_dataset_bytes}")
    # Disjoint episode seeds, never an adjacent-frame random split.
    train_seeds = [seed * 100000 + i for i in range(episodes)]
    val_seeds = [10_000_000 + seed * 100000 + i for i in range(max(8, episodes // 8))]
    data = [recorded_episode(s, scene) for s in train_seeds]
    val = [recorded_episode(s, scene) for s in val_seeds]
    if isinstance(model_config, StructuredConfig) and model_config.room_m != scene.room_m:
        raise ValueError("Structured model geometry must match scene room_m")
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    manifest = {
        "version": "0.4.0",
        "initialization": "random; no pretrained weights",
        "scene": scene.to_dict(),
        "model": model.config.to_dict(),
        "seed": seed,
        "epochs": epochs,
        "episodes": episodes,
        "batch_size": batch_size,
        "train_episode_seeds": train_seeds,
        "validation_episode_seeds": val_seeds,
        "parameters": sum(p.numel() for p in model.parameters()),
        "runtime": runtime_info(device),
        "hardware_validated": [str(device)],
    }
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
    start = time.perf_counter()
    best = float("inf")
    history = []
    begin, stale = 0, 0
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
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume configuration does not match saved run")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = json.loads(saved["numpy_rng_json"])
        torch.set_rng_state(saved["torch_rng"])
        begin, best, history = saved["completed_epochs"], saved["best_rmse"], saved["history"]
        if epochs <= begin:
            raise ValueError("--epochs is the total epoch target and must exceed completed epochs")
        stale = saved.get("stale_epochs", 0)
        # Preserve the true best checkpoint even if continuation never improves.
        save_checkpoint(outdir / "model.pt", saved["best_checkpoint"])
    else:
        saved = None
    best_checkpoint = None if saved is None else saved["best_checkpoint"]
    for epoch in range(begin, epochs):
        model.train()
        losses = []
        for indices in np.array_split(rng.permutation(episodes), int(np.ceil(episodes / batch_size))):
            selection = torch.as_tensor(indices, dtype=torch.long)
            batch = {k: v.index_select(0, selection).to(device) for k, v in train_tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, forecast_s=3 * scene.dt, batched_encoding=True)
            loss, _ = training_loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        sums, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                b = batchify(val[i : i + batch_size], device)
                pred = model(b, forecast_s=3 * scene.dt, batched_encoding=True)
                mse = (pred["mean"][:, 5:, :2] - b["state_target"][:, 5:, :2]).square().sum(-1)
                sums += float(mse.sum()) * scene.room_m**2
                count += mse.numel()
        rmse = float(np.sqrt(sums / count))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_position_rmse_m": rmse}
        history.append(row)
        print(json.dumps(row), flush=True)
        if not math.isfinite(rmse):
            raise FloatingPointError("Non-finite validation metric")
        if rmse < best:
            best = rmse
            stale = 0
            # CPU tensors + primitive metadata; loader uses weights_only=True.
            best_checkpoint = {
                "schema_version": 2,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "model_config": model.config.to_dict(),
                "scene_config": scene.to_dict(),
                "best_epoch": epoch + 1,
                "seed": seed,
                "train_episode_seeds": train_seeds,
                "validation_episode_seeds": val_seeds,
            }
            save_checkpoint(outdir / "model.pt", best_checkpoint)
        else:
            stale += 1
        save_checkpoint(
            outdir / "resume.pt",
            {
                "schema_version": 2,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "numpy_rng_json": json.dumps(rng.bit_generator.state),
                "torch_rng": torch.get_rng_state(),
                "completed_epochs": epoch + 1,
                "best_rmse": best,
                "best_checkpoint": best_checkpoint,
                "history": history,
                "stale_epochs": stale,
                "seed": seed,
                "episodes": episodes,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "scene_config": scene.to_dict(),
                "model_config": model.config.to_dict(),
            },
        )
        write_json(outdir / "history.json", history)
        if patience and stale >= patience:
            break
    manifest["completed_epochs"] = len(history)
    manifest["training_seconds"] = time.perf_counter() - start
    manifest["best_validation_rmse_m"] = best
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
