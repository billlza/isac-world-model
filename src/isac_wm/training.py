from __future__ import annotations
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from .model import EventBeliefModel, ModelConfig, training_loss
from .simulator import SceneConfig, recorded_episode


INPUT_KEYS = ("image", "radio", "age", "valid", "new", "dt")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
    return {key: torch.from_numpy(np.stack([e[key] for e in episodes])).to(device)
            for key in episodes[0]}


def runtime_info(device: torch.device) -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": str(np.__version__),
            "device": str(device), "threads": torch.get_num_threads()}


def train(outdir: Path, epochs: int = 10, episodes: int = 96, batch_size: int = 8,
          device_name: str = "auto", seed: int = 7, scene: SceneConfig | None = None,
          model_config: ModelConfig | None = None) -> dict:
    if min(epochs, episodes, batch_size) <= 0 or seed < 0:
        raise ValueError("Training counts must be positive; seed must be non-negative")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new run directory")
    outdir.mkdir(parents=True, exist_ok=True)
    scene = scene or SceneConfig()
    device = resolve_device(device_name)
    seed_all(seed)
    # Disjoint episode seeds, never an adjacent-frame random split.
    train_seeds = [seed*100000+i for i in range(episodes)]
    val_seeds = [10_000_000+seed*100000+i for i in range(max(8, episodes//8))]
    data = [recorded_episode(s, scene) for s in train_seeds]
    val = [recorded_episode(s, scene) for s in val_seeds]
    model = EventBeliefModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    manifest = {"version": "0.1.0", "initialization": "random; no pretrained weights",
                "scene": scene.to_dict(), "model": model.config.to_dict(),
                "seed": seed, "epochs": epochs, "episodes": episodes, "batch_size": batch_size,
                "train_episode_seeds": train_seeds, "validation_episode_seeds": val_seeds,
                "parameters": sum(p.numel() for p in model.parameters()),
                "runtime": runtime_info(device), "hardware_validated": [str(device)]}
    (outdir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    start = time.perf_counter()
    best = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in np.array_split(rng.permutation(episodes), int(np.ceil(episodes/batch_size))):
            batch = batchify([data[int(i)] for i in indices], device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, forecast_s=3*scene.dt)
            loss, _ = training_loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        sums, count = 0., 0
        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                b = batchify(val[i:i+batch_size], device)
                pred = model(b, forecast_s=3*scene.dt)
                mse = (pred["mean"][:, 5:, :2]-b["state_target"][:, 5:, :2]).square().sum(-1)
                sums += float(mse.sum())*scene.room_m**2
                count += mse.numel()
        rmse = float(np.sqrt(sums/count))
        row = {"epoch": epoch+1, "train_loss": float(np.mean(losses)), "validation_position_rmse_m": rmse}
        history.append(row)
        print(json.dumps(row), flush=True)
        if rmse < best:
            best = rmse
            # CPU tensors + primitive metadata; loader uses weights_only=True.
            torch.save({"state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()},
                        "model_config": model.config.to_dict(), "scene_config": scene.to_dict(),
                        "best_epoch": epoch+1, "seed": seed}, outdir/"model.pt")
    manifest["training_seconds"] = time.perf_counter()-start
    manifest["best_validation_rmse_m"] = best
    (outdir/"history.json").write_text(json.dumps(history, indent=2)+"\n")
    (outdir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def load_model(path: Path, device_name: str = "auto") -> tuple[EventBeliefModel, SceneConfig, torch.device]:
    device = resolve_device(device_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EventBeliefModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, SceneConfig(**checkpoint["scene_config"]), device
