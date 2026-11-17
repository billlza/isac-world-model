"""Controlled, random-initialization pretraining on mixed synthetic conditions.

This is a state/observation world-model experiment. It does not train a language
decoder, implement RLHF, or turn synthetic validation into a production claim.
The three recipes separate the data intervention from the combined architecture
and auxiliary-objective intervention. All completion states are written only
after the corresponding computation and artifact writes actually succeed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    model_fingerprint,
    read_checkpoint,
    read_resume_checkpoint,
    save_checkpoint,
    save_resume_checkpoint,
    sha256_file,
    write_json,
)
from .optimizer_policy import make_optimizer
from .pretraining_data import make_episode, scene_profiles
from .pretraining_objective import pretraining_loss
from .simulator import SceneConfig
from .structured import StructuredConfig, build_model
from .training import INPUT_KEYS, batchify, resolve_device, runtime_info, seed_all
from .validation import finite, integer

RECIPES = ("nominal-control", "mixed-control", "mixed-enhanced")
PHASE_D_BASE = "30fe45f5837b6ed09fe739aa2371166a89b43fb9"


@dataclass(frozen=True)
class PretrainingConfig:
    profile: str = "development"
    episodes: int = 96
    epochs: int = 8
    batch_size: int = 8
    steps: int = 40
    hidden_dim: int = 96
    validation_per_profile: int = 4
    test_per_profile: int = 8
    initialization_seeds: tuple[int, ...] = (7,)
    horizons: tuple[int, ...] = (3, 10, 20)
    learning_rate: float = 1e-3
    batch_order_seed: int = 117

    def __post_init__(self):
        if self.profile not in {"smoke", "development", "formal"}:
            raise ValueError("Unknown pretraining profile")
        for name, upper in (("episodes", 100_000), ("epochs", 10_000), ("batch_size", 1024)):
            integer(getattr(self, name), name, 1, upper)
        integer(self.steps, "steps", 8, 512)
        integer(self.hidden_dim, "hidden_dim", 16, 2048)
        if self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four attention heads")
        integer(self.validation_per_profile, "validation_per_profile", 1, 9999)
        integer(self.test_per_profile, "test_per_profile", 1, 9999)
        integer(self.batch_order_seed, "batch_order_seed", 0, 2**32 - 1)
        finite(self.learning_rate, "learning_rate", positive=True)
        if not self.initialization_seeds or len(set(self.initialization_seeds)) != len(
            self.initialization_seeds
        ):
            raise ValueError("Initialization seeds must be nonempty and unique")
        for seed in self.initialization_seeds:
            integer(seed, "initialization seed", 0, 99)
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Forecast horizons must be nonempty and unique")
        for horizon in self.horizons:
            integer(horizon, "forecast horizon", 1, self.steps - 6)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


def profile_config(name: str) -> PretrainingConfig:
    if name == "smoke":
        return PretrainingConfig(
            profile=name,
            episodes=4,
            epochs=1,
            batch_size=2,
            steps=12,
            hidden_dim=32,
            validation_per_profile=1,
            test_per_profile=1,
            horizons=(3, 6),
        )
    if name == "development":
        return PretrainingConfig()
    if name == "formal":
        return PretrainingConfig(
            profile=name,
            episodes=3072,
            epochs=8,
            batch_size=16,
            steps=64,
            hidden_dim=512,
            validation_per_profile=16,
            test_per_profile=32,
            initialization_seeds=(7, 11),
            learning_rate=3e-4,
        )
    raise ValueError("Unknown pretraining profile")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_provenance(root: Path) -> dict:
    """Identify actual bytes, including an explicitly dirty development checkout."""
    root = Path(root).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
    ).splitlines()
    paths = sorted((root / "src/isac_wm").glob("*.py")) + [root / "scripts/complex_scene_pretraining.py"]
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    return {
        "git_commit": commit,
        "dirty": bool(dirty),
        "dirty_paths": dirty,
        "implementation_sha256": hashlib.sha256(_canonical(hashes).encode()).hexdigest(),
        "implementation_files_sha256": hashes,
        "historical_base_commit": PHASE_D_BASE,
    }


def episode_plan(config: PretrainingConfig, recipe: str, split: str) -> list[dict]:
    """Disjoint, explicit seed namespaces; model initialization never changes data."""
    if recipe not in RECIPES or split not in {"train", "validation", "test"}:
        raise ValueError("Unknown recipe or split")
    # Old calibration/evaluation use 20M/30M. Keep those holdouts untouched.
    namespace = {"smoke": 610_000_000, "development": 620_000_000, "formal": 630_000_000}[config.profile]
    names = list(scene_profiles(SceneConfig(steps=config.steps)))
    if split == "train":
        return [
            {
                "seed": namespace + i,
                "profile": "nominal" if recipe == "nominal-control" else names[i % len(names)],
                "appearance": recipe != "nominal-control" and names[i % len(names)] != "nominal",
            }
            for i in range(config.episodes)
        ]
    offset = 1_000_000 if split == "validation" else 2_000_000
    count = config.validation_per_profile if split == "validation" else config.test_per_profile
    return [
        {"seed": namespace + offset + k * 10_000 + j, "profile": name, "appearance": name != "nominal"}
        for k, name in enumerate(names)
        for j in range(count)
    ]


def model_config(config: PretrainingConfig, recipe: str) -> StructuredConfig:
    if recipe not in RECIPES:
        raise ValueError("Unknown recipe")
    enhanced = recipe == "mixed-enhanced"
    return StructuredConfig(
        hidden_dim=config.hidden_dim,
        latent_state_norm="layernorm",
        visual_channels=64 if enhanced else 16,
        visual_depth=3 if enhanced else 1,
        max_acceleration_mps2=6.0 if enhanced else 0.5,
        # Keep initial physical increments small while allowing larger learned
        # accelerations. The first development trial exposed the 12x jump from
        # increasing the output bound without scaling the random output head.
        acceleration_init_scale=1.0 / 12.0 if enhanced else 1.0,
    )


def _corpus(plan: list[dict], profiles: dict[str, SceneConfig]) -> list[dict[str, np.ndarray]]:
    return [make_episode(row["seed"], profiles[row["profile"]], appearance=row["appearance"]) for row in plan]


def _prediction(model, batch, horizons: tuple[int, ...], dt: float) -> dict:
    """Evaluate all requested horizons from the same causal filtered states."""
    b, steps = batch["image"].shape[:2]
    flat = {
        key: batch[key].reshape(b * steps, *batch[key].shape[2:])
        for key in ("image", "radio", "new", "valid")
    }
    encodings = tuple(value.reshape(b, steps, -1) for value in model.encode_new(flat))
    memory = model.initial(b, batch["image"].device)
    means, logvars = [], []
    futures = {h: [] for h in horizons}
    for tick in range(steps):
        output, memory = model.step(
            {key: batch[key][:, tick] for key in INPUT_KEYS},
            memory,
            tuple(value[:, tick] for value in encodings),
        )
        means.append(output["mean"])
        logvars.append(output["logvar"])
        for h in horizons:
            futures[h].append(model.forecast(output["hidden"], h * dt)["mean"])
    return {
        "mean": torch.stack(means, 1),
        "logvar": torch.stack(logvars, 1),
        "future": {h: torch.stack(value, 1) for h, value in futures.items()},
    }


def evaluate(
    model,
    plan: list[dict],
    profiles: dict[str, SceneConfig],
    batch_size: int,
    device: torch.device,
    horizons: tuple[int, ...] = (),
) -> dict:
    """Unweighted macro RMSE across profiles; no test data selects checkpoints."""
    model.eval()
    result = {}
    with torch.inference_mode():
        for name, scene in profiles.items():
            selected = [row for row in plan if row["profile"] == name]
            if not selected:
                raise ValueError(f"Evaluation plan lacks profile {name}")
            errors, coordinate_coverages = [], []
            future_errors = {h: [] for h in horizons}
            for start in range(0, len(selected), batch_size):
                batch = batchify(_corpus(selected[start : start + batch_size], profiles), device)
                prediction = _prediction(model, batch, horizons, scene.dt)
                residual = prediction["mean"][:, 5:, :2] - batch["state_target"][:, 5:, :2]
                errors.append((residual.norm(dim=-1) * scene.room_m).cpu().numpy().reshape(-1))
                std = prediction["logvar"][:, 5:, :2].mul(0.5).exp()
                coordinate_coverages.append(
                    (residual.abs() <= 1.6448536269514722 * std).cpu().numpy().reshape(-1)
                )
                for h in horizons:
                    residual_h = prediction["future"][h][:, 5:-h, :2] - batch["state_target"][:, 5 + h :, :2]
                    future_errors[h].append(
                        (residual_h.norm(dim=-1) * scene.room_m).cpu().numpy().reshape(-1)
                    )
            error = np.concatenate(errors).astype(np.float64)
            if not error.size or not np.isfinite(error).all():
                raise FloatingPointError("Non-finite or empty evaluation errors")
            result[name] = {
                "episodes": len(selected),
                "position_samples": int(error.size),
                "position_rmse_m": float(np.sqrt(np.mean(error**2))),
                "position_p95_m": float(np.quantile(error, 0.95)),
                "coordinate_90pct_coverage": float(np.concatenate(coordinate_coverages).mean()),
                "forecast_rmse_m": {
                    str(h): float(np.sqrt(np.mean(np.concatenate(values).astype(np.float64) ** 2)))
                    for h, values in future_errors.items()
                },
            }
    macro = float(np.mean([row["position_rmse_m"] for row in result.values()]))
    if not math.isfinite(macro):
        raise FloatingPointError("Non-finite macro evaluation metric")
    return {
        "macro_position_rmse_m": macro,
        "profiles": result,
        "warmup_steps_excluded": 5,
        "forecast_horizons_steps": list(horizons),
        "coverage_scope": "Marginal coordinate coverage; empirical, not a safety certificate",
    }


def _checkpoint(model, scene, protocol, seed, epoch, update, plans) -> dict:
    return {
        "schema_version": 2,
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "model_config": model.config.to_dict(),
        "scene_config": scene.to_dict(),
        "pretraining_protocol": protocol,
        "seed": seed,
        "best_epoch": epoch,
        "best_update": update,
        "train_episode_seeds": [row["seed"] for row in plans["train"]],
        "validation_episode_seeds": [row["seed"] for row in plans["validation"]],
    }


def _file_hashes(study_dir: Path, cell_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(study_dir)): sha256_file(path)
        for path in sorted(cell_dir.iterdir())
        if path.is_file()
    }


def verify_cell_artifacts(study_dir: Path, cell: dict) -> None:
    if cell.get("status") != "completed" or not cell.get("artifacts_sha256"):
        raise ValueError("Cell has not completed with verified artifacts")
    root = Path(study_dir).resolve()
    key = cell.get("path")
    if (
        not isinstance(key, str)
        or Path(key).parts != (key,)
        or cell.get("recipe") not in RECIPES
        or key != f"{cell['recipe']}-seed{cell.get('seed')}"
    ):
        raise ValueError("Invalid completed cell identity")
    required = {
        f"{key}/{name}"
        for name in (
            "manifest.json",
            "history.json",
            "metrics.json",
            "episode_plan.json",
            "model.pt",
            "model.pt.sha256",
            "final.pt",
            "final.pt.sha256",
            "resume.pt",
            "resume.pt.sha256",
        )
    }
    if set(cell["artifacts_sha256"]) != required:
        raise ValueError("Completed cell is missing required artifacts or contains unregistered files")
    for name in ("manifest", "history", "metrics"):
        if cell.get(name) != f"{key}/{name}.json":
            raise ValueError("Completed cell metadata paths differ from its identity")
    for relative, expected in cell["artifacts_sha256"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Cell artifact must be a regular file inside the study")
        if sha256_file(path) != expected:
            raise ValueError(f"Cell artifact checksum mismatch: {relative}")
    manifest, history, metrics = (
        json.loads((root / cell[name]).read_text()) for name in ("manifest", "history", "metrics")
    )
    protocol = manifest["protocol"]
    config = protocol["config"]
    epochs = config["epochs"]
    updates_per_epoch = math.ceil(config["episodes"] / config["batch_size"])
    if (
        manifest["status"] != "completed"
        or manifest["recipe"] != cell["recipe"]
        or manifest["seed"] != cell["seed"]
        or manifest["completed_epochs"] != epochs
        or manifest["optimizer_updates"] != epochs * updates_per_epoch
        or [row["epoch"] for row in history] != list(range(1, epochs + 1))
        or [row["optimizer_updates"] for row in history]
        != [e * updates_per_epoch for e in range(1, epochs + 1)]
    ):
        raise ValueError("Completed cell has not fulfilled its identity or training budget")
    selected_epoch = min(history, key=lambda row: row["validation"]["macro_position_rmse_m"])["epoch"]
    if (
        metrics["selected_epoch"] != selected_epoch
        or manifest["selected_epoch"] != selected_epoch
        or metrics["selected_update"] != selected_epoch * updates_per_epoch
        or metrics["selected_validation"] != history[selected_epoch - 1]["validation"]
        or metrics["selected_test"]["macro_position_rmse_m"] != cell["test_macro_position_rmse_m"]
        or metrics["selected_checkpoint_sha256"] != cell["artifacts_sha256"][f"{key}/model.pt"]
    ):
        raise ValueError("Selected checkpoint and reported metrics disagree")
    for checkpoint_name in ("model.pt", "final.pt", "resume.pt"):
        expected = cell["artifacts_sha256"][f"{key}/{checkpoint_name}"]
        if (root / key / f"{checkpoint_name}.sha256").read_text().strip() != expected:
            raise ValueError("Checkpoint sidecar disagrees with the result manifest")
    selected = read_checkpoint(root / key / "model.pt")
    final = read_checkpoint(root / key / "final.pt")
    continuation = read_resume_checkpoint(root / key / "resume.pt")
    if (
        selected["best_epoch"] != selected_epoch
        or selected["best_update"] != metrics["selected_update"]
        or final["best_epoch"] != epochs
        or final["best_update"] != epochs * updates_per_epoch
        or continuation["completed_epochs"] != epochs
        or continuation["history"] != history
        or any(item["pretraining_protocol"] != protocol for item in (selected, final, continuation))
    ):
        raise ValueError("Checkpoint contents disagree with the completed protocol")


def train_cell(
    study_dir: Path,
    config: PretrainingConfig,
    recipe: str,
    seed: int,
    device: torch.device,
    source: dict,
    *,
    resume: bool = False,
) -> dict:
    cell_key = f"{recipe}-seed{seed}"
    out = Path(study_dir) / cell_key
    if out.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing cell: {out}")
    out.mkdir(parents=True, exist_ok=True)
    scene = SceneConfig(steps=config.steps)
    profiles = scene_profiles(scene)
    plans = {split: episode_plan(config, recipe, split) for split in ("train", "validation", "test")}
    plan_sha = hashlib.sha256(_canonical(plans).encode()).hexdigest()
    protocol = {
        "schema_version": 1,
        "config": config.to_dict(),
        "recipe": recipe,
        "initialization_seed": seed,
        "episode_plan_sha256": plan_sha,
        "implementation_sha256": source["implementation_sha256"],
        "model_config": model_config(config, recipe).to_dict(),
        "scene_profiles": {name: item.to_dict() for name, item in profiles.items()},
        "weights": {
            "state": 1.0,
            "future": 0.5,
            "vision": 1.0,
            "observation": 0.15,
            "displacement": 0.1 if recipe == "mixed-enhanced" else 0.0,
        },
        "selection": "Minimum validation macro position RMSE after each complete epoch",
        "label_reference": "Exact simulator state; variance=0 and coordinate validity=1",
        "test_usage": "Disjoint episodes; evaluated only after checkpoint selection is complete",
        "scope": "Single-target synthetic RGB/radio state prediction; no language or human RLHF",
    }
    seed_all(seed)
    model = build_model(model_config(config, recipe)).to(device)
    initial_fingerprint = model_fingerprint(model)
    optimizer, optimizer_policy = make_optimizer(model, config.learning_rate)
    order_rng = np.random.default_rng(config.batch_order_seed)
    history, best_checkpoint = [], None
    completed_epochs, updates, best = 0, 0, float("inf")
    initial_validation = None
    if resume:
        saved = read_resume_checkpoint(out / "resume.pt")
        if saved.get("pretraining_protocol") != protocol:
            raise ValueError("Resume protocol, data plan, or implementation bytes changed")
        for key in ("torch", "numpy", "python", "device"):
            if saved["runtime"][key] != runtime_info(device)[key]:
                raise ValueError(f"Resume runtime differs in {key}")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        order_rng.bit_generator.state = json.loads(saved["numpy_rng_json"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
        completed_epochs, updates = saved["completed_epochs"], saved["optimizer_updates"]
        history, best, best_checkpoint = saved["history"], saved["best_rmse"], saved["best_checkpoint"]
        initial_validation = saved["initial_validation"]
        initial_fingerprint = saved["initial_parameters_sha256"]
        if best_checkpoint is None:
            raise ValueError("A completed epoch must contain its selected checkpoint")
        save_checkpoint(out / "model.pt", best_checkpoint)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "recipe": recipe,
        "seed": seed,
        "profile": config.profile,
        "source": source,
        "runtime": runtime_info(device),
        "protocol": protocol,
        "model_config": model.config.to_dict(),
        "scene_config": scene.to_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "visual_parameters": sum(p.numel() for p in model.visual.parameters()),
        "initial_parameters_sha256": initial_fingerprint,
        "optimizer_policy": optimizer_policy,
        "completed_epochs": completed_epochs,
        "optimizer_updates": updates,
        "resumed": resume,
        "weights_included": True,
    }
    write_json(out / "episode_plan.json", plans)
    write_json(out / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        if initial_validation is None:
            initial_validation = evaluate(model, plans["validation"], profiles, config.batch_size, device)
        manifest["initial_validation"] = initial_validation
        write_json(out / "manifest.json", manifest)
        corpus = _corpus(plans["train"], profiles)
        manifest["training_array_bytes"] = sum(
            value.nbytes for episode in corpus for value in episode.values()
        )
        for epoch in range(completed_epochs, config.epochs):
            epoch_started = time.perf_counter()
            model.train()
            totals, batches, horizons_seen = {}, 0, {str(h): 0 for h in config.horizons}
            permutation = order_rng.permutation(config.episodes)
            for start in range(0, config.episodes, config.batch_size):
                indices = permutation[start : start + config.batch_size]
                batch = batchify([corpus[int(index)] for index in indices], device)
                horizon = int(order_rng.choice(config.horizons))
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    batch, forecast_s=horizon * scene.dt, batched_encoding=True, return_visual=True
                )
                loss, details = pretraining_loss(
                    output, batch, horizon, process_weight=protocol["weights"]["displacement"]
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite pretraining loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                for name, value in details.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
                totals["gradient_norm_before_clip"] = totals.get("gradient_norm_before_clip", 0.0) + float(
                    norm
                )
                batches += 1
                updates += 1
                horizons_seen[str(horizon)] += 1
            validation = evaluate(model, plans["validation"], profiles, config.batch_size, device)
            score = validation["macro_position_rmse_m"]
            if score < best:
                best = score
                best_checkpoint = _checkpoint(model, scene, protocol, seed, epoch + 1, updates, plans)
                save_checkpoint(out / "model.pt", best_checkpoint)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "epoch": epoch + 1,
                "optimizer_updates": updates,
                "loss_components_batch_mean": {key: value / batches for key, value in totals.items()},
                "horizons_update_counts": horizons_seen,
                "validation": validation,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            completed_epochs = epoch + 1
            print(
                json.dumps(
                    {
                        "cell": cell_key,
                        "epoch": completed_epochs,
                        "updates": updates,
                        "validation_macro_rmse_m": score,
                        "train_loss": totals["loss"] / batches,
                    }
                ),
                flush=True,
            )
            save_resume_checkpoint(
                out / "resume.pt",
                {
                    **_checkpoint(model, scene, protocol, seed, completed_epochs, updates, plans),
                    "optimizer": optimizer.state_dict(),
                    "numpy_rng_json": json.dumps(order_rng.bit_generator.state),
                    "torch_rng": torch.get_rng_state(),
                    **({"torch_cuda_rng": torch.cuda.get_rng_state_all()} if device.type == "cuda" else {}),
                    "completed_epochs": completed_epochs,
                    "optimizer_updates": updates,
                    "best_rmse": best,
                    "best_checkpoint": best_checkpoint,
                    "history": history,
                    "runtime": runtime_info(device),
                    "initial_validation": initial_validation,
                    "initial_parameters_sha256": initial_fingerprint,
                },
            )
            write_json(out / "history.json", history)
            manifest.update(
                completed_epochs=completed_epochs, optimizer_updates=updates, best_validation_rmse_m=best
            )
            write_json(out / "manifest.json", manifest)
        del corpus
        if best_checkpoint is None or completed_epochs != config.epochs:
            raise RuntimeError("Training ended without completing the planned epochs")
        save_checkpoint(
            out / "final.pt", _checkpoint(model, scene, protocol, seed, completed_epochs, updates, plans)
        )
        selected = read_checkpoint(out / "model.pt")
        model.load_state_dict(selected["state_dict"], strict=True)
        test = evaluate(model, plans["test"], profiles, config.batch_size, device, config.horizons)
        metrics = {
            "schema_version": 1,
            "selection_split": "validation",
            "evaluation_split": "test",
            "selected_epoch": selected["best_epoch"],
            "selected_update": selected["best_update"],
            "initial_validation": initial_validation,
            "selected_validation": history[selected["best_epoch"] - 1]["validation"],
            "selected_test": test,
            "selected_parameters_sha256": model_fingerprint(model),
            "selected_checkpoint_sha256": sha256_file(out / "model.pt"),
            "scope": "Frozen synthetic evaluation for this profile; formal uses a separate seed namespace",
        }
        write_json(out / "metrics.json", metrics)
        write_json(out / "history.json", history)
        manifest.update(
            status="completed",
            completed_epochs=completed_epochs,
            optimizer_updates=updates,
            best_validation_rmse_m=best,
            selected_epoch=selected["best_epoch"],
            elapsed_seconds_this_call=time.perf_counter() - started,
        )
        write_json(out / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            completed_epochs=completed_epochs,
            optimizer_updates=updates,
        )
        write_json(out / "manifest.json", manifest)
        raise
    cell = {
        "status": "completed",
        "path": cell_key,
        "recipe": recipe,
        "seed": seed,
        "manifest": f"{cell_key}/manifest.json",
        "history": f"{cell_key}/history.json",
        "metrics": f"{cell_key}/metrics.json",
        "artifacts_sha256": _file_hashes(Path(study_dir), out),
        "test_macro_position_rmse_m": test["macro_position_rmse_m"],
    }
    verify_cell_artifacts(study_dir, cell)
    return cell


def _comparison(study_dir: Path, cells: dict) -> dict:
    summaries = {}
    for recipe in RECIPES:
        chosen = [row for row in cells.values() if row["recipe"] == recipe]
        values = np.array([row["test_macro_position_rmse_m"] for row in chosen], dtype=float)
        reports = [json.loads((study_dir / row["metrics"]).read_text()) for row in chosen]
        summaries[recipe] = {
            "initializations": len(chosen),
            "test_macro_position_rmse_m_mean": float(values.mean()),
            "test_macro_position_rmse_m_std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "per_profile_rmse_m_mean": {
                name: float(
                    np.mean(
                        [report["selected_test"]["profiles"][name]["position_rmse_m"] for report in reports]
                    )
                )
                for name in reports[0]["selected_test"]["profiles"]
            },
        }
    enhanced = summaries["mixed-enhanced"]["test_macro_position_rmse_m_mean"]
    return {
        "recipes": summaries,
        "enhanced_improvement_percent": {
            name: 100 * (1 - enhanced / summaries[name]["test_macro_position_rmse_m_mean"])
            for name in ("nominal-control", "mixed-control")
        },
        "interpretation": "Positive percentage means lower RMSE. Combined enhanced changes are not separately attributed.",
    }


def run_pretraining(
    outdir: Path,
    config: PretrainingConfig,
    *,
    device_name: str = "cpu",
    source_root: Path | None = None,
    resume: bool = False,
) -> dict:
    outdir = Path(outdir)
    device = resolve_device(device_name)
    root = source_root or Path(__file__).resolve().parents[2]
    source = source_provenance(root)
    planned = [f"{recipe}-seed{seed}" for recipe in RECIPES for seed in config.initialization_seeds]
    if outdir.exists() and any(outdir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite existing study: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    previous = None
    if resume:
        previous = json.loads((outdir / "study.json").read_text())
        if previous["configuration"] != config.to_dict() or previous["planned_cells"] != planned:
            raise ValueError("Resume study configuration or planned cells changed")
        if previous["source"]["implementation_sha256"] != source["implementation_sha256"]:
            raise ValueError("Resume implementation source changed")
    study = {
        "schema_version": 1,
        "status": "running",
        "profile": config.profile,
        "git_commit": source["git_commit"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": runtime_info(device),
        "source": source,
        "configuration": config.to_dict(),
        "planned_cells": planned,
        "cells": {} if previous is None else previous["cells"],
        "artifacts_sha256": {},
        "recovery_events": [] if previous is None else previous.get("recovery_events", []),
    }
    write_json(outdir / "study.json", study)
    try:
        for recipe in RECIPES:
            for seed in config.initialization_seeds:
                key = f"{recipe}-seed{seed}"
                existing = study["cells"].get(key)
                if existing and existing.get("status") == "completed":
                    verify_cell_artifacts(outdir, existing)
                    continue
                study["cells"][key] = {"status": "running", "path": key, "recipe": recipe, "seed": seed}
                write_json(outdir / "study.json", study)
                cell_resume = resume and (outdir / key / "resume.pt").exists()
                if resume and (outdir / key).exists() and not cell_resume:
                    # No complete epoch was committed. Preserve every partial
                    # artifact and explicitly restart this cell from its seed.
                    archive = outdir.parent / f"{outdir.name}-interrupted" / f"{key}-{time.time_ns()}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    (outdir / key).rename(archive)
                    event = {
                        "cell": key,
                        "action": "restart_from_random_initialization",
                        "reason": "No completed-epoch checkpoint",
                        "preserved_attempt": str(archive),
                    }
                    study.setdefault("recovery_events", []).append(event)
                    print(json.dumps(event), flush=True)
                    write_json(outdir / "study.json", study)
                study["cells"][key] = train_cell(
                    outdir, config, recipe, seed, device, source, resume=cell_resume
                )
                write_json(outdir / "study.json", study)
        if set(study["cells"]) != set(planned):
            raise RuntimeError("Completed cells differ from the planned experiment")
        for cell in study["cells"].values():
            verify_cell_artifacts(outdir, cell)
            study["artifacts_sha256"].update(cell["artifacts_sha256"])
        if source_provenance(root)["implementation_sha256"] != source["implementation_sha256"]:
            raise RuntimeError("Implementation source changed while the study was running")
        study["comparison"] = _comparison(outdir, study["cells"])
        study["status"] = "completed"
        write_json(outdir / "study.json", study)
    except BaseException as exc:
        study.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        for cell in study["cells"].values():
            if cell.get("status") == "running":
                cell.update(status="failed", error=study["error"])
        write_json(outdir / "study.json", study)
        raise
    return study
