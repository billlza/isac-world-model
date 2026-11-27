"""Auditable pretraining / supervised continuation for the independent model."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time as clock

import torch

from .config import config_from_dict
from .data import SPLITS, VideoDataset, validate_manifest
from .io import ARCHITECTURE, canonical_hash, load_checkpoint, runtime_identity, save_checkpoint, sha256_file, source_identity, write_json
from .model import TextVideoModel, tokenize_captions
from .objective import flow_objective


def resolve_device(name, precision="float32"):
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be explicitly cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback is permitted")
    device = torch.device("cuda:0" if name == "cuda" else "cpu")
    if precision == "bfloat16" and (name != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bfloat16 training requires an explicitly selected, supported CUDA device")
    return device


def validate_data(manifest, config):
    records = validate_manifest(manifest)
    counts = {split: 0 for split in SPLITS}
    groups = {split: set() for split in SPLITS}
    for row in records:
        counts[row["split"]] += 1
        groups[row["split"]].add(row["group_id"])
        if len(row["reference_weight"]) < config.model.frames:
            raise ValueError(f"{row['id']}: fewer frames than configured")
        if sum(row["reference_weight"][:config.model.frames]) <= 0:
            raise ValueError(f"{row['id']}: no trusted support in selected frames")
        if row["fps"] != config.sampling.fps:
            raise ValueError(f"{row['id']}: fps={row['fps']} differs from configured fps={config.sampling.fps}; resample explicitly before building the manifest")
        tokenize_captions([row["caption"]], max_text_bytes=config.model.max_text_bytes)
    if any(count == 0 for count in counts.values()):
        raise ValueError("Manifest must contain nonempty train, validation, and test splits")
    assets = [{"id": row["id"], "sha256": row["sha256"]} for row in sorted(records, key=lambda row: row["id"])]
    binding = {"manifest_sha256": sha256_file(manifest), "assets": assets}
    return {
        "schema_version": 1, "data_fingerprint": canonical_hash(binding),
        "manifest_sha256": binding["manifest_sha256"], "asset_count": len(records),
        "split_counts": counts, "group_counts": {split: len(values) for split, values in groups.items()},
        "requested_shape": [3, config.model.frames, config.model.height, config.model.width],
        "fps": config.sampling.fps,
        "split_check": "group, file-bytes, and decoded-pixel identity checked across splits",
        "provenance_check": "declared source and license present; ownership not independently verified",
    }


def _dataset(manifest, split, config):
    return VideoDataset(manifest, split, config.model.frames, config.model.height, config.model.width)


def _batch(dataset, indices, device):
    rows = [dataset[int(index)] for index in indices]
    return (
        torch.stack([row["video"] for row in rows]).to(device),
        [row["caption"] for row in rows],
        torch.stack([row["reference_weight"] for row in rows]).to(device),
    )


def accumulation_scales(weights_list):
    """Normalize over the effective batch, independently for frames and pairs."""
    if not weights_list:
        raise ValueError("At least one microbatch is required")
    supports, pairs = [], []
    for weights in weights_list:
        if weights.ndim != 2 or not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
            raise ValueError("Expected finite reference weights [B,T] in [0,1]")
        support = float(weights.detach().sum())
        if support <= 0:
            raise ValueError("Every microbatch requires trusted reference support")
        supports.append(support)
        pairs.append(float(torch.minimum(weights[:, 1:], weights[:, :-1]).detach().sum()))
    total_support, total_pairs = sum(supports), sum(pairs)
    return [(support / total_support, pair / total_pairs if total_pairs else 0.0) for support, pair in zip(supports, pairs)]


def _split_lineage(manifest, inherited=()):
    """Keep episode/content split assignments across all supervised stages."""
    lineage = [dict(item) for item in inherited]
    seen = {canonical_hash(item) for item in lineage}
    indexes = {key: {} for key in ("group_id", "sha256", "decoded_sha256")}
    for row in lineage:
        for key in indexes:
            value = row.get(key)
            if value is not None:
                indexes[key][value] = row["split"]
    for row in validate_manifest(manifest):
        entry = {key: row[key] for key in ("id", "group_id", "split", "sha256")}
        entry["decoded_sha256"] = row["_decoded_sha256"]
        for key, assignments in indexes.items():
            value = entry[key]
            if value in assignments and assignments[value] != entry["split"]:
                raise ValueError(f"Cross-stage split leakage for {key}: {entry['id']}")
            assignments[value] = entry["split"]
        identity = canonical_hash(entry)
        if identity not in seen:
            lineage.append(entry)
            seen.add(identity)
    return lineage


def _autocast(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()


def _state_cpu(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@torch.no_grad()
def evaluate_model(model, dataset, config, device, seed=17001):
    """Fixed noise/time draw; all held-out references contribute with their weights."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"flow_sum": 0.0, "temporal_sum": 0.0, "reference_support": 0.0, "temporal_support": 0.0, "frame_mse_sum": 0.0}
    try:
        for start in range(0, len(dataset), config.training.batch_size):
            video, captions, weights = _batch(dataset, range(start, min(start + config.training.batch_size, len(dataset))), device)
            noise = torch.randn(video.shape, generator=generator, device=device)
            time = torch.rand(video.shape[0], generator=generator, device=device)
            mixed = (1 - time[:, None, None, None, None]) * noise + time[:, None, None, None, None] * video
            with _autocast(device, config.training.precision):
                prediction = model(mixed, time, captions)
                _, metrics = flow_objective(prediction, video, noise, time, weights, config.training.temporal_weight)
            totals["flow_sum"] += metrics["flow_loss"] * metrics["reference_support"]
            totals["temporal_sum"] += metrics["temporal_loss"] * metrics["temporal_support"]
            totals["reference_support"] += metrics["reference_support"]
            totals["temporal_support"] += metrics["temporal_support"]
            totals["frame_mse_sum"] += metrics["frame_mse_unweighted"] * video.shape[0]
    finally:
        model.train(was_training)
    flow = totals["flow_sum"] / totals["reference_support"]
    temporal = totals["temporal_sum"] / totals["temporal_support"] if totals["temporal_support"] else 0.0
    return {
        "flow_loss": flow, "temporal_loss": temporal,
        "total_loss": flow + config.training.temporal_weight * temporal,
        "frame_mse_unweighted": totals["frame_mse_sum"] / len(dataset),
        "reference_support": totals["reference_support"], "temporal_support": totals["temporal_support"],
        "samples": len(dataset), "noise_seed": seed,
        "interpretation": "fixed-noise supervised loss; not a perceptual, semantic, physics, or generalization benchmark",
    }


def _weights_payload(model_state, config, provenance, stage, step, selection, weights_kind):
    return {
        "schema_version": 1, "architecture": ARCHITECTURE, "kind": "weights",
        "model_state": model_state, "config": config.to_dict(), "provenance": provenance,
        "stage": stage, "step": step, "selection": selection, "weights_kind": weights_kind,
    }


def train(config, manifest, out, *, device_name, stage="pretrain", init_checkpoint=None, resume=False):
    if stage not in ("pretrain", "sft"):
        raise ValueError("Only pretrain and sft are implemented; RL/RLHF are not executed")
    if (stage == "pretrain" and init_checkpoint) or (stage == "sft" and not init_checkpoint):
        raise ValueError("pretrain starts random; sft requires --init-checkpoint from this architecture")
    device = resolve_device(device_name, config.training.precision)
    torch.set_num_threads(config.training.threads)
    torch.manual_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)
    # Same-runtime CPU resume is tested bit-for-bit. GPU determinism depends on
    # kernel/runtime choices and is not inferred from a fixed seed.
    torch.backends.cudnn.benchmark = False
    data_report = validate_data(manifest, config)
    training_data = _dataset(manifest, "train", config)
    validation_data = _dataset(manifest, "validation", config)
    provenance = {
        **source_identity(), "runtime": runtime_identity(device),
        "data": data_report, "config_hash": canonical_hash(config.to_dict()),
        "initialization": "random" if stage == "pretrain" else "own_checkpoint",
        "initial_checkpoint_sha256": sha256_file(init_checkpoint) if init_checkpoint else None,
        "train_sampling": "with replacement, dedicated checkpointed generator",
        "selection_split": "validation", "test_used_during_training": False,
    }
    model = TextVideoModel(config.model).to(device)
    inherited_lineage = []
    if init_checkpoint:
        initial = load_checkpoint(init_checkpoint)
        if initial.get("kind") != "weights" or initial["config"]["model"] != asdict(config.model):
            raise ValueError("SFT initializer must be compatible weights from this architecture")
        if initial["stage"] not in ("pretrain", "sft"):
            raise ValueError("SFT initializer has an unsupported training stage")
        model.load_state_dict(initial["model_state"], strict=True)
        provenance["initial_checkpoint_stage"] = initial["stage"]
        provenance["initial_checkpoint_step"] = initial["step"]
        inherited_lineage = initial["provenance"].get("split_lineage")
        if inherited_lineage is None:
            raise ValueError("SFT initializer lacks split lineage; cross-stage held-out isolation cannot be verified")
    provenance["split_lineage"] = _split_lineage(manifest, inherited_lineage)
    ema = deepcopy(model).eval().requires_grad_(False) if config.training.ema_decay else None
    weights_kind = "ema" if ema is not None else "raw"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    data_rng = torch.Generator(device="cpu").manual_seed(config.training.seed + 1)
    noise_rng = torch.Generator(device=device).manual_seed(config.training.seed + 2)
    out = Path(out).resolve()
    history, validation_history = [], []
    step, best_step, best_score, best_state = 0, 0, float("inf"), None
    previous_seconds = 0.0

    if resume:
        if not out.is_dir():
            raise ValueError("--resume requires an existing run directory")
        prior_status = json.loads((out / "status.json").read_text())
        if prior_status.get("status") == "completed":
            raise ValueError("Run is already completed; extending a budget requires a new explicit stage/run")
        checkpoint = load_checkpoint(out / "resume.pt")
        if checkpoint.get("kind") != "resume" or checkpoint["config"] != config.to_dict() or checkpoint["stage"] != stage:
            raise ValueError("Resume requires exactly the original configuration and stage")
        original = checkpoint["provenance"]
        for field in ("source_fingerprint", "config_hash", "initial_checkpoint_sha256", "runtime"):
            if original[field] != provenance[field]:
                raise ValueError(f"Resume provenance mismatch: {field}")
        if original["data"]["data_fingerprint"] != data_report["data_fingerprint"]:
            raise ValueError("Resume data fingerprint mismatch")
        provenance = original
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if ema is not None:
            ema.load_state_dict(checkpoint["ema_state"], strict=True)
        data_rng.set_state(checkpoint["data_rng"])
        noise_rng.set_state(checkpoint["noise_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        step, best_step, best_score = checkpoint["step"], checkpoint["best_step"], checkpoint["best_score"]
        best_state = checkpoint["best_state"]
        history, validation_history = checkpoint["history"], checkpoint["validation_history"]
        previous_seconds = checkpoint["elapsed_seconds"]
        save_checkpoint(out / "selected.pt", _weights_payload(best_state, config, provenance, stage, best_step, "minimum validation total_loss", weights_kind))
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "config.json", config.to_dict())
        write_json(out / "provenance.json", provenance)

    start = clock.perf_counter()

    def elapsed():
        return previous_seconds + clock.perf_counter() - start

    def snapshot():
        payload = {
            "schema_version": 1, "architecture": ARCHITECTURE, "kind": "resume",
            "config": config.to_dict(), "provenance": provenance, "stage": stage, "step": step,
            "model_state": _state_cpu(model), "ema_state": _state_cpu(ema) if ema is not None else None,
            "optimizer_state": optimizer.state_dict(), "data_rng": data_rng.get_state(), "noise_rng": noise_rng.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_step": best_step, "best_score": best_score, "best_state": best_state,
            "history": history, "validation_history": validation_history, "elapsed_seconds": elapsed(),
        }
        save_checkpoint(out / "resume.pt", payload)
        write_json(out / "history.json", {"training": history, "validation": validation_history})
        write_json(out / "status.json", {"status": "running", "stage": stage, "step": step, "planned_steps": config.training.steps, "elapsed_seconds": elapsed()})

    try:
        if not resume:
            initial_metrics = evaluate_model(ema if ema is not None else model, validation_data, config, device)
            best_score, best_state = initial_metrics["total_loss"], _state_cpu(ema if ema is not None else model)
            validation_history.append({"step": 0, **initial_metrics})
            save_checkpoint(out / "selected.pt", _weights_payload(best_state, config, provenance, stage, 0, "minimum validation total_loss", weights_kind))
            snapshot()
        model.train()
        while step < config.training.steps:
            optimizer.zero_grad(set_to_none=True)
            micro_metrics = []
            batches = []
            for _ in range(config.training.gradient_accumulation):
                indices = torch.randint(len(training_data), (config.training.batch_size,), generator=data_rng)
                batches.append(_batch(training_data, indices, torch.device("cpu")))
            scales = accumulation_scales([batch[2] for batch in batches])
            for (video, captions, weights), (flow_scale, temporal_scale) in zip(batches, scales):
                video, weights = video.to(device), weights.to(device)
                noise = torch.randn(video.shape, generator=noise_rng, device=device)
                time = torch.rand(video.shape[0], generator=noise_rng, device=device)
                dropped = torch.rand(video.shape[0], generator=noise_rng, device=device) < config.training.cfg_dropout
                captions = ["" if drop else caption for caption, drop in zip(captions, dropped.tolist())]
                mixed = (1 - time[:, None, None, None, None]) * noise + time[:, None, None, None, None] * video
                with _autocast(device, config.training.precision):
                    prediction = model(mixed, time, captions)
                    loss, metrics = flow_objective(prediction, video, noise, time, weights, config.training.temporal_weight, flow_scale=flow_scale, temporal_scale=temporal_scale)
                loss.backward()
                metrics["caption_dropout_count"] = int(dropped.sum())
                micro_metrics.append(metrics)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if ema is not None:
                with torch.no_grad():
                    for target, current in zip(ema.parameters(), model.parameters()):
                        target.lerp_(current, 1 - config.training.ema_decay)
                    for target, current in zip(ema.buffers(), model.buffers()):
                        target.copy_(current)
            entry = {key: sum(item[key] for item in micro_metrics) for key in ("reference_support", "temporal_support", "caption_dropout_count")}
            entry["flow_loss"] = sum(item["flow_loss"] * scale[0] for item, scale in zip(micro_metrics, scales))
            entry["temporal_loss"] = sum(item["temporal_loss"] * scale[1] for item, scale in zip(micro_metrics, scales))
            entry["total_loss"] = entry["flow_loss"] + config.training.temporal_weight * entry["temporal_loss"]
            entry["frame_mse_unweighted"] = sum(item["frame_mse_unweighted"] for item in micro_metrics) / len(micro_metrics)
            history.append({"step": step, "gradient_norm_before_clip": float(norm), **entry})
            is_validation = step % config.training.validate_every == 0 or step == config.training.steps
            if is_validation:
                metrics = evaluate_model(ema if ema is not None else model, validation_data, config, device)
                validation_history.append({"step": step, **metrics})
                if metrics["total_loss"] < best_score:
                    best_score, best_step = metrics["total_loss"], step
                    best_state = _state_cpu(ema if ema is not None else model)
                    save_checkpoint(out / "selected.pt", _weights_payload(best_state, config, provenance, stage, step, "minimum validation total_loss", weights_kind))
                print(json.dumps({"step": step, "training_loss": entry["total_loss"], "validation_loss": metrics["total_loss"], "elapsed_seconds": round(elapsed(), 2)}), file=sys.stderr, flush=True)
            if is_validation or step % config.training.checkpoint_every == 0:
                snapshot()
        final_state = _state_cpu(ema if ema is not None else model)
        final_hash = save_checkpoint(out / "final.pt", _weights_payload(final_state, config, provenance, stage, step, "last update", weights_kind))
        result = {
            "schema_version": 1, "status": "completed", "architecture": ARCHITECTURE, "stage": stage,
            "step": step, "planned_steps": config.training.steps,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds": elapsed(), "selected_step": best_step,
            "initial_validation": validation_history[0], "selected_validation_loss": best_score,
            "final_validation": validation_history[-1], "test_used_during_training": False,
            "selected_checkpoint_sha256": sha256_file(out / "selected.pt"), "final_checkpoint_sha256": final_hash,
            "rl": "not implemented or executed", "rlhf": "not implemented or executed",
        }
        write_json(out / "summary.json", result)
        write_json(out / "status.json", result)
        return result
    except BaseException as exc:
        write_json(out / "status.json", {
            "status": "failed", "stage": stage, "step": step, "planned_steps": config.training.steps,
            "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": elapsed(),
            "recovery": "Only a hash-verified resume.pt with identical config/source/data/runtime may resume",
        })
        raise


def evaluate_checkpoint(checkpoint_path, manifest, out, *, device_name, split="test"):
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("kind") != "weights":
        raise ValueError("Evaluation requires a selected/final weights checkpoint")
    config = config_from_dict(checkpoint["config"])
    device = resolve_device(device_name, config.training.precision)
    torch.set_num_threads(config.training.threads)
    report = validate_data(manifest, config)
    if report["data_fingerprint"] != checkpoint["provenance"]["data"]["data_fingerprint"]:
        raise ValueError("Evaluation data differs from the checkpoint's pinned manifest")
    model = TextVideoModel(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dataset = _dataset(manifest, split, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_step": checkpoint["step"],
        "split": split, "data_fingerprint": report["data_fingerprint"],
        "runtime": runtime_identity(device),
        **evaluate_model(model, dataset, config, device, seed=29003 if split == "test" else 17001),
    }
    write_json(out / "evaluation.json", result)
    return result
