"""Export actual neural samples, including exact pixels and rendering metadata."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from .config import config_from_dict
from .engine import resolve_device
from .io import load_checkpoint, runtime_identity, sha256_file, write_json
from .model import TextVideoModel, sample, tokenize_captions


def generate(checkpoint_path, captions, out, *, device_name, seed=101, steps=None, guidance_scale=None, preview_scale=8, same_seed=False):
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("kind") != "weights":
        raise ValueError("Sampling requires a selected/final weights checkpoint")
    config = config_from_dict(checkpoint["config"])
    # Sampling uses float32 weights even if training used BF16 autocast; this is
    # explicit in the output metadata and makes CPU inspection possible.
    device = resolve_device(device_name)
    torch.set_num_threads(config.training.threads)
    tokenize_captions(captions, max_text_bytes=config.model.max_text_bytes)
    if type(seed) is not int or seed < 0 or type(preview_scale) is not int or preview_scale < 1:
        raise ValueError("seed must be nonnegative and preview_scale must be positive")
    steps = config.sampling.steps if steps is None else steps
    guidance_scale = config.sampling.guidance_scale if guidance_scale is None else guidance_scale
    if guidance_scale != 1 and config.training.cfg_dropout == 0:
        raise ValueError("CFG requires null-caption training; this checkpoint has cfg_dropout=0 (use guidance_scale=1)")
    model = TextVideoModel(config.model).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    sheet_width = config.model.width * preview_scale * config.model.frames
    row_height = config.model.height * preview_scale + 36
    sheet = Image.new("RGB", (sheet_width, row_height * len(captions)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, caption in enumerate(captions):
        sample_seed = seed if same_seed else seed + index
        clip = sample(model, [caption], steps=steps, guidance_scale=guidance_scale, seed=sample_seed, device=device)
        frames = ((clip[0].cpu().permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        prefix = f"sample_{index:03d}"
        npz = out / f"{prefix}.npz"
        np.savez_compressed(npz, frames=frames)
        # macro_block_size=1 prevents the encoder from silently resizing video.
        imageio.mimwrite(out / f"{prefix}.mp4", frames, fps=config.sampling.fps, codec="libx264", macro_block_size=1, ffmpeg_log_level="error")
        previews = [Image.fromarray(frame).resize((config.model.width * preview_scale, config.model.height * preview_scale), Image.Resampling.NEAREST) for frame in frames]
        previews[0].save(out / f"{prefix}_preview.gif", save_all=True, append_images=previews[1:], duration=round(1000 / config.sampling.fps), loop=0)
        for frame_index, preview in enumerate(previews):
            sheet.paste(preview, (frame_index * config.model.width * preview_scale, index * row_height + 36))
        draw.text((4, index * row_height + 4), f"{index} | seed {sample_seed} | {caption}", fill="black")
        records.append({
            "index": index, "caption": caption, "seed": sample_seed,
            "npz": npz.name, "npz_sha256": sha256_file(npz),
            "mp4": f"{prefix}.mp4", "mp4_sha256": sha256_file(out / f"{prefix}.mp4"),
            "gif": f"{prefix}_preview.gif", "raw_shape": list(frames.shape),
        })
    sheet.save(out / "contact_sheet.png")
    result = {
        "schema_version": 1, "source": "neural flow integration from Gaussian noise; no reference video used during sampling",
        "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_step": checkpoint["step"],
        "architecture": checkpoint["architecture"], "model": checkpoint["config"]["model"],
        "steps": steps, "integrator": "heun", "guidance_scale": guidance_scale,
        "fps": config.sampling.fps, "duration_seconds": config.model.frames / config.sampling.fps,
        "precision": "float32", "preview_scale": preview_scale, "preview_interpolation": "nearest neighbor",
        "seed_mode": "same noise for every prompt" if same_seed else "seed plus prompt index",
        "runtime": runtime_identity(device), "samples": records,
        "audio": "not implemented", "test_references_used": False,
    }
    write_json(out / "generation.json", result)
    return result
