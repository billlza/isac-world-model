"""A locally initialized pixel-space video flow model.

This implementation combines standard patch embeddings, factorized spatial and
temporal self-attention, and cross-attention to an independently trained UTF-8
byte encoder. It loads no tokenizer, VAE, model weights, or external assets.
Attention is bidirectional within the generated clip; this is not a causal
forecasting model. The flow convention is noise at t=0 and video at t=1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

PAD_TOKEN = 256
BOS_TOKEN = 257
EOS_TOKEN = 258
VOCAB_SIZE = 259


@dataclass(frozen=True)
class ModelConfig:
    height: int = 32
    width: int = 32
    frames: int = 8
    patch_size: int = 4
    hidden: int = 128
    depth: int = 4
    heads: int = 4
    text_layers: int = 2
    max_text_bytes: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "height", "width", "frames", "patch_size", "hidden", "depth",
            "heads", "text_layers", "max_text_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.height % self.patch_size or self.width % self.patch_size:
            raise ValueError("height and width must be divisible by patch_size")
        if self.hidden < 4 or self.hidden % self.heads:
            raise ValueError("hidden must be at least 4 and divisible by heads")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")


def tokenize_captions(
    captions: Sequence[str],
    *,
    max_text_bytes: int,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return byte IDs and a padding mask (True means padding), without truncation.

    The byte limit excludes BOS/EOS. An empty caption has two valid special
    tokens, so classifier-free conditioning never receives an all-masked row.
    """
    if isinstance(captions, (str, bytes)) or not isinstance(captions, Sequence):
        raise TypeError("captions must be a non-empty sequence of strings")
    if not captions:
        raise ValueError("captions must not be empty; use [''] for null conditioning")
    if isinstance(max_text_bytes, bool) or not isinstance(max_text_bytes, int) or max_text_bytes <= 0:
        raise ValueError("max_text_bytes must be a positive integer")
    encoded = []
    for index, caption in enumerate(captions):
        if not isinstance(caption, str):
            raise TypeError(f"caption {index} must be a string")
        try:
            value = caption.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError(f"caption {index} is not valid UTF-8 text") from exc
        if len(value) > max_text_bytes:
            raise ValueError(
                f"caption {index} has {len(value)} UTF-8 bytes, exceeding "
                f"max_text_bytes={max_text_bytes}; captions are not truncated"
            )
        encoded.append([BOS_TOKEN, *value, EOS_TOKEN])
    length = max(map(len, encoded))
    ids = torch.full((len(encoded), length), PAD_TOKEN, dtype=torch.long, device=device)
    for index, values in enumerate(encoded):
        ids[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    return ids, ids.eq(PAD_TOKEN)


class ByteTextEncoder(nn.Module):
    """Randomly initialized byte embeddings and independently initialized layers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_text_bytes = config.max_text_bytes
        self.embedding = nn.Embedding(VOCAB_SIZE, config.hidden, padding_idx=PAD_TOKEN)
        self.position = nn.Parameter(torch.empty(1, config.max_text_bytes + 2, config.hidden))
        nn.init.normal_(self.position, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden,
                nhead=config.heads,
                dim_feedforward=4 * config.hidden,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.text_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden)

    def forward(self, captions: Sequence[str]) -> tuple[Tensor, Tensor]:
        ids, padding_mask = tokenize_captions(
            captions, max_text_bytes=self.max_text_bytes, device=self.position.device
        )
        context = self.embedding(ids) + self.position[:, : ids.shape[1]]
        for layer in self.layers:
            context = layer(context, src_key_padding_mask=padding_mask)
        context = self.norm(context).masked_fill(padding_mask.unsqueeze(-1), 0)
        return context, padding_mask


def _time_features(time: Tensor, hidden: int) -> Tensor:
    half = hidden // 2
    frequency = torch.exp(
        -math.log(10000) * torch.arange(half, device=time.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = 1000 * time.float().unsqueeze(-1) * frequency.unsqueeze(0)
    features = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if hidden % 2:
        features = torch.cat((features, torch.zeros_like(features[:, :1])), dim=-1)
    return features


class VideoBlock(nn.Module):
    """Spatial attention per frame, temporal attention per patch, then text attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden
        self.norms = nn.ModuleList([nn.LayerNorm(hidden, elementwise_affine=False) for _ in range(4)])
        self.spatial_attention = nn.MultiheadAttention(
            hidden, config.heads, dropout=config.dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            hidden, config.heads, dropout=config.dropout, batch_first=True
        )
        self.text_attention = nn.MultiheadAttention(
            hidden, config.heads, dropout=config.dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(config.dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 8 * hidden))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, video: Tensor, condition: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        batch, frames, patches, hidden = video.shape
        modulation = self.modulation(condition).chunk(8, dim=-1)

        def normalized(index: int, value: Tensor) -> Tensor:
            shift, scale = modulation[2 * index: 2 * index + 2]
            return self.norms[index](value) * (1 + scale[:, None, None]) + shift[:, None, None]

        spatial = normalized(0, video).reshape(batch * frames, patches, hidden)
        spatial = self.spatial_attention(spatial, spatial, spatial, need_weights=False)[0]
        video = video + self.dropout(spatial.reshape(batch, frames, patches, hidden))

        temporal = normalized(1, video).permute(0, 2, 1, 3).reshape(batch * patches, frames, hidden)
        temporal = self.temporal_attention(temporal, temporal, temporal, need_weights=False)[0]
        temporal = temporal.reshape(batch, patches, frames, hidden).permute(0, 2, 1, 3)
        video = video + self.dropout(temporal)

        query = normalized(2, video).reshape(batch, frames * patches, hidden)
        attended = self.text_attention(
            query, context, context, key_padding_mask=mask, need_weights=False
        )[0]
        video = video + self.dropout(attended.reshape(batch, frames, patches, hidden))
        return video + self.feedforward(normalized(3, video))


class TextVideoModel(nn.Module):
    """Predict pixel-space flow velocity for a complete text-conditioned video clip."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config
        self.text_encoder = ByteTextEncoder(config)
        self.patch_embed = nn.Conv3d(
            3, config.hidden, kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size),
        )
        patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        self.spatial_position = nn.Parameter(torch.empty(1, 1, patches, config.hidden))
        self.temporal_position = nn.Parameter(torch.empty(1, config.frames, 1, config.hidden))
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)
        self.time_embedding = nn.Sequential(
            nn.Linear(config.hidden, 4 * config.hidden), nn.SiLU(),
            nn.Linear(4 * config.hidden, config.hidden),
        )
        self.blocks = nn.ModuleList([VideoBlock(config) for _ in range(config.depth)])
        self.output_norm = nn.LayerNorm(config.hidden, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden, 2 * config.hidden))
        self.output_projection = nn.Linear(config.hidden, 3 * config.patch_size**2)
        # Zero initial velocity stabilizes flow training. The head learns on the
        # first update; gradients reach the random backbone on subsequent updates.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _validate_inputs(self, video: Tensor, time: Tensor, captions: Sequence[str]) -> None:
        if not isinstance(video, Tensor) or video.ndim != 5:
            raise ValueError("video must be a tensor with shape [B, 3, T, H, W]")
        expected = (3, self.config.frames, self.config.height, self.config.width)
        if video.shape[0] == 0 or tuple(video.shape[1:]) != expected:
            raise ValueError(f"video must have non-empty batch and shape [B, {expected}], got {tuple(video.shape)}")
        if not video.is_floating_point():
            raise TypeError("video must have a floating-point dtype")
        if video.device != self.patch_embed.weight.device:
            raise ValueError("video and model must be on the same device")
        if (
            video.dtype != self.patch_embed.weight.dtype
            and not torch.is_autocast_enabled(video.device.type)
        ):
            raise ValueError("video and model must use the same dtype outside autocast")
        if not torch.isfinite(video).all().item():
            raise ValueError("video must contain only finite values")
        if not isinstance(time, Tensor) or time.shape != (video.shape[0],):
            raise ValueError("time must be a tensor with shape [B]")
        if not time.is_floating_point():
            raise TypeError("time must have a floating-point dtype")
        if time.device != video.device:
            raise ValueError("time and video must be on the same device")
        if not torch.isfinite(time).all().item() or ((time < 0) | (time > 1)).any().item():
            raise ValueError("time must contain finite values in [0, 1]")
        if isinstance(captions, (str, bytes)) or not isinstance(captions, Sequence):
            raise TypeError("captions must be a sequence of strings")
        if len(captions) != video.shape[0]:
            raise ValueError("the number of captions must equal the video batch size")

    def _velocity(self, video: Tensor, time: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        batch = video.shape[0]
        config = self.config
        token_count = (~mask).sum(dim=1, keepdim=True)
        pooled_text = context.sum(dim=1) / token_count.to(context.dtype)
        time_features = _time_features(time, config.hidden).to(self.time_embedding[0].weight.dtype)
        condition = self.time_embedding(time_features) + pooled_text
        tokens = self.patch_embed(video).flatten(3).permute(0, 2, 3, 1)
        tokens = tokens + self.spatial_position + self.temporal_position
        for block in self.blocks:
            tokens = block(tokens, condition, context, mask)
        shift, scale = self.output_modulation(condition).chunk(2, dim=-1)
        tokens = self.output_norm(tokens) * (1 + scale[:, None, None]) + shift[:, None, None]
        patches = self.output_projection(tokens)
        size = config.patch_size
        patches = patches.reshape(
            batch, config.frames, config.height // size, config.width // size, 3, size, size
        )
        return patches.permute(0, 4, 1, 2, 5, 3, 6).reshape(
            batch, 3, config.frames, config.height, config.width
        )

    def forward(self, video: Tensor, time: Tensor, captions: Sequence[str]) -> Tensor:
        """Return velocity; training pairs use x_t=(1-t)*noise+t*data, target=data-noise."""
        self._validate_inputs(video, time, captions)
        context, mask = self.text_encoder(captions)
        return self._velocity(video, time, context, mask)

    def sample(
        self,
        captions: Sequence[str],
        *,
        steps: int = 30,
        guidance_scale: float = 3.0,
        seed: int = 0,
        device: torch.device | str | None = None,
        method: str = "heun",
    ) -> Tensor:
        return sample(
            self, captions, steps=steps, guidance_scale=guidance_scale,
            seed=seed, device=device, method=method,
        )


@torch.inference_mode()
def sample(
    model: TextVideoModel,
    captions: Sequence[str],
    *,
    steps: int = 30,
    guidance_scale: float = 3.0,
    seed: int = 0,
    device: torch.device | str | None = None,
    method: str = "heun",
) -> Tensor:
    """Integrate noise -> video and return pixels in [-1, 1], without moving model.

    The seed controls a local RNG; reproducibility across devices additionally
    depends on their kernels. Heun uses two velocity evaluations per step.
    Classifier-free guidance requires null-caption examples during training.
    """
    if not isinstance(model, TextVideoModel):
        raise TypeError("model must be a TextVideoModel")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or not math.isfinite(guidance_scale)
        or guidance_scale < 0
    ):
        raise ValueError("guidance_scale must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if method not in ("euler", "heun"):
        raise ValueError("method must be 'euler' or 'heun'")
    model_device = model.patch_embed.weight.device
    if device is not None:
        requested = torch.device(device)
        if requested.type == model_device.type and requested.index is None:
            requested = model_device
        if requested != model_device:
            raise ValueError("sample device must match model device; move the model explicitly first")
    # Validate before concatenating so a bare string cannot become individual prompts.
    tokenize_captions(captions, max_text_bytes=model.config.max_text_bytes)
    captions = list(captions)
    batch = len(captions)
    cfg = guidance_scale not in (0, 1)
    conditioning = [""] * batch + captions if cfg else ([""] * batch if guidance_scale == 0 else captions)
    config = model.config
    was_training = model.training
    model.eval()
    try:
        context, mask = model.text_encoder(conditioning)
        generator = torch.Generator(device=model_device).manual_seed(seed)
        state = torch.randn(
            (batch, 3, config.frames, config.height, config.width),
            generator=generator, device=model_device, dtype=model.patch_embed.weight.dtype,
        )

        def velocity(value: Tensor, current_time: float) -> Tensor:
            if not torch.isfinite(value).all().item():
                raise FloatingPointError("sampling produced a non-finite video state")
            inputs = torch.cat((value, value), dim=0) if cfg else value
            time = torch.full((inputs.shape[0],), current_time, device=model_device, dtype=value.dtype)
            result = model._velocity(inputs, time, context, mask)
            if cfg:
                unconditional, conditional = result.chunk(2, dim=0)
                result = unconditional + guidance_scale * (conditional - unconditional)
            if not torch.isfinite(result).all().item():
                raise FloatingPointError("sampling produced a non-finite flow velocity")
            return result

        step_size = 1.0 / steps
        for index in range(steps):
            start = velocity(state, index / steps)
            prediction = state + step_size * start
            if method == "heun":
                end = velocity(prediction, (index + 1) / steps)
                state = state + 0.5 * step_size * (start + end)
            else:
                state = prediction
        if not torch.isfinite(state).all().item():
            raise FloatingPointError("sampling produced a non-finite final video")
        return state.clamp(-1, 1)
    finally:
        model.train(was_training)
