"""Train-from-scratch event model with a physical state and bounded retrieval.

The covariance is a learned model quantity, NOT a calibrated error guarantee.
The Cartesian measurement lift assumes the documented orthographic toy scene.
Attention is an explicit ablation, not an automatic accuracy/safety improvement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .model import EventBeliefModel, ModelConfig, radio_from_state
from .validation import finite, integer

LATENT_STATE_NORM_EPS = 1e-5
MAX_FORECAST_STEPS = 256


@dataclass(frozen=True)
class StructuredConfig:
    feature_dim: int = 48
    hidden_dim: int = 96
    architecture: str = "structured-attention"
    memory_size: int = 16
    heads: int = 4
    room_m: float = 10.0
    latent_state_norm: str = "none"
    visual_channels: int = 16
    visual_depth: int = 1
    # Per-axis physical bound; the velocity state is normalized by 2 m/s.
    max_acceleration_mps2: float = 0.5
    acceleration_init_scale: float = 1.0
    # Zero retains the historical fixed-latent, single-step forecast exactly.
    forecast_step_s: float = 0.0

    def __post_init__(self) -> None:
        ModelConfig(self.feature_dim, self.hidden_dim)
        integer(self.memory_size, "memory_size", 1, 256)
        integer(self.heads, "heads", 1, 16)
        finite(self.room_m, "room_m", positive=True)
        integer(self.visual_channels, "visual_channels", 1, 256)
        integer(self.visual_depth, "visual_depth", 1, 8)
        finite(self.max_acceleration_mps2, "max_acceleration_mps2", positive=True)
        if self.max_acceleration_mps2 > 100:
            raise ValueError("max_acceleration_mps2 must be at most 100 m/s^2")
        finite(self.acceleration_init_scale, "acceleration_init_scale", positive=True)
        finite(self.forecast_step_s, "forecast_step_s")
        if self.acceleration_init_scale > 1:
            raise ValueError("acceleration_init_scale must be in (0, 1]")
        if self.architecture not in {"structured", "structured-attention"}:
            raise ValueError("Unknown structured architecture")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not isinstance(self.latent_state_norm, str) or self.latent_state_norm not in {"none", "layernorm"}:
            raise ValueError("latent_state_norm must be 'none' or 'layernorm'")
        if self.latent_state_norm != "none" and self.architecture != "structured-attention":
            raise ValueError("Latent state normalization requires structured-attention")

    def to_dict(self) -> dict:
        values = asdict(self)
        # Preserve the historical checkpoint/config representation of the control.
        # Missing fields in older checkpoints retain their original architecture
        # and physical acceleration bound, without changing config fingerprints.
        if self.latent_state_norm == "none":
            values.pop("latent_state_norm")
        for name, default in (
            ("visual_channels", 16),
            ("visual_depth", 1),
            ("max_acceleration_mps2", 0.5),
            ("acceleration_init_scale", 1.0),
            ("forecast_step_s", 0.0),
        ):
            if values[name] == default:
                values.pop(name)
        return values


class SpatialEncoder(nn.Module):
    """Learn a spatial likelihood map without a pretrained detector or colour rule."""

    def __init__(self, features: int, channels: int = 16, depth: int = 1):
        super().__init__()
        integer(channels, "visual_channels", 1, 256)
        integer(depth, "visual_depth", 1, 8)
        # Defaults preserve module names, construction order and RNG consumption.
        # Spatial resolution stays fixed; no batch statistics mix future events
        # into earlier encodings when training encodes a whole sequence at once.
        layers = [nn.Conv2d(3, channels, 3, padding=1), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.heat = nn.Conv2d(channels, 1, 1)
        self.context = nn.Linear(channels, features - 3)

    def forward(self, image: Tensor) -> Tensor:
        feature = self.trunk(image)
        scores = self.heat(feature).flatten(1)
        # No ground truth, thresholded red pixels, or external teacher enters here.
        weights = (8 * scores).softmax(-1)
        height, width = image.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=image.device, dtype=image.dtype),
            torch.linspace(0, 1, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        xy = torch.stack([weights @ xx.flatten(), weights @ yy.flatten()], -1)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        entropy = entropy / scores.new_tensor(scores.shape[-1]).log()
        context = self.context((feature.flatten(2) * weights[:, None]).sum(-1))
        return torch.cat([xy, entropy, context], -1)


class StructuredEventModel(EventBeliefModel):
    """Continuous-interval prediction, learned correction, optional event retrieval.

    Memory is caller-owned; no cross-stream state is stored in the module. Only
    valid new evidence is appended. A null token makes empty memory well-defined.
    The ring is bounded by event count, not silently by elapsed time.
    """

    supports_visual_supervision = True

    def __init__(self, config: StructuredConfig):
        nn.Module.__init__(self)
        self.config = config
        f, h = config.feature_dim, config.hidden_dim
        self.visual = SpatialEncoder(f, config.visual_channels, config.visual_depth)
        self.radio_encoder = nn.Sequential(nn.Linear(4, f), nn.SiLU(), nn.Linear(f, f))
        self.fusion = nn.GRUCell(2 * f + 7, h)
        self.dynamics = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.acceleration = nn.Linear(h, 2)
        self.noise = nn.Linear(h, 6)
        self.action_embedding = nn.Embedding(3, 8)
        self.observation_head = nn.Sequential(nn.Linear(h + 8, h), nn.SiLU(), nn.Linear(h, 8))
        self.attentive = config.architecture == "structured-attention"
        if self.attentive:
            self.event_projection = nn.Linear(f + 2, h)
            self.time_projection = nn.Sequential(nn.Linear(2, h), nn.Tanh())
            self.query_norm = nn.LayerNorm(h)
            self.memory_norm = nn.LayerNorm(h)
            self.attention = nn.MultiheadAttention(h, config.heads, dropout=0.0, batch_first=True)
            self.null = nn.Parameter(torch.zeros(1, 1, h))
            self.feedforward = nn.Sequential(
                nn.LayerNorm(h), nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h)
            )
        # This parameter-free boundary neither consumes RNG nor changes state_dict
        # keys. The control skips the operation entirely, retaining its exact path.
        self.latent_state_norm = (
            nn.LayerNorm(h, eps=LATENT_STATE_NORM_EPS, elementwise_affine=False)
            if config.latent_state_norm == "layernorm"
            else None
        )
        # Apply only to random initialization, after every module is constructed.
        # This consumes no RNG and changes neither other weights nor the learned
        # acceleration range. Loading a checkpoint replaces these initial values.
        if config.acceleration_init_scale != 1.0:
            with torch.no_grad():
                self.acceleration.weight.mul_(config.acceleration_init_scale)
                self.acceleration.bias.mul_(config.acceleration_init_scale)

    def initial(self, batch_size: int, device: torch.device | str) -> tuple[Tensor, ...]:
        integer(batch_size, "batch_size", 1)
        f, h, w = self.config.feature_dim, self.config.hidden_dim, self.config.memory_size
        dtype = next(self.parameters()).dtype

        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=dtype)

        mean = zeros(batch_size, 4)
        mean[:, :2] = 0.5
        covariance = (
            torch.diag(mean.new_tensor([1 / 12, 1 / 12, 0.25, 0.25])).expand(batch_size, 4, 4).clone()
        )
        return (
            zeros(batch_size, h),
            zeros(batch_size, f),
            zeros(batch_size, f),
            mean,
            covariance,
            zeros(batch_size, w, h),
            zeros(batch_size, w, 2),
            torch.zeros(batch_size, w, device=device, dtype=torch.bool),
            zeros(batch_size, 1),
        )

    def _predict(self, h: Tensor, mean: Tensor, covariance: Tensor, dt: Tensor, *, diagnostics=None):
        ratio = 2 / self.config.room_m
        acceleration_logits = self.acceleration(h)
        # Velocity is normalized by 2 m/s; per-axis acceleration uses that scale.
        # The historical 0.5 m/s^2 bound therefore retains exactly 0.25 here.
        acc = (self.config.max_acceleration_mps2 / 2) * acceleration_logits.tanh()
        position = mean[:, :2] + ratio * (dt * mean[:, 2:] + 0.5 * dt.square() * acc)
        velocity = mean[:, 2:] + dt * acc
        predicted = torch.cat([position, velocity], -1)
        f = torch.eye(4, device=h.device, dtype=h.dtype).expand(h.shape[0], 4, 4).clone()
        f[:, 0, 2] = f[:, 1, 3] = dt[:, 0] * ratio
        noise_logits = self.noise(h)
        q = 1e-5 + 0.02 * noise_logits[:, :2].sigmoid()
        q = torch.cat([q * ratio**2, q], -1) * dt
        predicted_cov = f @ covariance @ f.transpose(-1, -2) + torch.diag_embed(q)
        if diagnostics is not None:
            diagnostics["acceleration_logits"] = acceleration_logits.detach().clone()
            diagnostics["noise_logits"] = noise_logits.detach().clone()
        return predicted, predicted_cov

    def _retrieve(self, h, encodings, new, age, clock, tokens, stamps, used):
        for modality in range(2):
            identity = h.new_zeros((h.shape[0], 2))
            identity[:, modality] = 1
            token = self.event_projection(torch.cat([encodings[modality], identity], -1))
            stamp = torch.cat([clock - age[:, modality : modality + 1], clock], -1)
            active = new[:, modality].bool()
            tokens = torch.where(active[:, None, None], torch.cat([tokens[:, 1:], token[:, None]], 1), tokens)
            stamps = torch.where(active[:, None, None], torch.cat([stamps[:, 1:], stamp[:, None]], 1), stamps)
            used = torch.where(active[:, None], torch.cat([used[:, 1:], active[:, None]], 1), used)
        elapsed = (clock[:, None] - stamps).clamp_min(0)
        keys = tokens + self.time_projection(torch.log1p(elapsed))
        keys = torch.cat([self.null.expand(h.shape[0], -1, -1), keys], 1)
        padding = torch.cat([torch.zeros_like(used[:, :1]), ~used], 1)
        keys = self.memory_norm(keys)
        context, _ = self.attention(
            self.query_norm(h)[:, None], keys, keys, key_padding_mask=padding, need_weights=False
        )
        out = h + context[:, 0]
        return out + self.feedforward(out), tokens, stamps, used

    @staticmethod
    def _correct(mean, covariance, measurement, variance, age, active, ratio, *, diagnostics=None):
        # A delayed position measures [I, -age*scale*I] of current kinematic state.
        h = mean.new_zeros((mean.shape[0], 2, 4))
        h[:, 0, 0] = h[:, 1, 1] = 1
        h[:, 0, 2] = h[:, 1, 3] = -age[:, 0] * ratio
        r = torch.diag_embed(variance)
        residual = measurement - (h @ mean.unsqueeze(-1)).squeeze(-1)
        innovation = h @ covariance @ h.transpose(-1, -2) + r
        gain = torch.linalg.solve(innovation, h @ covariance).transpose(-1, -2)
        updated_mean = mean + (gain @ residual.unsqueeze(-1)).squeeze(-1)
        ikh = torch.eye(4, device=mean.device, dtype=mean.dtype) - gain @ h
        updated_cov = ikh @ covariance @ ikh.transpose(-1, -2) + gain @ r @ gain.transpose(-1, -2)
        updated_cov = (updated_cov + updated_cov.transpose(-1, -2)) * 0.5
        result = (
            torch.where(active, updated_mean, mean),
            torch.where(active[:, :, None], updated_cov, covariance),
        )
        if diagnostics is not None:
            # The observer receives detached COPIES, never aliases of model state,
            # inputs, or an autograd graph. No quantity here changes the correction.
            values = {
                "active": active,
                "age_s": age,
                "measurement_xy": measurement,
                "predicted_measurement_xy": (h @ mean.unsqueeze(-1)).squeeze(-1),
                "measurement_variance_xy": variance,
                "innovation_xy": residual,
                "innovation_covariance_xy": innovation,
                "gain": gain,
                "state_mean_before": mean,
                "state_mean_after": result[0],
                "applied_state_correction": result[0] - mean,
            }
            diagnostics.update({key: value.detach().clone() for key, value in values.items()})
        return result

    def _pack(self, h, mean, covariance):
        return torch.cat([h, mean, covariance.flatten(1)], -1)

    def _unpack(self, packed):
        h = self.config.hidden_dim
        return packed[:, :h], packed[:, h : h + 4], packed[:, h + 4 :].reshape(-1, 4, 4)

    def decode_state(self, packed: Tensor) -> dict[str, Tensor]:
        _, mean, covariance = self._unpack(packed)
        return {
            "mean": mean,
            "logvar": covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-10).log(),
            "covariance": covariance,
        }

    def forecast(self, packed: Tensor, horizon: float | Tensor) -> dict[str, Tensor]:
        """Predict without evidence, preserving the caller's packed belief.

        With ``forecast_step_s=0``, only the physical state evolves, exactly as
        in historical checkpoints. A positive step enables bounded internal
        latent/physical transitions in the same order as an observation-free
        ``step()``. No cached token is reinterpreted as fresh evidence. Each row
        stops at its own horizon; zero-time rows are never renormalized.

        Covariance remains the model's process-noise approximation, not a
        calibrated guarantee or a Jacobian propagation of latent uncertainty.
        """
        h, mean, covariance = self._unpack(packed)
        if isinstance(horizon, Tensor):
            if (
                horizon.shape not in {torch.Size([]), torch.Size([h.shape[0], 1])}
                or horizon.is_complex()
                or horizon.dtype == torch.bool
                or not bool(torch.isfinite(horizon).all())
                or bool((horizon < 0).any())
            ):
                raise ValueError("horizon must be finite, non-negative and batch-compatible")
            dt = horizon.to(h).expand(h.shape[0], 1)
        else:
            value = finite(horizon, "horizon")
            if value > torch.finfo(h.dtype).max:
                raise ValueError("horizon is not representable in the model dtype")
            dt = h.new_full((h.shape[0], 1), value)
        if not bool(torch.isfinite(dt).all()):
            raise ValueError("horizon is not representable in the model dtype")
        if self.config.forecast_step_s == 0:
            mean, covariance = self._predict(h, mean, covariance, dt)
            return self.decode_state(self._pack(h, mean, covariance))

        step_s = h.new_tensor(self.config.forecast_step_s)
        if not bool(torch.isfinite(step_s)) or not bool(step_s > 0):
            raise ValueError("forecast_step_s is not representable in the model dtype")
        steps = dt.max() / step_s
        if not bool(torch.isfinite(steps)) or bool(steps > MAX_FORECAST_STEPS):
            raise ValueError(f"forecast requires more than {MAX_FORECAST_STEPS} internal steps")
        for index in range(int(torch.ceil(steps).item())):
            interval = torch.minimum((dt - index * step_s).clamp_min(0), step_s)
            active = interval > 0
            proposed_h = h + interval * self.dynamics(h)
            if self.latent_state_norm is not None:
                proposed_h = self.latent_state_norm(proposed_h)
            h = torch.where(active, proposed_h, h)
            proposed_mean, proposed_covariance = self._predict(h, mean, covariance, interval)
            mean = torch.where(active, proposed_mean, mean)
            covariance = torch.where(active[:, :, None], proposed_covariance, covariance)
        return self.decode_state(self._pack(h, mean, covariance))

    def predict_observation(self, packed: Tensor, action: Tensor) -> dict[str, Tensor]:
        h, mean, _ = self._unpack(packed)
        raw = self.observation_head(torch.cat([h, self.action_embedding(action)], -1))
        return {
            "mean": radio_from_state(mean) + 0.15 * raw[:, :4].tanh(),
            "logvar": -9 + 9 * raw[:, 4:].sigmoid(),
        }

    def step(self, x, memory, encoded=None, *, diagnostics=None):
        """Advance one snapshot; an optional observer records the unchanged calculation.

        Diagnostic tensors use the model's normalized coordinates. ``active``
        identifies a correction that was actually applied; an inactive row's
        measurement/gain must not be treated as another observation. The caller
        supplies absolute capture/availability/state times separately from age.
        """
        if diagnostics is not None and (not isinstance(diagnostics, dict) or diagnostics):
            raise ValueError("diagnostics must be an empty caller-owned dict")
        h, pv, pr, mean, covariance, tokens, stamps, used, clock = memory
        for name in ("age", "dt", "valid", "new"):
            value = x[name]
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("valid", "new"):
            if not bool(((x[name] == 0) | (x[name] == 1)).all()):
                raise ValueError(f"{name} must be a binary mask")
        new = (x["new"] * x["valid"]).bool()
        age, dt = x["age"], x["dt"]
        clock = clock + dt
        for i, key in enumerate(("image", "radio")):
            if not bool(torch.isfinite(x[key][new[:, i]]).all()):
                raise ValueError(f"Non-finite fresh {key} observation")
        encodings = self.encode_new(x) if encoded is None else encoded
        pv = torch.where(new[:, :1], encodings[0], pv)
        pr = torch.where(new[:, 1:], encodings[1], pr)
        prior_h = h + dt * self.dynamics(h)
        features = torch.cat([pv, pr, age, x["valid"], new.to(h), dt], -1)
        updated_h = self.fusion(features, prior_h)
        fusion_h = updated_h if diagnostics is not None else None
        if self.attentive:
            updated_h, tokens, stamps, used = self._retrieve(
                updated_h, encodings, new, age, clock, tokens, stamps, used
            )
        h = torch.where(new.any(-1, keepdim=True), updated_h, prior_h)
        hidden_pre_norm = h
        if self.latent_state_norm is not None:
            h = self.latent_state_norm(h)
        network = None if diagnostics is None else {}
        mean, covariance = self._predict(h, mean, covariance, dt, diagnostics=network)
        noise = self.noise(h)
        variance = (-10 + 7 * noise[:, 2:].sigmoid()).exp()
        # Entropy downweights diffuse maps (occlusion); not a safety certificate.
        visual_variance = variance[:, :2] + 0.05 * pv[:, 2:3].clamp(0, 1).pow(4)
        safe_radio = torch.where(new[:, 1:], x["radio"], torch.zeros_like(x["radio"]))
        radio_xy = (2**0.5) * safe_radio[:, :1] * safe_radio[:, 1:3]
        for index, modality, measurement, r, a, active in (
            (0, "vision", pv[:, :2], visual_variance, age[:, :1], new[:, :1]),
            (1, "radio", radio_xy, variance[:, 2:], age[:, 1:], new[:, 1:]),
        ):
            observed = None if diagnostics is None else {}
            mean, covariance = self._correct(
                mean,
                covariance,
                measurement,
                r,
                a,
                active,
                2 / self.config.room_m,
                diagnostics=observed,
            )
            if diagnostics is not None:
                observed["valid"] = x["valid"][:, index : index + 1].detach().clone()
                observed["new"] = x["new"][:, index : index + 1].detach().clone()
                if modality == "vision":
                    observed["heatmap_entropy"] = pv[:, 2:3].detach().clone()
                    observed["learned_measurement_variance_xy"] = variance[:, :2].detach().clone()
                diagnostics[modality] = observed
        if diagnostics is not None:
            diagnostics["schema_version"] = 1
            network.update(
                hidden=h.detach().clone(),
                hidden_pre_norm=hidden_pre_norm.detach().clone(),
                prior_hidden=prior_h.detach().clone(),
                fusion_hidden=fusion_h.detach().clone(),
                attention_residual=(updated_h - fusion_h).detach().clone(),
                evidence_update=new.any(-1, keepdim=True).detach().clone(),
            )
            diagnostics["network"] = network
        packed = self._pack(h, mean, covariance)
        out = self.decode_state(packed)
        out["hidden"] = packed
        return out, (h, pv, pr, mean, covariance, tokens, stamps, used, clock)


def model_config_from_dict(values: dict) -> ModelConfig | StructuredConfig:
    if "architecture" not in values:
        for field in (
            "latent_state_norm",
            "visual_channels",
            "visual_depth",
            "max_acceleration_mps2",
            "acceleration_init_scale",
            "forecast_step_s",
        ):
            if field in values:
                raise ValueError(f"{field} is only configured for structured architectures")
    return StructuredConfig(**values) if "architecture" in values else ModelConfig(**values)


def build_model(config: ModelConfig | StructuredConfig | None = None) -> EventBeliefModel:
    return StructuredEventModel(config) if isinstance(config, StructuredConfig) else EventBeliefModel(config)
