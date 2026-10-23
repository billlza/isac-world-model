"""Randomly initialized visual-radio model with explicit time and modality memory.

Design proposal, not an established novel architecture or calibrated Bayes filter.
No downloaded checkpoints, model teachers or external pretrained features.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 48
    hidden_dim: int = 96

    def __post_init__(self) -> None:
        if self.feature_dim < 8 or self.hidden_dim < 8:
            raise ValueError("Model dimensions must be >= 8")

    def to_dict(self) -> dict:
        return asdict(self)


def gaussian_nll(mean: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """Per-example diagonal Gaussian NLL, omitting the common log(2*pi)/2."""
    return .5*(logvar+(target-mean).square()*torch.exp(-logvar)).mean(dim=-1)


def radio_from_state(state: Tensor) -> Tensor:
    """Physical proxy, using x/y normalized by room and vx/vy normalized by 2."""
    p, v = state[..., :2], state[..., 2:]
    norm = p.square().sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return torch.cat([norm/(2**.5), p/norm, (p*v).sum(-1, keepdim=True)/norm], -1)


class EventBeliefModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        f, h = self.config.feature_dim, self.config.hidden_dim
        self.visual = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, f), nn.SiLU())
        self.radio_encoder = nn.Sequential(nn.Linear(4, f), nn.SiLU(), nn.Linear(f, f))
        self.visual_update = nn.GRUCell(f+1, f)
        self.radio_update = nn.GRUCell(f+1, f)
        self.dynamics = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.fusion = nn.GRUCell(2*f+7, h)
        self.state_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 8))
        self.action_embedding = nn.Embedding(3, 8)
        self.observation_head = nn.Sequential(nn.Linear(h+8, h), nn.SiLU(), nn.Linear(h, 8))

    def initial(self, batch_size: int, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        return (torch.zeros(batch_size, self.config.hidden_dim, device=device),
                torch.zeros(batch_size, self.config.feature_dim, device=device),
                torch.zeros(batch_size, self.config.feature_dim, device=device))

    def decode_state(self, h: Tensor) -> dict[str, Tensor]:
        raw = self.state_head(h)
        mean = torch.cat([raw[..., :2].sigmoid(), raw[..., 2:4].tanh()], -1)
        logvar = -7.0+8.0*raw[..., 4:].sigmoid()
        return {"mean": mean, "logvar": logvar}

    def forecast(self, h: Tensor, horizon: float | Tensor) -> dict[str, Tensor]:
        if isinstance(horizon, (int, float)):
            if horizon < 0:
                raise ValueError("Forecast horizon must be non-negative")
            dt = torch.full((*h.shape[:-1], 1), float(horizon), device=h.device)
        else:
            dt = horizon
        # Sensing actions are NOT arguments to exogenous environment dynamics.
        return self.decode_state(h + dt*self.dynamics(h))

    def predict_observation(self, h: Tensor, action: Tensor) -> dict[str, Tensor]:
        raw = self.observation_head(torch.cat([h, self.action_embedding(action)], -1))
        base = radio_from_state(self.decode_state(h)["mean"])
        # Residual around known geometric measurement mapping. Its size is bounded.
        return {"mean": base+.15*raw[..., :4].tanh(),
                "logvar": -9.0+9.0*raw[..., 4:].sigmoid()}

    def step(self, x: dict[str, Tensor], memory: tuple[Tensor, Tensor, Tensor]
             ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor]]:
        h, pv, pr = memory
        new = x["new"]*x["valid"]
        age = x["age"].clamp(0, 20)
        dt = x["dt"].clamp(0, 10)
        visual = self.visual(x["image"])
        radio = self.radio_encoder(x["radio"])
        pv_new = self.visual_update(torch.cat([visual, age[:, :1]], -1), pv)
        pr_new = self.radio_update(torch.cat([radio, age[:, 1:]], -1), pr)
        pv = torch.where(new[:, :1].bool(), pv_new, pv)
        pr = torch.where(new[:, 1:].bool(), pr_new, pr)
        prior = h+dt*self.dynamics(h)
        features = torch.cat([pv, pr, age, x["valid"], new, dt], -1)
        updated = self.fusion(features, prior)
        h = torch.where(new.any(-1, keepdim=True), updated, prior)
        out = self.decode_state(h)
        out["hidden"] = h
        return out, (h, pv, pr)

    def forward(self, batch: dict[str, Tensor], forecast_s: float = .3) -> dict[str, Tensor]:
        b, t = batch["image"].shape[:2]
        memory = self.initial(b, batch["image"].device)
        outputs: dict[str, list[Tensor]] = {k: [] for k in (
            "mean", "logvar", "future_mean", "future_logvar", "obs_mean", "obs_logvar")}
        for k in range(t):
            inputs = {key: batch[key][:, k] for key in ("image", "radio", "age", "valid", "new", "dt")}
            out, memory = self.step(inputs, memory)
            future = self.forecast(out["hidden"], forecast_s)
            obs = self.predict_observation(out["hidden"], batch["action_target"][:, k])
            for key in ("mean", "logvar"):
                outputs[key].append(out[key])
                outputs["future_"+key].append(future[key])
                outputs["obs_"+key].append(obs[key])
        return {key: torch.stack(values, dim=1) for key, values in outputs.items()}


def training_loss(output: dict[str, Tensor], batch: dict[str, Tensor], horizon_steps: int = 3
                  ) -> tuple[Tensor, dict[str, float]]:
    target = batch["state_target"]
    if not 0 < horizon_steps < target.shape[1]:
        raise ValueError("horizon_steps must be inside the sequence")
    current = gaussian_nll(output["mean"], output["logvar"], target).mean()
    future = gaussian_nll(output["future_mean"][:, :-horizon_steps],
                          output["future_logvar"][:, :-horizon_steps],
                          target[:, horizon_steps:]).mean()
    mask = batch["radio_target_valid"]
    obs_losses = gaussian_nll(output["obs_mean"], output["obs_logvar"], batch["radio_target"])
    observation = (obs_losses*mask).sum()/mask.sum().clamp_min(1)
    loss = current+.5*future+.15*observation
    return loss, {"loss": float(loss.detach()), "state_nll": float(current.detach()),
                  "future_nll": float(future.detach()), "observation_nll": float(observation.detach())}
