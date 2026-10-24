"""Versioned execution-time contract; no hidden wall-clock simulation assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .validation import finite, integer

PROTOCOL_VERSION = "streaming-v1"


@dataclass(frozen=True)
class RuntimeConfig:
    """Single non-preemptive worker, latest snapshot when free; no work backlog.

    `reference` retains v0.2's instantaneous logical inference for comparison.
    `fixed` delays publication/actions using a declared service time.
    `measured` uses synchronized host timing: diagnostic, not reproducible.
    Publication and sensing actuation are rounded UP to the next simulator tick.
    Sensor capture continues while inference is busy. Inter-publication states
    use constant-velocity propagation and explicit white-acceleration covariance
    inflation, a model assumption, not a proven physical bound.
    """

    mode: str = "reference"
    compute_ms: float = 0.0
    queue_capacity: int = 4096
    output_acceleration_noise: float = 0.8  # m^2/s^3, covariance model parameter

    def __post_init__(self) -> None:
        if self.mode not in {"reference", "fixed", "measured"}:
            raise ValueError("Unknown latency mode")
        finite(self.compute_ms, "compute_ms")
        if self.compute_ms > 3_600_000:
            raise ValueError("compute_ms exceeds the one-hour experiment bound")
        if self.mode != "fixed" and self.compute_ms != 0:
            raise ValueError("compute_ms is only valid in fixed mode")
        integer(self.queue_capacity, "queue_capacity", 1, 1_000_000)
        finite(self.output_acceleration_noise, "output_acceleration_noise")
        if self.mode == "reference" and self.queue_capacity != 4096:
            raise ValueError("Reference mode has a fixed queue contract")

    def to_dict(self) -> dict:
        return {"protocol_version": PROTOCOL_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeConfig":
        if not isinstance(data, dict) or data.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported runtime protocol")
        return cls(**{k: v for k, v in data.items() if k != "protocol_version"})

    def service_seconds(self, measured_seconds: float) -> float:
        finite(measured_seconds, "measured_seconds")
        if self.mode == "reference":
            return 0.0
        return self.compute_ms / 1000 if self.mode == "fixed" else measured_seconds
