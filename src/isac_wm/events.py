"""Acquisition time and availability time are deliberately different fields."""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import math
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Observation:
    uid: int
    modality: str
    capture_s: float
    available_s: float
    payload: np.ndarray = field(repr=False, compare=False)
    action: int = 0

    def __post_init__(self) -> None:
        if self.uid < 0 or self.modality not in {"vision", "radio"}:
            raise ValueError("Invalid observation identity/modality")
        if not all(math.isfinite(x) and x >= 0 for x in (self.capture_s, self.available_s)):
            raise ValueError("Times must be finite and non-negative")
        if self.available_s < self.capture_s:
            raise ValueError("An observation cannot arrive before acquisition")
        if self.modality == "radio" and self.action not in (1, 2):
            raise ValueError("Radio observations require quick=1 or precise=2 action")
        if self.modality == "vision" and self.action != 0:
            raise ValueError("Vision action must be zero")
        value = np.array(self.payload, dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("Non-finite observation payload")
        value.setflags(write=False)
        object.__setattr__(self, "payload", value)


class EventQueue:
    """Bounded, deterministic arrival queue; duplicate IDs are rejected.

    IDs remain recorded for the lifetime of one episode. Instantiate a new queue
    for another episode. Overflow raises rather than silently dropping evidence.
    """
    def __init__(self, capacity: int = 4096):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._heap: list[tuple[float, int, Observation]] = []
        self._seen: set[int] = set()
        self.now = 0.0

    def push(self, observation: Observation) -> None:
        if observation.uid in self._seen:
            raise ValueError("Duplicate observation ID")
        if observation.available_s < self.now:
            raise ValueError("Cannot insert an event in the already-released past")
        if len(self._heap) >= self.capacity:
            raise OverflowError("Observation queue is full")
        heapq.heappush(self._heap, (observation.available_s, observation.uid, observation))
        self._seen.add(observation.uid)

    def release(self, now: float) -> list[Observation]:
        if not math.isfinite(now) or now < self.now:
            raise ValueError("Release time must be finite and monotonic")
        self.now = now
        ready: list[Observation] = []
        while self._heap and self._heap[0][0] <= now + 1e-10:
            ready.append(heapq.heappop(self._heap)[2])
        return ready

    def __len__(self) -> int:
        return len(self._heap)


class SnapshotBuffer:
    """Latest-acquisition snapshot, not an exact OOSM Bayesian smoother.

    Newer-arriving but older-acquired packets are recorded as ignored. Each packet
    can trigger an encoder update at most once. Full OOSM replay is a baseline.
    """
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.latest: dict[str, Observation] = {}
        self.ignored_older = 0

    def update(self, ready: list[Observation], now: float, dt: float) -> dict[str, Any]:
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("dt must be positive")
        new = np.zeros(2, dtype=np.float32)
        for obs in ready:
            if obs.available_s > now + 1e-10:
                raise ValueError("Future observation leakage")
            idx = 0 if obs.modality == "vision" else 1
            old = self.latest.get(obs.modality)
            if old is None or (obs.capture_s, obs.uid) > (old.capture_s, old.uid):
                self.latest[obs.modality] = obs
                new[idx] = 1.0
            else:
                self.ignored_older += 1
        vision = self.latest.get("vision")
        radio = self.latest.get("radio")
        return {
            "image": np.zeros((3, self.image_size, self.image_size), np.float32)
                if vision is None else vision.payload.copy(),
            "radio": np.zeros(4, np.float32) if radio is None else radio.payload.copy(),
            "valid": np.array([vision is not None, radio is not None], np.float32),
            "new": new,
            "age": np.array([0.0 if o is None else max(0.0, now-o.capture_s)
                             for o in (vision, radio)], np.float32),
            "dt": np.array([dt], np.float32),
        }
