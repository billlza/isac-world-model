"""Acquisition time and availability time are deliberately different fields."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .validation import finite, integer


@dataclass(frozen=True)
class Observation:
    uid: int
    modality: str
    capture_s: float
    available_s: float
    payload: np.ndarray = field(repr=False, compare=False)
    action: int = 0

    def __post_init__(self) -> None:
        integer(self.uid, "uid")
        integer(self.action, "action", 0, 2)
        if self.modality not in {"vision", "radio"}:
            raise ValueError("Invalid observation identity/modality")
        finite(self.capture_s, "capture_s")
        finite(self.available_s, "available_s")
        if self.available_s < self.capture_s:
            raise ValueError("An observation cannot arrive before acquisition")
        if self.modality == "radio" and self.action not in (1, 2):
            raise ValueError("Radio observations require quick=1 or precise=2 action")
        if self.modality == "vision" and self.action != 0:
            raise ValueError("Vision action must be zero")
        original = np.asarray(self.payload)
        expected = (4,) if self.modality == "radio" else None
        if expected and original.shape != expected:
            raise ValueError("Radio payload must have shape (4,)")
        if self.modality == "vision" and (
            original.ndim != 3
            or original.shape[0] != 3
            or original.shape[1] != original.shape[2]
            or not 16 <= original.shape[1] <= 1024
        ):
            raise ValueError("Vision payload must have shape (3, N, N), 16 <= N <= 1024")
        value = np.array(original, dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("Non-finite observation payload")
        value.setflags(write=False)
        object.__setattr__(self, "payload", value)


class EventQueue:
    """Bounded, deterministic arrival queue; duplicate IDs are rejected.

    IDs remain recorded for the lifetime of one episode. Instantiate a new queue
    for another episode. Overflow raises rather than silently dropping evidence.
    """

    def __init__(self, capacity: int = 4096, max_ids: int = 1_000_000):
        integer(capacity, "capacity", 1, 1_000_000)
        integer(max_ids, "max_ids", capacity, 10_000_000)
        self.max_ids = max_ids
        self.capacity = capacity
        self._heap: list[tuple[float, int, Observation]] = []
        self._seen: set[int] = set()
        self.now = 0.0

    def push(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise TypeError("Expected Observation")
        if observation.uid in self._seen:
            raise ValueError("Duplicate observation ID")
        if observation.available_s < self.now:
            raise ValueError("Cannot insert an event in the already-released past")
        if len(self._seen) >= self.max_ids:
            raise OverflowError("Episode evidence-ID budget exhausted; rotate episode explicitly")
        if len(self._heap) >= self.capacity:
            raise OverflowError("Observation queue is full")
        heapq.heappush(self._heap, (observation.available_s, observation.uid, observation))
        self._seen.add(observation.uid)

    def release(self, now: float) -> list[Observation]:
        finite(now, "now")
        if now < self.now:
            raise ValueError("Release time must be finite and monotonic")
        self.now = now
        ready: list[Observation] = []
        while self._heap and self._heap[0][0] <= now + 1e-10:
            ready.append(heapq.heappop(self._heap)[2])
        return ready

    @property
    def next_available_s(self) -> float | None:
        return None if not self._heap else self._heap[0][0]

    def __len__(self) -> int:
        return len(self._heap)


class SnapshotBuffer:
    """Latest-acquisition snapshot, not an exact OOSM Bayesian smoother.

    Newer-arriving but older-acquired packets are recorded as ignored. Each packet
    can trigger an encoder update at most once. Full OOSM replay is a baseline.
    """

    def __init__(self, image_size: int, max_ids: int = 1_000_000):
        integer(image_size, "image_size", 16, 1024)
        integer(max_ids, "max_ids", 1)
        self.max_ids = max_ids
        self._seen: set[int] = set()
        self.now = 0.0
        self.image_size = image_size
        self.latest: dict[str, Observation] = {}
        self.ignored_older = 0

    def update(self, ready: list[Observation], now: float, dt: float) -> dict[str, Any]:
        finite(now, "now")
        finite(dt, "dt", positive=True)
        if now < self.now:
            raise ValueError("Snapshot time must be monotonic")
        ids = [o.uid for o in ready]
        if len(set(ids)) != len(ids) or self._seen.intersection(ids):
            raise ValueError("Duplicate evidence in snapshot")
        if len(self._seen) + len(ids) > self.max_ids:
            raise OverflowError("Snapshot evidence-ID budget exhausted")
        # Validate a whole batch before mutating state.
        for o in ready:
            if o.available_s > now + 1e-10:
                raise ValueError("Future observation leakage")
            if o.modality == "vision" and o.payload.shape != (3, self.image_size, self.image_size):
                raise ValueError("Vision dimensions do not match snapshot")
        self._seen.update(ids)
        self.now = now
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
            if vision is None
            else vision.payload.copy(),
            "radio": np.zeros(4, np.float32) if radio is None else radio.payload.copy(),
            "valid": np.array([vision is not None, radio is not None], np.float32),
            "new": new,
            "age": np.array(
                [0.0 if o is None else max(0.0, now - o.capture_s) for o in (vision, radio)], np.float32
            ),
            "dt": np.array([dt], np.float32),
        }
