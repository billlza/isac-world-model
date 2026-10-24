"""Small, explicit validators at public IO boundaries (no silent NaN clamping)."""

from __future__ import annotations

import math
from numbers import Integral, Real


def integer(value: int, name: str, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def finite(value: float, name: str, minimum: float = 0.0, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (positive and value <= minimum):
        raise ValueError(f"{name} is outside its permitted range")
    return float(value)
