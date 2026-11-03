"""Explicit verification protocol for the bounded two-candidate experiment.

This state machine is an experimental decision rule, not a calibrated test of
independent evidence. Repeated views can share sensor and RF-anchor errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

PLANNERS = ("verification-aware", "legacy-voi")


@dataclass(frozen=True)
class EvidenceState:
    accepted: int = 0
    informative: int = 0
    supporting_informative: int = 0
    contradictions: int = 0
    contradiction_streak: int = 0
    rejected: bool = False
    last_observed_lateral_m: float | None = None

    def observe(self, *, compatible: bool, informative: bool, lateral_m: float) -> EvidenceState:
        """Advance only for a new, valid, arrived detection.

        A missing detection is not evidence for or against the candidates and
        must not call this method. Rejection is latched for this candidate set.
        A contradiction clears confirmation support but does not erase history.
        """
        if type(compatible) is not bool or type(informative) is not bool or not math.isfinite(lateral_m):
            raise ValueError("Invalid evidence classification or viewpoint")
        streak = 0 if compatible else self.contradiction_streak + 1
        support = self.supporting_informative + int(informative) if compatible else 0
        return replace(
            self,
            accepted=self.accepted + 1,
            informative=self.informative + int(informative),
            supporting_informative=support,
            contradictions=self.contradictions + int(not compatible),
            contradiction_streak=streak,
            rejected=self.rejected or streak >= 2,
            last_observed_lateral_m=float(lateral_m),
        )

    @property
    def ready_to_confirm(self) -> bool:
        return not self.rejected and self.contradiction_streak == 0 and self.supporting_informative >= 2

    @property
    def pending_verification(self) -> str | None:
        if self.rejected:
            return None
        if self.contradiction_streak:
            return "contradiction_verification"
        if self.supporting_informative == 1:
            return "confirmation_verification"
        return None


def verification_action(state: EvidenceState, proposals: list[dict]) -> dict | None:
    """Select from already feasible actions without altering their VoI score.

    Confirmation requires a geometrically discriminating view. A contradiction
    can be checked even where the two candidates predict the same bearing. View
    separation is a diversity heuristic, not evidence independence or an optimal
    value-of-information calculation. Same-view fallback remains explicitly
    marked if travel or arrival constraints make a different view unavailable.
    """
    reason = state.pending_verification
    if reason is None or state.last_observed_lateral_m is None:
        return None
    eligible = [
        p for p in proposals if reason == "contradiction_verification" or p["geometrically_informative"]
    ]
    if not eligible:
        return None
    previous = state.last_observed_lateral_m
    selected = max(
        eligible,
        key=lambda p: (abs(p["lateral_m"] - previous), -p["available_s"], -p["travel_m"]),
    )
    separation = abs(selected["lateral_m"] - previous)
    return {
        **selected,
        "selection_reason": reason,
        "verification_view_separation_m": separation,
        "verification_same_view": separation <= 1e-9,
    }
