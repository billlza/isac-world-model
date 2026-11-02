"""Evidence protocol regressions runnable without Torch or a new environment."""

import unittest

from isac_wm.observation_planning import EvidenceState, verification_action


def proposal(lateral, *, informative=True, available=3.0, travel=0.0, score=-0.01):
    return {
        "lateral_m": lateral,
        "geometrically_informative": informative,
        "available_s": available,
        "travel_m": travel,
        "score": score,
    }


class EvidenceProtocolTests(unittest.TestCase):
    def test_confirmation_requires_two_compatible_informative_observations(self):
        empty = EvidenceState()
        first = empty.observe(compatible=True, informative=True, lateral_m=4)
        self.assertEqual(first.pending_verification, "confirmation_verification")
        self.assertFalse(first.ready_to_confirm)
        second = first.observe(compatible=True, informative=True, lateral_m=-4)
        self.assertTrue(second.ready_to_confirm)
        self.assertIsNone(second.pending_verification)
        self.assertEqual(empty.accepted, 0)

    def test_uninformative_observations_do_not_force_confirmation(self):
        state = EvidenceState()
        for _ in range(3):
            state = state.observe(compatible=True, informative=False, lateral_m=0)
        self.assertFalse(state.ready_to_confirm)
        self.assertIsNone(state.pending_verification)

    def test_separated_contradictions_are_not_a_consecutive_pair(self):
        state = EvidenceState()
        for compatible in (False, True, False):
            state = state.observe(compatible=compatible, informative=True, lateral_m=4)
        self.assertEqual((state.contradictions, state.contradiction_streak), (2, 1))
        self.assertFalse(state.rejected)
        self.assertFalse(state.ready_to_confirm)
        self.assertEqual(state.pending_verification, "contradiction_verification")

    def test_rejection_latches_for_the_existing_candidate_set(self):
        state = EvidenceState()
        for compatible in (False, False, True, True):
            state = state.observe(compatible=compatible, informative=True, lateral_m=4)
        self.assertTrue(state.rejected)
        self.assertFalse(state.ready_to_confirm)
        self.assertIsNone(state.pending_verification)

    def test_contradiction_requires_fresh_support_without_deleting_history(self):
        state = EvidenceState()
        for compatible in (True, True, False, True):
            state = state.observe(compatible=compatible, informative=True, lateral_m=4)
        self.assertEqual((state.accepted, state.informative, state.contradictions), (4, 4, 1))
        self.assertEqual(state.supporting_informative, 1)
        self.assertFalse(state.ready_to_confirm)
        self.assertEqual(state.pending_verification, "confirmation_verification")

    def test_invalid_event_does_not_mutate_state(self):
        state = EvidenceState()
        for values in (
            {"compatible": 1, "informative": True, "lateral_m": 0},
            {"compatible": True, "informative": 1, "lateral_m": 0},
            {"compatible": True, "informative": True, "lateral_m": float("nan")},
        ):
            with self.assertRaises(ValueError):
                state.observe(**values)
        self.assertEqual(state, EvidenceState())

    def test_verification_can_select_negative_voi_without_faking_its_score(self):
        state = EvidenceState().observe(compatible=True, informative=True, lateral_m=4)
        choices = [proposal(4), proposal(0, informative=False), proposal(-4, travel=8)]
        selected = verification_action(state, choices)
        self.assertEqual(selected["lateral_m"], -4)
        self.assertEqual(selected["score"], -0.01)
        self.assertEqual(selected["verification_view_separation_m"], 8)
        self.assertFalse(selected["verification_same_view"])
        self.assertNotIn("selection_reason", choices[-1])

    def test_same_view_fallback_is_recorded_and_ambiguity_remains_unconfirmed(self):
        state = EvidenceState().observe(compatible=True, informative=True, lateral_m=4)
        selected = verification_action(state, [proposal(4)])
        self.assertTrue(selected["verification_same_view"])
        self.assertIsNone(verification_action(state, [proposal(0, informative=False)]))
        self.assertIsNone(verification_action(state, []))

    def test_contradiction_can_be_checked_without_candidate_separation(self):
        state = EvidenceState().observe(compatible=False, informative=False, lateral_m=0)
        selected = verification_action(state, [proposal(0, informative=False)])
        self.assertEqual(selected["selection_reason"], "contradiction_verification")
        self.assertIsNone(verification_action(EvidenceState(), [proposal(4)]))


if __name__ == "__main__":
    unittest.main()
