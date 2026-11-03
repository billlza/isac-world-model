import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_contrasts import contrasts


class TestPairedContrasts(unittest.TestCase):
    def test_hidden_width_can_help_one_architecture_and_hurt_another(self):
        cells = {
            key: {(s, e): level + (s + e) / 10 for s in (7, 11, 13) for e in (200, 201)}
            for key, level in [
                (("structured", 96), 4),
                (("structured", 192), 3),
                (("structured-attention", 96), 2),
                (("structured-attention", 192), 5),
            ]
        }
        out = contrasts(cells)["contrasts"]
        self.assertAlmostEqual(out["wide_minus_narrow_structured"]["mean"], -1)
        self.assertAlmostEqual(out["wide_minus_narrow_attention"]["mean"], 3)
        self.assertEqual(out["interaction_width_effect_attention_minus_structured"]["positive_pairs"], 6)
        del cells["structured", 96][7, 200]
        with self.assertRaisesRegex(ValueError, "mismatched"):
            contrasts(cells)
