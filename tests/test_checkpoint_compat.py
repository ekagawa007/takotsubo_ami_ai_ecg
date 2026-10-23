"""Small acceptance/rejection checks for the ViT checkpoint compatibility adapter."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from takotsubo_ami_ai_ecg.checkpoint_compat import prepare_vit_state_dict


class CompatibilityChecks(unittest.TestCase):
    def setUp(self):
        self.tensor = SimpleNamespace(shape=(2, 3))
        self.bias = SimpleNamespace(shape=(2,))
        self.legacy = {
            "vit.encoder.layer.0.attention.attention.query.weight": self.tensor,
            "classifier.bias": self.bias,
        }
        self.expected = {
            "vit.layers.0.attention.q_proj.weight": SimpleNamespace(shape=(2, 3)),
            "classifier.bias": SimpleNamespace(shape=(2,)),
        }

    def test_existing_matching_layout_is_untouched(self):
        actual, name = prepare_vit_state_dict(self.legacy, self.legacy)
        self.assertIs(actual, self.legacy)
        self.assertIsNone(name)

    def test_known_rename_preserves_all_tensor_objects_and_input(self):
        original_keys = set(self.legacy)
        actual, name = prepare_vit_state_dict(self.legacy, self.expected)
        self.assertEqual(name, "hf_vit_legacy_to_modular_v1")
        self.assertIs(actual["vit.layers.0.attention.q_proj.weight"], self.tensor)
        self.assertIs(actual["classifier.bias"], self.bias)
        self.assertEqual(set(self.legacy), original_keys)
        self.assertEqual(len(actual), len(self.legacy))

    def test_shape_change_is_rejected(self):
        self.expected["vit.layers.0.attention.q_proj.weight"].shape = (3, 2)
        with self.assertRaises(ValueError):
            prepare_vit_state_dict(self.legacy, self.expected)

    def test_extra_tensor_is_not_dropped(self):
        self.legacy["unexpected.weight"] = self.tensor
        with self.assertRaises(ValueError):
            prepare_vit_state_dict(self.legacy, self.expected)

    def test_missing_tensor_is_not_initialized(self):
        del self.legacy["classifier.bias"]
        with self.assertRaises(ValueError):
            prepare_vit_state_dict(self.legacy, self.expected)

    def test_mixed_schema_is_rejected(self):
        self.legacy["vit.layers.0.attention.q_proj.weight"] = self.tensor
        with self.assertRaises(ValueError):
            prepare_vit_state_dict(self.legacy, self.expected)

    def test_unknown_legacy_field_is_rejected(self):
        self.legacy["vit.encoder.layer.0.custom_loader.weight"] = self.tensor
        with self.assertRaises(ValueError):
            prepare_vit_state_dict(self.legacy, self.expected)

    def test_backward_or_unknown_conversion_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_vit_state_dict(self.expected, self.legacy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
