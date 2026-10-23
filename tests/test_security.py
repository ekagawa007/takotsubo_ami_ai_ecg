"""Regression checks for dependency gates and untrusted configuration fields.

The malformed-input tests intercept decoding, checkpoint loading and external
kernel dispatch. They do not download code or open an exploit image/checkpoint.
"""

import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import PIL
from PIL import Image
import torch

from takotsubo_ami_ai_ecg import inference, security

EXPORT_SPEC = importlib.util.spec_from_file_location(
    "ecg_export_security_test", ROOT / "tools" / "export_inference_weights.py"
)
exporter = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(exporter)

VALID_CONFIG = {
    "architectures": ["ViTForImageClassification"],
    "attention_probs_dropout_prob": 0.2, "encoder_stride": 16,
    "hidden_act": "gelu", "hidden_dropout_prob": 0.2, "hidden_size": 768,
    "id2label": {"0": "tako", "1": "AMI"}, "image_size": 480,
    "initializer_range": 0.02, "intermediate_size": 3072,
    "label2id": {"AMI": 1, "tako": 0}, "layer_norm_eps": 1e-12,
    "model_type": "vit", "num_attention_heads": 12, "num_channels": 3,
    "num_hidden_layers": 12, "patch_size": 16, "pooler_act": "tanh",
    "pooler_output_size": 768, "problem_type": "single_label_classification",
    "qkv_bias": True, "torch_dtype": "float32", "transformers_version": "4.55.1",
}


class PillowGateTests(unittest.TestCase):
    def test_old_pillow_cannot_reach_image_decoding(self):
        dataset = inference.ImageDataset([Path("example.png")], lambda image: image)
        for operation in (
            lambda: inference._validate_images([Path("example.png")], "strict480"),
            lambda: dataset[0],
        ):
            with self.subTest(operation=operation), patch.object(PIL, "__version__", "12.0.0"), \
                    patch.object(security.importlib.metadata, "version", return_value="12.0.0"), \
                    patch.object(Image, "open") as open_image, \
                    self.assertRaisesRegex(ValueError, "Pillow >= 13.0.0"):
                operation()
            open_image.assert_not_called()

    def test_installed_update_with_stale_loaded_module_requires_restart(self):
        with patch.object(PIL, "__version__", "12.0.0"), \
                patch.object(security.importlib.metadata, "version", return_value="13.0.0"), \
                self.assertRaisesRegex(ValueError, "restart"):
            security.require_patched_pillow()

    def test_distinct_patched_versions_also_require_restart(self):
        with patch.object(PIL, "__version__", "13.0.0"), \
                patch.object(security.importlib.metadata, "version", return_value="13.0.1"), \
                self.assertRaisesRegex(ValueError, "versions differ"):
            security.require_patched_pillow()

    def test_stable_patched_release_is_accepted(self):
        with patch.object(PIL, "__version__", "13.0.0"), \
                patch.object(Image, "__version__", "13.0.0"), \
                patch.object(security.importlib.metadata, "version", return_value="13.0.0"):
            security.require_patched_pillow()

    def test_cached_image_module_is_not_accepted_after_partial_reload(self):
        with patch.object(PIL, "__version__", "13.0.0"), \
                patch.object(Image, "__version__", "12.0.0"), \
                patch.object(security.importlib.metadata, "version", return_value="13.0.0"), \
                self.assertRaisesRegex(ValueError, "cached Pillow image module"):
            security.require_patched_pillow()

    def test_prerelease_or_unrecognized_version_is_not_assumed_patched(self):
        for version in ("13.0.0rc1", "13.0.0.dev1", "unknown"):
            with self.subTest(version=version), patch.object(PIL, "__version__", version), \
                    patch.object(security.importlib.metadata, "version", return_value=version), \
                    self.assertRaises(ValueError):
                security.require_patched_pillow()


class VitConfigurationTests(unittest.TestCase):
    def test_selected_configuration_survives_without_backend_override(self):
        original = copy.deepcopy(VALID_CONFIG)
        cleaned = security.validate_vit_config(original)
        self.assertEqual(cleaned, VALID_CONFIG)
        self.assertEqual(original, VALID_CONFIG)
        self.assertNotIn("attn_implementation", cleaned)
        self.assertNotIn("_attn_implementation_internal", cleaned)

    def test_execution_controls_fail_before_transformers_or_tensor_loading(self):
        controls = {
            "_attn_implementation_internal": "example/blocked-kernel",
            "attn_implementation": "example/blocked-kernel",
            "_attn_implementation": "example/blocked-kernel",
            "auto_map": {"AutoModel": "example/blocked-model"},
            "_name_or_path": "example/blocked-model",
            "quantization_config": {"quant_method": "example"},
        }
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.ViTConfig = types.SimpleNamespace(from_dict=Mock())
        fake_transformers.ViTForImageClassification = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tensor_file = folder / "model.safetensors"
            tensor_file.write_bytes(b"not a tensor file; decoding must not be reached")
            for key, value in controls.items():
                config = dict(VALID_CONFIG, **{key: value})
                (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(field=key), \
                        patch.dict(sys.modules, {"transformers": fake_transformers}), \
                        patch("safetensors.torch.load_file") as load_tensors, \
                        self.assertRaisesRegex(ValueError, "unsupported configuration fields"):
                    inference._load_model("vit", tensor_file, "cpu")
                load_tensors.assert_not_called()
        fake_transformers.ViTConfig.from_dict.assert_not_called()
        fake_transformers.ViTForImageClassification.assert_not_called()

    def test_types_and_dimensions_cannot_smuggle_nested_execution_controls(self):
        cases = [
            ("hidden_size", {"_attn_implementation_internal": "example/blocked-kernel"}),
            ("hidden_size", 10**100), ("num_hidden_layers", 10**100),
            ("patch_size", 1), ("num_attention_heads", 0),
            ("hidden_dropout_prob", float("nan")),
            ("initializer_range", float("inf")),
            ("initializer_range", 10**1000),
            ("qkv_bias", "true"), ("hidden_act", "example/blocked-kernel"),
        ]
        for key, value in cases:
            with self.subTest(field=key), self.assertRaises(ValueError):
                security.validate_vit_config(dict(VALID_CONFIG, **{key: value}))

    def test_oversized_configuration_is_rejected_before_json_parse(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tensor_file = folder / "model.safetensors"
            tensor_file.write_bytes(b"not used")
            (folder / "config.json").write_bytes(b" " * 65537)
            with patch.object(inference.json, "loads") as parse_json, \
                    self.assertRaisesRegex(ValueError, "file size"):
                inference._load_model("vit", tensor_file, "cpu")
            parse_json.assert_not_called()

    def test_valid_small_vit_initializes_without_external_kernel_dispatch(self):
        from transformers import ViTConfig, ViTForImageClassification

        small = dict(VALID_CONFIG, hidden_size=32, intermediate_size=64,
                     num_attention_heads=4, num_hidden_layers=1, pooler_output_size=32)
        from transformers import modeling_utils

        # Intercept the actual dispatch boundary for the installed library.
        # Do not create missing attributes: an unrecognized API should fail.
        if hasattr(modeling_utils, "load_and_register_kernel"):
            targets = ["transformers.modeling_utils.load_and_register_kernel"]
        else:
            targets = [
                "transformers.modeling_utils.lazy_import_flash_attention",
                "transformers.modeling_utils.lazy_import_paged_flash_attention",
                "transformers.integrations.hub_kernels.get_kernel",
            ]
        with ExitStack() as stack:
            dispatchers = [
                stack.enter_context(patch(
                    target, side_effect=AssertionError("External kernel dispatch is forbidden")
                ))
                for target in targets
            ]
            direct = ViTForImageClassification(ViTConfig.from_dict(copy.deepcopy(small)))
            guarded = ViTForImageClassification(ViTConfig.from_dict(security.validate_vit_config(small)))
        for dispatch in dispatchers:
            dispatch.assert_not_called()
        self.assertEqual(direct.config._attn_implementation, guarded.config._attn_implementation)


class ExportDependencyTests(unittest.TestCase):
    def test_old_torch_fails_before_checkpoint_read_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-bundle"
            with patch.object(torch, "__version__", "2.9.0+cu130"), \
                    patch.object(torch, "load") as load_checkpoint, \
                    patch.object(exporter, "load_file") as load_safetensors, \
                    self.assertRaisesRegex(ValueError, "PyTorch release >= 2.10.0"):
                exporter.export_bundle(Path("vit"), Path("processor.json"),
                                       Path("convnextv2.pth"), Path("mlpmixer.pth"), output)
            load_checkpoint.assert_not_called()
            load_safetensors.assert_not_called()
            self.assertFalse(output.exists())

    def test_unpatched_and_prerelease_versions_are_rejected(self):
        for version in ("2.9.1", "2.9.2", "2.10.0rc1", "2.10.0a0+git123", "unknown"):
            with self.subTest(version=version), patch.object(torch, "__version__", version), \
                    self.assertRaises(ValueError):
                exporter.require_patched_torch()

    def test_patched_stable_cpu_and_cuda_versions_are_accepted(self):
        for version in ("2.10.0", "2.10.0+cpu", "2.11.0+cu130", "2.10.0.post1"):
            with self.subTest(version=version), patch.object(torch, "__version__", version):
                exporter.require_patched_torch()


if __name__ == "__main__":
    unittest.main()
