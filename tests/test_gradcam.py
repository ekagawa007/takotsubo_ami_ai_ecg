"""Grad-CAM regressions with synthetic tensors and independent archived formulas.

No real ECGs, model downloads or trained checkpoints are needed.
"""
import ast
import copy
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from takotsubo_ami_ai_ecg import gradcam, inference
from takotsubo_ami_ai_ecg.settings import InferenceSettings


class TinyBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.layernorm_before = nn.Identity()
        self.scale = scale

    def forward(self, values):
        features = self.layernorm_before(values)
        return values + self.scale * features.square()


class TinyViT(nn.Module):
    """Small differentiable patch model exposing either supported HF layout."""
    def __init__(self, layout="legacy", blocks=4):
        super().__init__()
        self.vit = nn.Module()
        layers = nn.ModuleList([TinyBlock(0.2 + i * 0.1) for i in range(blocks)])
        if layout == "legacy":
            self.vit.encoder = nn.Module()
            self.vit.encoder.layer = layers
        else:
            self.vit.layers = layers
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.fail = False
        self.nonfinite = False

    def layers(self):
        return (self.vit.layers if hasattr(self.vit, "layers")
                else self.vit.encoder.layer)

    def forward(self, images, return_dict=True):
        patches = F.adaptive_avg_pool2d(images, (3, 3)).flatten(2).transpose(1, 2)
        # CLS contributes separately, so including its gradient in patch means
        # would change the map. This catches a common ViT CAM implementation bug.
        tokens = torch.cat((patches.mean(1, keepdim=True), patches), dim=1)
        for layer in self.layers():
            tokens = layer(tokens)
        if self.fail:
            raise RuntimeError("Synthetic forward failure")
        values = tokens[:, 1:]
        score0 = (values[..., 0] - 0.3 * values[..., 1]).mean(1)
        score1 = (values[..., 1] - 0.4 * values[..., 0]).mean(1)
        scores = torch.stack((score0, score1), dim=1) * self.scale
        scores = scores + tokens[:, 0, :2] * 5.0
        if self.nonfinite:
            scores = scores * float("nan")
        return SimpleNamespace(logits=scores)


class TinyConv(nn.Module):
    """Depthwise feature map whose class CAMs have known spatial patterns."""
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([nn.Module() for _ in range(4)])
        self.stages[3].blocks = nn.ModuleList([nn.Module() for _ in range(3)])
        self.stages[3].blocks[2].conv_dw = nn.Conv2d(3, 3, 1, groups=3, bias=False)
        with torch.no_grad():
            self.stages[3].blocks[2].conv_dw.weight.fill_(1.0)
        self.fail = False
        self.nonfinite = False

    def forward(self, images):
        features = self.stages[3].blocks[2].conv_dw(F.adaptive_avg_pool2d(images, (3, 3)))
        if self.fail:
            raise RuntimeError("Synthetic forward failure")
        scores = features.mean((2, 3))[:, :2]
        return scores * float("nan") if self.nonfinite else scores


def tensor_image(size=12):
    y, x = torch.meshgrid(torch.linspace(0.1, 1.0, size),
                          torch.linspace(0.1, 1.0, size), indexing="ij")
    return torch.stack((x, y, (x + y) / 2)).unsqueeze(0)


def archived_vit_formula():
    path = ROOT / "evaluation/original/ensemble/predict_single_image_ensemble.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names = {"_patch_start_index", "vit_gradcam_hidden"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != 2:
        raise AssertionError("Archived Grad-CAM reference functions changed.")
    scope = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["vit_gradcam_hidden"]


class GradCAMFormulaTests(unittest.TestCase):
    def test_conv_cam_matches_analytic_spatial_gradient_result(self):
        model, values = TinyConv(), tensor_image()
        result = gradcam.compute_gradcam(model, "convnextv2", values, 0)
        coarse = F.adaptive_avg_pool2d(values[:, :1], (3, 3)) / 9.0
        expected = F.interpolate(coarse, size=values.shape[2:], mode="bilinear", align_corners=False)
        expected = expected[0, 0].numpy()
        expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
        np.testing.assert_allclose(result["map"], expected, rtol=1e-6, atol=2e-7)
        self.assertEqual(result["layers"], ["stages.3.blocks.2.conv_dw"])
        self.assertFalse(result["map_is_constant"])
        self.assertEqual(result["map"].dtype, np.dtype("float32"))
        self.assertEqual(result["map"].shape, (12, 12))
        np.testing.assert_allclose(result["unaugmented_logits"], model(values).detach().numpy()[0])

    def test_class_switch_changes_map_in_expected_direction(self):
        model, values = TinyConv(), tensor_image()
        horizontal = gradcam.compute_gradcam(model, "convnextv2", values, 0)["map"]
        vertical = gradcam.compute_gradcam(model, "convnextv2", values, 1)["map"]
        self.assertGreater(horizontal[5, -1], horizontal[5, 0] + 0.9)
        self.assertGreater(vertical[-1, 5], vertical[0, 5] + 0.9)
        np.testing.assert_allclose(horizontal.T, vertical, atol=2e-7)

    def test_vit_matches_archived_three_layer_formula(self):
        model, values = TinyViT(), tensor_image()
        # Execute only the two archived mathematical functions, not its loader
        # or any legacy pickle/network/image IO.
        reference = archived_vit_formula()(copy.deepcopy(model), values.clone().requires_grad_(True),
                                           target_class=1, last_k_layers=3)
        reference = F.interpolate(torch.from_numpy(reference)[None, None],
                                  size=values.shape[2:], mode="bicubic", align_corners=False)[0, 0].numpy()
        reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-8)
        result = gradcam.compute_gradcam(model, "vit", values, 1)
        np.testing.assert_allclose(result["map"], reference, rtol=1e-6, atol=2e-7)
        self.assertEqual(result["layers"], [f"vit.encoder.layer.{i}.layernorm_before" for i in (1, 2, 3)])

    def test_hf5_and_legacy_layer_layouts_give_same_maps(self):
        values = tensor_image()
        legacy = gradcam.compute_gradcam(TinyViT("legacy"), "vit", values, 0)
        modern = gradcam.compute_gradcam(TinyViT("modern"), "vit", values, 0)
        np.testing.assert_array_equal(legacy["map"], modern["map"])
        self.assertEqual(modern["layers"], [f"vit.layers.{i}.layernorm_before" for i in (1, 2, 3)])

    def test_constant_map_is_finite_zero_and_explicitly_marked(self):
        result = gradcam.compute_gradcam(TinyConv(), "convnextv2", torch.ones(1, 3, 12, 12), 0)
        self.assertTrue(result["map_is_constant"])
        np.testing.assert_array_equal(result["map"], np.zeros((12, 12), dtype=np.float32))

    def test_frozen_weights_no_grad_and_inference_mode_are_supported(self):
        model, values = TinyViT("modern"), tensor_image()
        model.requires_grad_(False)
        expected = gradcam.compute_gradcam(model, "vit", values, 0)["map"]
        for context in (torch.no_grad, torch.inference_mode):
            with self.subTest(context=context.__name__), context():
                result = gradcam.compute_gradcam(model, "vit", values, 0)
                np.testing.assert_array_equal(result["map"], expected)
        self.assertFalse(values.requires_grad)
        self.assertIsNone(values.grad)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_model_training_flags_parameter_grads_and_hooks_are_preserved(self):
        model, values = TinyViT(), tensor_image()
        model.train()
        model.layers()[2].eval()
        model.scale.grad = torch.full_like(model.scale, 7.0)
        modes = [module.training for module in model.modules()]
        original = model.scale.detach().clone()
        hook = model.layers()[3].layernorm_before.register_forward_hook(lambda *args: None)
        hook_counts = [len(module._forward_hooks) for module in model.modules()]
        try:
            gradcam.compute_gradcam(model, "vit", values, 0)
            self.assertEqual([module.training for module in model.modules()], modes)
            self.assertEqual([len(module._forward_hooks) for module in model.modules()], hook_counts)
            torch.testing.assert_close(model.scale.grad, torch.tensor(7.0), rtol=0, atol=0)
            torch.testing.assert_close(model.scale.detach(), original, rtol=0, atol=0)
        finally:
            hook.remove()

    def test_forward_failure_removes_only_own_hooks_and_restores_modes(self):
        model = TinyConv()
        model.train()
        model.stages[3].eval()
        model.fail = True
        modes = [module.training for module in model.modules()]
        hook = model.stages[3].blocks[2].conv_dw.register_forward_hook(lambda *args: None)
        hook_counts = [len(module._forward_hooks) for module in model.modules()]
        try:
            with self.assertRaises(RuntimeError):
                gradcam.compute_gradcam(model, "convnextv2", tensor_image(), 0)
            self.assertEqual([module.training for module in model.modules()], modes)
            self.assertEqual([len(module._forward_hooks) for module in model.modules()], hook_counts)
        finally:
            hook.remove()

    def test_nonfinite_model_output_is_rejected_and_hooks_cleaned(self):
        for model, name in ((TinyConv(), "convnextv2"), (TinyViT(), "vit")):
            with self.subTest(model=name):
                model.nonfinite = True
                with self.assertRaises(ValueError):
                    gradcam.compute_gradcam(model, name, tensor_image(), 0)
                self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_invalid_class_dtype_shape_and_architecture_are_rejected(self):
        invalid = [(TinyConv(), "convnextv2", tensor_image(), -1),
                   (TinyConv(), "convnextv2", tensor_image(), 2),
                   (TinyConv(), "convnextv2", tensor_image(), True),
                   (TinyConv(), "convnextv2", tensor_image().double(), 0),
                   (TinyConv(), "convnextv2", tensor_image().repeat(2, 1, 1, 1), 0),
                   (TinyConv(), "mlpmixer", tensor_image(), 0),
                   (TinyViT(blocks=2), "vit", tensor_image(), 0),
                   (nn.Identity(), "convnextv2", tensor_image(), 0)]
        for model, name, values, target in invalid:
            with self.subTest(name=name, shape=tuple(values.shape), target=target), self.assertRaises(ValueError):
                gradcam.compute_gradcam(model, name, values, target)


class GradCAMPreprocessingTests(unittest.TestCase):
    def test_default_matches_validated_inference_transform(self):
        pixels = np.arange(23 * 19 * 3, dtype=np.uint16).reshape(23, 19, 3).astype(np.uint8)
        source = Image.fromarray(pixels)
        for name in ("vit", "convnextv2"):
            for policy in ("strict480", "resize"):
                with self.subTest(name=name, policy=policy):
                    expected = inference.build_transform(name, False, 1, policy)(source)
                    actual = gradcam.build_cam_transform(name, "inference", policy)(source)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_option_changes_only_vit_normalization(self):
        source = Image.new("RGB", (480, 480), "white")
        legacy = gradcam.build_cam_transform("vit", "legacy-vit", "strict480")(source)
        torch.testing.assert_close(legacy, torch.ones_like(legacy), rtol=0, atol=0)
        vit_default = gradcam.build_cam_transform("vit", "inference", "strict480")(source)
        self.assertFalse(torch.equal(vit_default, legacy))
        conv_default = gradcam.build_cam_transform("convnextv2", "inference", "strict480")(source)
        conv_legacy = gradcam.build_cam_transform("convnextv2", "legacy-vit", "strict480")(source)
        torch.testing.assert_close(conv_default, conv_legacy, rtol=0, atol=0)

    def test_ensemble_selects_only_two_evidenced_cam_models(self):
        self.assertEqual(tuple(gradcam.selected_cam_models("ensemble")), ("vit", "convnextv2"))
        for name in ("vit", "convnextv2"):
            self.assertEqual(tuple(gradcam.selected_cam_models(name)), (name,))
        with self.assertRaises(ValueError):
            gradcam.selected_cam_models("mlpmixer")

    def test_module_import_does_not_load_ml_libraries(self):
        code = ("import sys; sys.path.insert(0, " + repr(str(ROOT / "src")) + "); "
                "import takotsubo_ami_ai_ecg.gradcam; "
                "assert all(name not in sys.modules for name in ('torch','torchvision','timm','transformers'))")
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class GradCAMRunContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.image = self.directory / "synthetic.png"
        Image.new("RGB", (480, 480), (190, 210, 230)).save(self.image)
        self.weights = self.directory / "weights"
        identities = {}
        for name in ("vit", "convnextv2"):
            folder = self.weights / name
            folder.mkdir(parents=True)
            checkpoint = folder / "model.safetensors"
            checkpoint.write_bytes(b"synthetic test checkpoint " + name.encode())
            identities[name] = {"sha256": inference.sha256_file(checkpoint)}
            if name == "vit":
                config = folder / "config.json"
                config.write_text("{}", encoding="utf-8")
                identities[name]["config_sha256"] = inference.sha256_file(config)
        self.result = {
            "metadata": {"model": "ensemble", "number_of_images": 1,
                         "class_order": ["tako", "AMI"], "actual_device": "cpu",
                         "settings": InferenceSettings(device="cpu", num_workers=0).to_dict(),
                         "checkpoints": identities},
            "predictions": [{"sample_id": "sample_000001",
                             "input_sha256": inference.sha256_file(self.image),
                             "pred_id": 1, "pred_label": "AMI",
                             "models": {"vit": {"logits": [2.0, 2.0]},
                                        "convnextv2": {"logits": [-1.0, 3.0]},
                                        "mlpmixer": {"logits": [0.0, 2.0]}}}],
        }

    def fake_load(self, name, path, device):
        network = TinyViT() if name == "vit" else TinyConv()
        return network, copy.deepcopy(self.result["metadata"]["checkpoints"][name])

    @staticmethod
    def fake_compute(network, name, values, target):
        return {"map": np.zeros((480, 480), dtype=np.float32),
                "layers": ["synthetic_layer"], "unaugmented_logits": [1.0, 2.0],
                "map_is_constant": True}

    def test_target_selection_and_provenance_preserve_input_result_and_rng(self):
        original = copy.deepcopy(self.result)
        for option, expected in (("model-predicted", [0, 1]), ("ensemble-predicted", [1, 1]),
                                 ("tako", [0, 0]), ("AMI", [1, 1])):
            events = []
            rng_before = torch.random.get_rng_state().clone()
            with self.subTest(target=option), patch.object(gradcam, "_load_model", side_effect=self.fake_load) as loader, \
                    patch.object(gradcam, "compute_gradcam", side_effect=self.fake_compute) as calculate:
                output = gradcam.generate_gradcam([self.image], self.result, self.weights,
                                                  target=option, progress_callback=events.append)
            self.assertEqual([call.args[3] for call in calculate.call_args_list], expected)
            self.assertEqual([call.args[0] for call in loader.call_args_list], ["vit", "convnextv2"])
            self.assertEqual(output["metadata"]["actual_precision"], "float32")
            self.assertEqual(output["metadata"]["tta"], 1)
            self.assertEqual(output["images"][0]["sample_id"], "sample_000001")
            self.assertEqual(events[-1]["completed"], 2)
            self.assertEqual(events[-1]["total"], 2)
            self.assertEqual(self.result, original)
            torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
            self.assertNotIn(str(self.directory), repr(output["metadata"]))
            self.assertNotIn(str(self.directory), repr(events))

    def test_legacy_normalization_is_explicitly_recorded(self):
        with patch.object(gradcam, "_load_model", side_effect=self.fake_load), \
                patch.object(gradcam, "compute_gradcam", side_effect=self.fake_compute):
            output = gradcam.generate_gradcam([self.image], self.result, self.weights,
                                              normalization="legacy-vit")
        models = output["images"][0]["models"]
        self.assertFalse(models["vit"]["matches_prediction_normalization"])
        self.assertEqual(models["vit"]["normalization"], {"mean": [0.5] * 3, "std": [0.5] * 3})
        self.assertTrue(models["convnextv2"]["matches_prediction_normalization"])
        self.assertEqual(output["metadata"]["normalization"], "legacy-vit")

    def test_input_checkpoint_and_config_drift_stop_before_model_load(self):
        targets = [self.image, self.weights / "convnextv2/model.safetensors",
                   self.weights / "vit/config.json"]
        for path in targets:
            old = path.read_bytes()
            try:
                if path == self.image:
                    Image.new("RGB", (480, 480), "black").save(path)
                else:
                    path.write_bytes(old + b"modified")
                with self.subTest(kind=path.name), patch.object(gradcam, "_load_model") as loader, \
                        self.assertRaises(ValueError):
                    gradcam.generate_gradcam([self.image], self.result, self.weights)
                loader.assert_not_called()
            finally:
                path.write_bytes(old)

    def test_loaded_checkpoint_identity_is_rechecked(self):
        def changed_loader(name, path, device):
            network, identity = self.fake_load(name, path, device)
            identity["sha256"] = "0" * 64
            return network, identity
        with patch.object(gradcam, "_load_model", side_effect=changed_loader), \
                patch.object(gradcam, "compute_gradcam") as calculate, self.assertRaises(ValueError):
            gradcam.generate_gradcam([self.image], self.result, self.weights)
        calculate.assert_not_called()

    def test_invalid_record_and_options_fail_before_model_load(self):
        cases = []
        bad = copy.deepcopy(self.result)
        bad["metadata"]["class_order"] = ["AMI", "tako"]
        cases.append((bad, {}))
        bad = copy.deepcopy(self.result)
        bad["predictions"][0]["sample_id"] = "private_name"
        cases.append((bad, {}))
        bad = copy.deepcopy(self.result)
        bad["predictions"][0]["models"]["vit"]["logits"] = [float("nan"), 1.0]
        cases.append((bad, {}))
        bad = copy.deepcopy(self.result)
        bad["metadata"]["model"] = "mlpmixer"
        cases.append((bad, {}))
        bad = copy.deepcopy(self.result)
        bad["metadata"]["model"] = "vit"
        cases.append((bad, {"target": "ensemble-predicted"}))
        cases.extend([(self.result, {"normalization": "unsupported"}),
                      (self.result, {"target": "unsupported"})])
        for index, (bad, options) in enumerate(cases):
            with self.subTest(case=index), patch.object(gradcam, "_load_model") as loader, \
                    self.assertRaises(ValueError):
                gradcam.generate_gradcam([self.image], bad, self.weights, **options)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
