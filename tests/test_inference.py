"""Scientific regression checks using synthetic images, without trained weights.

Run from the repository root:
    python -m unittest discover -s tests -p test_inference.py
"""

import ast
from collections import OrderedDict
from dataclasses import replace
import io
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import PIL
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandAugment

from takotsubo_ami_ai_ecg import cli, inference
from takotsubo_ami_ai_ecg.settings import InferenceSettings, ModelSettings


class LogitAggregationTests(unittest.TestCase):
    def test_mean_logits_is_not_mean_probabilities(self):
        logits = [np.array([[0.0, 12.0]]), np.array([[0.0, -3.0]]), np.array([[0.0, -3.0]])]
        averaged, probabilities = inference.combine_logits(logits, [1, 1, 1])
        np.testing.assert_array_equal(averaged, [[0.0, 2.0]])
        self.assertAlmostEqual(probabilities[0, 1], 1.0 / (1.0 + math.exp(-2.0)), places=14)
        probability_average = np.mean([inference.softmax(values) for values in logits], axis=0)
        self.assertGreater(abs(probabilities[0, 1] - probability_average[0, 1]), 0.4)

    def test_weights_are_normalized_and_zero_excludes_a_model(self):
        logits = [np.array([[2.0, -2.0]]), np.array([[100.0, -100.0]]), np.array([[0.0, 4.0]])]
        averaged, probabilities = inference.combine_logits(logits, [2, 0, 1])
        np.testing.assert_allclose(averaged, [[4.0 / 3.0, 0.0]], rtol=0, atol=1e-14)
        scaled, scaled_probabilities = inference.combine_logits(logits, [20, 0, 10])
        np.testing.assert_array_equal(averaged, scaled)
        np.testing.assert_array_equal(probabilities, scaled_probabilities)

    def test_softmax_is_stable_with_large_common_offset(self):
        expected = inference.softmax([[0.0, 2.0]])
        actual = inference.softmax([[10000.0, 10002.0]])
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=15)

    def test_rejects_invalid_logits_and_weights(self):
        good = np.array([[0.0, 1.0]])
        for weights in ([0, 0], [1, -1], [1, float("nan")], [1, float("inf")]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                inference.combine_logits([good, good], weights)
        for bad in (np.array([[0.0, float("nan")]]), np.zeros((2, 2)), np.zeros((1, 3))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                inference.combine_logits([good, bad], [1, 1])

    def test_historical_decision_rules_preserve_exact_ties(self):
        probabilities = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
        np.testing.assert_array_equal(inference.classify_probabilities(probabilities), [1, 1, 0])
        for name in ("vit", "convnextv2", "mlpmixer"):
            np.testing.assert_array_equal(inference.classify_probabilities(probabilities, model=name), [0, 1, 0])
            np.testing.assert_array_equal(inference.classify_probabilities(probabilities, model=name, threshold=0.5), [1, 1, 0])

    def test_tta_count_includes_identity_and_uses_float64_accumulation(self):
        class ControlledModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, images):
                first_logit = (100000000.0, 1.0, -100000000.0)[self.calls]
                self.calls += 1
                return torch.tensor([[first_logit, 0.0]], dtype=torch.float32).repeat(len(images), 1)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.png"
            Image.new("RGB", (480, 480), "white").save(path)
            settings = replace(InferenceSettings(), convnextv2=ModelSettings(3, 1), num_workers=0, device="cpu")
            model = ControlledModel()
            result = inference._predict_one_model(model, "convnextv2", [path], settings, "cpu")
        self.assertEqual(model.calls, 3)
        self.assertEqual(result.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(result, np.array([[1.0 / 3.0, 0.0]], dtype=np.float32))


class CheckpointLoadingTests(unittest.TestCase):
    def test_incomplete_state_cannot_silently_leave_a_random_head(self):
        from safetensors.torch import save_file

        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.safetensors"
            save_file({"weight": torch.ones((2, 2))}, str(checkpoint))
            with patch.dict(sys.modules, {"timm": fake_timm}), self.assertRaisesRegex(ValueError, "keys or shapes"):
                inference._load_model("convnextv2", checkpoint, "cpu")

    def test_pickle_checkpoints_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "pickle checkpoints are not accepted"):
            inference._load_model("convnextv2", Path("example.pth"), "cpu")


def original_transform_factory(model_name):
    """Compile only the archived transform definitions, not evaluator imports/main."""
    filenames = {
        "vit": "vit/evaluate_dataset_dir_ViT.py",
        "convnextv2": "convnextv2/evaluate_dataset_dir_convnextv2.py",
        "mlpmixer": "mlpmixer/evaluate_dataset_dir_mlpmixer.py",
    }
    path = ROOT / "evaluation" / "original" / filenames[model_name]
    parsed = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names = {"RandAugmentNoFlip", "build_identity_transform", "build_tta_transform"}
    selected = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(selected) != 3:
        raise AssertionError("Archived evaluator no longer exposes the expected transform definitions.")
    namespace = {
        "RandAugment": RandAugment, "OrderedDict": OrderedDict,
        "transforms": transforms, "InterpolationMode": InterpolationMode,
        "PIL": PIL, "IMAGE_SIZE": 480,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class OriginalPreprocessingParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y = np.meshgrid(np.arange(480, dtype=np.uint16), np.arange(480, dtype=np.uint16))
        pixels = np.stack([(3 * x + y) % 256, (x + 5 * y) % 256, (7 * x + 2 * y) % 256], axis=-1)
        cls.image = Image.fromarray(pixels.astype(np.uint8))

    def test_identity_and_tta_match_archived_evaluators(self):
        for model_name, magnitude in (("vit", 1), ("convnextv2", 5), ("mlpmixer", 3)):
            original = original_transform_factory(model_name)
            with self.subTest(model=model_name, augmentation="identity"):
                expected = original["build_identity_transform"]()(self.image)
                actual = inference.build_transform(model_name, False, magnitude)(self.image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for seed in (1234, 1241):
                with self.subTest(model=model_name, augmentation="randaugment", seed=seed):
                    reference = (original["build_tta_transform"]("randaug", magnitude)
                                 if model_name == "convnextv2" else original["build_tta_transform"](magnitude))
                    torch.manual_seed(seed)
                    expected = reference(self.image)
                    torch.manual_seed(seed)
                    actual = inference.build_transform(model_name, True, magnitude)(self.image)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_white_image_uses_imagenet_normalization_for_every_model(self):
        white = Image.new("RGB", (480, 480), "white")
        expected = torch.tensor([(1.0 - mean) / std for mean, std in
                                 zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        for name in ("vit", "convnextv2", "mlpmixer"):
            with self.subTest(model=name):
                transformed = inference.build_transform(name, False, 1)(white)
                torch.testing.assert_close(transformed[:, 0, 0], expected)

    def test_strict_input_size_fails_and_resize_requires_explicit_choice(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.png"
            Image.new("RGB", (320, 240), "white").save(path)
            with self.assertRaisesRegex(ValueError, "480x480"):
                inference._validate_images([path], "strict480")
            self.assertEqual(len(inference._validate_images([path], "resize")), 1)
            with Image.open(path) as image:
                tensor = inference.build_transform("vit", False, 1, "resize")(image)
            self.assertEqual(tuple(tensor.shape), (3, 480, 480))


class SettingsAndCliTests(unittest.TestCase):
    def test_author_confirmed_defaults(self):
        args = cli.build_parser().parse_args(["--image", "example.png", "--output", "prediction.json"])
        settings = cli.settings_from_args(args)
        self.assertEqual((settings.vit.tta, settings.vit.magnitude), (24, 1))
        self.assertEqual(settings.convnextv2.tta, 1)
        self.assertEqual((settings.mlpmixer.tta, settings.mlpmixer.magnitude), (32, 3))
        self.assertEqual(settings.ensemble_weights, (1, 1, 1))
        self.assertEqual(settings.seed, 1234)
        self.assertEqual(settings.precision, "float32")
        self.assertEqual(settings.input_policy, "strict480")
        self.assertEqual(settings.num_workers, 4)

    def test_cli_can_override_each_model_and_shared_options(self):
        args = cli.build_parser().parse_args([
            "--image", "example.png", "--output", "prediction.json",
            "--vit-tta", "2", "--vit-magnitude", "4",
            "--convnextv2-tta", "3", "--convnextv2-magnitude", "2",
            "--mlpmixer-tta", "5", "--mlpmixer-magnitude", "6",
            "--ensemble-weights", "2", "0", "1", "--threshold", "0.6",
            "--seed", "42", "--num-workers", "0", "--batch-size", "2",
            "--input-policy", "resize", "--device", "cpu",
        ])
        settings = cli.settings_from_args(args)
        self.assertEqual(settings.vit, ModelSettings(2, 4))
        self.assertEqual(settings.convnextv2, ModelSettings(3, 2))
        self.assertEqual(settings.mlpmixer, ModelSettings(5, 6))
        self.assertEqual(settings.ensemble_weights, (2, 0, 1))
        self.assertEqual((settings.threshold, settings.seed, settings.num_workers, settings.batch_size), (0.6, 42, 0, 2))

    def test_single_model_aliases_override_its_settings(self):
        args = cli.build_parser("vit").parse_args([
            "--image", "example.png", "--output", "prediction.json", "--tta", "7", "--tta-magnitude", "2",
        ])
        self.assertEqual(cli.settings_from_args(args).vit, ModelSettings(7, 2))

    def test_rejects_undefined_magnitude_bin_and_zero_tta(self):
        for specification in (ModelSettings(0, 1), ModelSettings(2, 10), ModelSettings(2, -1)):
            with self.subTest(specification=specification), self.assertRaises(ValueError):
                replace(InferenceSettings(), vit=specification).validate()

    def test_csv_order_is_preserved_and_paths_are_relative_to_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = directory / "inputs.csv"
            manifest.write_text("path\nimages/c.png\nimages/a.png\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["--input-csv", str(manifest), "--output", "prediction.json"])
            actual = cli.collect_images(args)
            self.assertEqual(actual, [(directory / "images/c.png").resolve(), (directory / "images/a.png").resolve()])

    def test_override_vit_config_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model_directory = directory / "custom-vit"
            model_directory.mkdir()
            config = model_directory / "config.json"
            config.write_text('{"preserve": true}', encoding="utf-8")
            image = directory / "example.png"
            Image.new("RGB", (480, 480), "white").save(image)
            with patch.object(inference, "predict") as predict_mock, patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                cli.main(["--image", str(image), "--vit-weights", str(model_directory),
                          "--output", str(config), "--overwrite"], fixed_model="vit")
            self.assertEqual(error.exception.code, 2)
            predict_mock.assert_not_called()
            self.assertEqual(config.read_text(encoding="utf-8"), '{"preserve": true}')


if __name__ == "__main__":
    unittest.main()
