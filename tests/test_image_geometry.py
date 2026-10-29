"""Geometry-policy checks on generated pixels, independent of model weights."""

import io
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from takotsubo_ami_ai_ecg import cli, gradcam, gradcam_output, inference
from takotsubo_ami_ai_ecg.image_input import (
    ImageGeometry, ImageInputError, describe_image_geometry,
    prepare_image_geometry,
)
from takotsubo_ami_ai_ecg.settings import InferenceSettings, MEAN, STD


def pixels(size):
    width, height = size
    x, y = np.meshgrid(np.arange(width, dtype=np.uint32), np.arange(height, dtype=np.uint32))
    values = np.stack([(3 * x + y) % 256, (x + 5 * y) % 256, (7 * x + 2 * y) % 256], axis=-1)
    return Image.fromarray(values.astype(np.uint8))


class ImageGeometryTests(unittest.TestCase):
    def test_strict_default_rejects_non_480_before_preparation(self):
        for size in [(479, 480), (960, 960), (960, 480)]:
            with self.subTest(size=size), self.assertRaisesRegex(ImageInputError, "480x480"):
                describe_image_geometry(size, "strict480")
        self.assertEqual(InferenceSettings().input_policy, "strict480")

    def test_square_resize_refuses_rectangles(self):
        with self.assertRaisesRegex(ImageInputError, "square image"):
            prepare_image_geometry(pixels((19, 13)), "square-resize")
        image = pixels((13, 13))
        expected = image.resize((480, 480), Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(prepare_image_geometry(image, "square-resize"), expected)
        self.assertEqual(image.size, (13, 13))

    def test_center_crop_keeps_short_side_and_odd_extra_right_or_bottom(self):
        cases = [((9, 5), (2, 0, 7, 5)), ((5, 10), (0, 2, 5, 7))]
        for size, box in cases:
            with self.subTest(size=size):
                image = pixels(size)
                expected = image.crop(box).resize((480, 480), Image.Resampling.BILINEAR)
                np.testing.assert_array_equal(prepare_image_geometry(image, "center-crop"), expected)
                description = describe_image_geometry(size, "center-crop")
                self.assertEqual(description["crop_box"], list(box))
                self.assertEqual(description["original_size"], list(size))
                self.assertEqual(description["prepared_size"], [480, 480])

    def test_padding_uses_white_long_side_square_then_resizes(self):
        for size, offset in [((4, 7), (1, 0)), ((8, 3), (0, 2))]:
            with self.subTest(size=size):
                image = pixels(size)
                side = max(size)
                expected = Image.new("RGB", (side, side), (255, 255, 255))
                expected.paste(image, offset)
                expected = expected.resize((480, 480), Image.Resampling.BILINEAR)
                actual = prepare_image_geometry(image, "pad")
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(np.asarray(actual)[0, 0], [255, 255, 255])

    def test_padding_odd_pixel_placement_without_resizing(self):
        image = Image.new("RGB", (301, 480), (0, 0, 0))
        actual = np.asarray(prepare_image_geometry(image, "pad"))
        self.assertTrue((actual[:, :89] == 255).all())
        self.assertTrue((actual[:, 89:390] == 0).all())
        self.assertTrue((actual[:, 390:] == 255).all())
        description = describe_image_geometry(image.size, "pad")
        self.assertEqual(description["padding"], [89, 0, 90, 0])
        self.assertFalse(description["resize_applied"])

    def test_expanded_canvas_bound_checked_without_allocation(self):
        with self.assertRaisesRegex(ImageInputError, "40,000,000"):
            describe_image_geometry((1, 100000), "pad")
        # The original image can still be explicitly cropped or stretched.
        self.assertEqual(describe_image_geometry((1, 100000), "center-crop")["size_before_resize"], [1, 1])
        self.assertEqual(describe_image_geometry((1, 100000), "resize")["prepared_size"], [480, 480])

    def test_invalid_dimensions_policy_and_mode_rejected(self):
        for size in [(0, 4), (4, -1), (True, 4), (4.0, 4), (1,), "480"]:
            with self.subTest(size=size), self.assertRaises(ImageInputError):
                describe_image_geometry(size, "pad")
        with self.assertRaises(ImageInputError):
            describe_image_geometry((480, 480), "unknown")
        with self.assertRaisesRegex(ImageInputError, "RGB"):
            prepare_image_geometry(Image.new("L", (480, 480)), "strict480")

    def test_geometry_transform_is_picklable(self):
        transform = pickle.loads(pickle.dumps(ImageGeometry("center-crop")))
        self.assertEqual(transform(pixels((6, 4))).size, (480, 480))

    def test_existing_480_pixels_and_tta_rng_are_unchanged(self):
        image = pixels((480, 480))
        for policy in ["square-resize", "center-crop", "pad"]:
            self.assertIs(prepare_image_geometry(image, policy), image)
            for model in ["vit", "convnextv2", "mlpmixer"]:
                for augmented in [False, True]:
                    with self.subTest(policy=policy, model=model, augmented=augmented):
                        torch.manual_seed(1277)
                        expected = inference.build_transform(model, augmented, 3, "strict480")(image)
                        state = torch.random.get_rng_state()
                        torch.manual_seed(1277)
                        actual = inference.build_transform(model, augmented, 3, policy)(image)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_legacy_resize_tensor_is_unchanged(self):
        image = pixels((611, 391))
        expected = transforms.Compose([
            transforms.Resize((480, 480), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD),
        ])(image)
        for model in ["vit", "convnextv2", "mlpmixer"]:
            torch.testing.assert_close(inference.build_transform(model, False, 0, "resize")(image), expected, rtol=0, atol=0)

    def test_all_prediction_and_cam_models_use_same_prepared_pixels(self):
        for policy, size in [("square-resize", (51, 51)), ("center-crop", (71, 51)), ("pad", (51, 71))]:
            image = pixels(size)
            prepared = prepare_image_geometry(image, policy)
            expected = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])(prepared)
            for model in ["vit", "convnextv2", "mlpmixer"]:
                with self.subTest(policy=policy, model=model):
                    torch.testing.assert_close(inference.build_transform(model, False, 0, policy)(image), expected, rtol=0, atol=0)
            for model in ["vit", "convnextv2"]:
                torch.testing.assert_close(gradcam.build_cam_transform(model, "inference", policy)(image), expected, rtol=0, atol=0)
            legacy = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(.5,) * 3, std=(.5,) * 3)])(prepared)
            torch.testing.assert_close(gradcam.build_cam_transform("vit", "legacy-vit", policy)(image), legacy, rtol=0, atol=0)

    def test_validation_records_geometry_beside_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private_original.png"
            pixels((21, 13)).save(path)
            records = []
            hashes = inference._validate_images([path], "center-crop", records)
            self.assertEqual(hashes, [inference.sha256_file(path)])
            self.assertEqual(records[0]["sample_id"], "sample_000001")
            self.assertEqual(records[0]["crop_box"], [4, 0, 17, 13])
            self.assertNotIn("private", str(records))
            with self.assertRaises(ImageInputError):
                inference._validate_images([path], "square-resize")

    def test_cli_exposes_new_policies_without_changing_defaults(self):
        for policy in ["square-resize", "center-crop", "pad"]:
            args = cli.build_parser().parse_args(["--image", "image.png", "--output", "prediction.json", "--input-policy", policy])
            self.assertEqual(cli.settings_from_args(args).input_policy, policy)
        self.assertEqual(cli.settings_from_args(cli.build_parser().parse_args(["--image", "image.png", "--output", "prediction.json"])).input_policy, "strict480")

    def test_gradcam_export_uses_the_same_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            for policy, size in [("square-resize", (17, 17)), ("center-crop", (27, 17)), ("pad", (17, 27))]:
                with self.subTest(policy=policy):
                    path = Path(temp) / "private_source.png"
                    image = pixels(size)
                    image.save(path)
                    digest = inference.sha256_file(path)
                    row = {"sample_id": "sample_000001", "input_sha256": digest}
                    result = {"metadata": {"model": "vit", "settings": {"input_policy": policy}}, "predictions": [row]}
                    values = np.linspace(0, 1, 480 * 480, dtype=np.float32).reshape(480, 480)
                    cam = {"metadata": {}, "images": [{"sample_id": row["sample_id"], "input_sha256": digest, "models": {"vit": {
                        "map": values, "map_is_constant": False, "target_id": 0, "target_label": "tako",
                        "target_source": "model-predicted", "layers": ["fixture"],
                        "unaugmented_logits": [.2, -.2], "normalization": {"mean": MEAN, "std": STD},
                    }}}]}
                    contents = gradcam_output.serialize_gradcam([path], result, cam, Path(temp) / "result.json")
                    with Image.open(io.BytesIO(contents[0][1])) as exported:
                        np.testing.assert_array_equal(exported, prepare_image_geometry(image, policy))


if __name__ == "__main__":
    unittest.main()
