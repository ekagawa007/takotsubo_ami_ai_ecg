"""Colab input and device helpers using local synthetic fixtures only.

No browser, Drive connection, model weights, GPU or network is required.
"""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from takotsubo_ami_ai_ecg.colab_support import (
    choose_device,
    generate_synthetic,
    inspect_input,
    resolve_tta,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ColabDeviceTests(unittest.TestCase):
    def test_auto_uses_available_cuda_without_a_fallback_warning(self):
        device, warning = choose_device("auto", True)
        self.assertEqual(device, "cuda")
        self.assertFalse(warning)

    def test_missing_cuda_warns_and_falls_back_for_auto_and_cuda_requests(self):
        for request in ("auto", "cuda"):
            with self.subTest(request=request):
                device, warning = choose_device(request, False)
                self.assertEqual(device, "cpu")
                self.assertTrue(warning)
                self.assertIn("CPU", warning)

    def test_explicit_cpu_stays_cpu_even_when_cuda_exists(self):
        for available in (False, True):
            with self.subTest(cuda_available=available):
                device, warning = choose_device("cpu", available)
                self.assertEqual(device, "cpu")
                self.assertTrue(warning)

    def test_unknown_device_does_not_silently_choose_a_runtime(self):
        for request in ("gpu", "tpu", "", None):
            with self.subTest(request=request), self.assertRaises(ValueError):
                choose_device(request, True)


class ColabTtaTests(unittest.TestCase):
    def test_paper_settings_preserve_all_author_selected_defaults(self):
        self.assertEqual(resolve_tta("Paper defaults"), {
            "vit": {"tta": 24, "magnitude": 1},
            "convnextv2": {"tta": 1, "magnitude": 5},
            "mlpmixer": {"tta": 32, "magnitude": 3},
        })

    def test_no_tta_uses_one_identity_view_and_retains_unused_magnitudes(self):
        self.assertEqual(resolve_tta("No TTA"), {
            "vit": {"tta": 1, "magnitude": 1},
            "convnextv2": {"tta": 1, "magnitude": 5},
            "mlpmixer": {"tta": 1, "magnitude": 3},
        })

    def test_custom_controls_counts_and_magnitudes_independently(self):
        result = resolve_tta("Custom", vit_views=2, vit_magnitude=0,
                             convnextv2_views=7, convnextv2_magnitude=9,
                             mlpmixer_views=3, mlpmixer_magnitude=2)
        self.assertEqual(result, {
            "vit": {"tta": 2, "magnitude": 0},
            "convnextv2": {"tta": 7, "magnitude": 9},
            "mlpmixer": {"tta": 3, "magnitude": 2},
        })

    def test_invalid_custom_values_are_rejected_before_inference(self):
        for argument in ("vit_views", "convnextv2_views", "mlpmixer_views"):
            for value in (0, -1, 1.5, True, "2"):
                with self.subTest(argument=argument, value=value), self.assertRaises(ValueError):
                    resolve_tta("Custom", **{argument: value})
        for argument in ("vit_magnitude", "convnextv2_magnitude", "mlpmixer_magnitude"):
            for value in (-1, 10, 1.5, True, "3"):
                with self.subTest(argument=argument, value=value), self.assertRaises(ValueError):
                    resolve_tta("Custom", **{argument: value})

    def test_unknown_preset_is_not_treated_as_paper_defaults(self):
        with self.assertRaises(ValueError):
            resolve_tta("fast")

    def test_cpu_selection_does_not_change_paper_tta(self):
        before = resolve_tta("Paper defaults")
        choose_device("auto", False)
        self.assertEqual(resolve_tta("Paper defaults"), before)


class ColabInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ecg_colab_", ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "private_subject_001.png"
        self.preview = self.root / "prepared.png"

    def image(self, size=(480, 480), color=(30, 80, 140), mode="RGB"):
        image = Image.new(mode, size, color)
        image.save(self.source, format="PNG")
        return image

    def test_record_and_preview_identify_the_same_unmodified_input(self):
        original_image = self.image()
        original_bytes = self.source.read_bytes()
        record = inspect_input(self.source, "user_upload", "strict480", self.preview)
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(record["size"], [480, 480])
        self.assertEqual(record["mode"], "RGB")
        self.assertEqual(record["format"], "PNG")
        self.assertEqual(record["kind"], "user_upload")
        self.assertEqual(record["input_sha256"], digest(original_bytes))
        self.assertEqual(record["rgb_pixel_sha256"], digest(original_image.tobytes()))
        self.assertEqual(record["prepared_rgb_pixel_sha256"], digest(original_image.tobytes()))
        self.assertEqual(record["preview_sha256"], digest(self.preview.read_bytes()))
        self.assertEqual(record["preview_size_bytes"], self.preview.stat().st_size)
        with Image.open(self.preview) as displayed:
            self.assertEqual(displayed.size, (480, 480))
            self.assertEqual(displayed.mode, "RGB")
            self.assertEqual(displayed.tobytes(), original_image.tobytes())
        serialized = json.dumps(record)
        self.assertNotIn(self.source.name, serialized)
        self.assertNotIn(str(self.root), serialized)

    def test_all_input_sources_use_identical_decoded_pixels(self):
        image = self.image()
        original = self.source.read_bytes()
        records = [inspect_input(self.source, kind, "strict480")
                   for kind in ("generated_synthetic_trace", "user_upload", "google_drive_file")]
        for record in records:
            self.assertEqual(record["input_sha256"], digest(original))
            self.assertEqual(record["prepared_rgb_pixel_sha256"], digest(image.tobytes()))
            self.assertNotIn("preview_sha256", record)
        self.assertEqual(self.source.read_bytes(), original)

    def test_center_crop_discards_only_outer_columns_before_preview_resize(self):
        image = Image.new("RGB", (6, 4), (0, 128, 0))
        for y in range(4):
            image.putpixel((0, y), (255, 0, 0))
            image.putpixel((5, y), (0, 0, 255))
        image.save(self.source)
        original = self.source.read_bytes()
        record = inspect_input(self.source, "google_drive_file", "center-crop", self.preview)
        expected = Image.new("RGB", (480, 480), (0, 128, 0))
        with Image.open(self.preview) as displayed:
            self.assertEqual(displayed.tobytes(), expected.tobytes())
        self.assertEqual(record["size"], [6, 4])
        self.assertEqual(record["prepared_rgb_pixel_sha256"], digest(expected.tobytes()))
        self.assertEqual(self.source.read_bytes(), original)

    def test_padding_keeps_full_rectangle_and_adds_white_above_and_below(self):
        self.image(size=(6, 4), color=(0, 128, 0))
        record = inspect_input(self.source, "user_upload", "pad", self.preview)
        with Image.open(self.preview) as displayed:
            self.assertEqual(displayed.size, (480, 480))
            self.assertEqual(displayed.getpixel((240, 0)), (255, 255, 255))
            self.assertEqual(displayed.getpixel((240, 479)), (255, 255, 255))
            self.assertEqual(displayed.getpixel((0, 240)), (0, 128, 0))
            self.assertEqual(displayed.getpixel((479, 240)), (0, 128, 0))
            self.assertEqual(record["prepared_rgb_pixel_sha256"], digest(displayed.tobytes()))

    def test_square_resize_changes_dimensions_without_changing_constant_pixels(self):
        self.image(size=(32, 32), color=(30, 80, 140))
        record = inspect_input(self.source, "user_upload", "square-resize", self.preview)
        expected = Image.new("RGB", (480, 480), (30, 80, 140))
        self.assertEqual(record["size"], [32, 32])
        self.assertEqual(record["prepared_rgb_pixel_sha256"], digest(expected.tobytes()))

    def test_invalid_size_and_square_only_policy_fail_without_preview(self):
        self.image(size=(640, 400))
        for policy in ("strict480", "square-resize"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                inspect_input(self.source, "user_upload", policy, self.preview)
            self.assertFalse(self.preview.exists())

    def test_unsupported_image_does_not_leak_private_path_or_create_preview(self):
        self.source.write_bytes(b"%PDF-1.7\nnot an image")
        with self.assertRaises(ValueError) as caught:
            inspect_input(self.source, "user_upload", "strict480", self.preview)
        self.assertNotIn(self.source.name, str(caught.exception))
        self.assertNotIn(str(self.root), str(caught.exception))
        self.assertFalse(self.preview.exists())

    def test_preview_cannot_replace_the_input_or_existing_output(self):
        self.image()
        original = self.source.read_bytes()
        with self.assertRaises((ValueError, OSError)):
            inspect_input(self.source, "user_upload", "strict480", self.source)
        self.assertEqual(self.source.read_bytes(), original)
        self.preview.write_bytes(b"keep-existing-output")
        with self.assertRaises((ValueError, OSError)):
            inspect_input(self.source, "user_upload", "strict480", self.preview)
        self.assertEqual(self.preview.read_bytes(), b"keep-existing-output")

    def test_unknown_source_kind_is_rejected(self):
        self.image()
        with self.assertRaises(ValueError):
            inspect_input(self.source, "unexpected_source", "strict480", self.preview)
        self.assertFalse(self.preview.exists())

    def test_generated_image_preserves_the_existing_default_pixels(self):
        generate_synthetic(self.source)
        first = self.source.read_bytes()
        with Image.open(self.source) as image:
            self.assertEqual(image.size, (480, 480))
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(digest(image.tobytes()),
                             "8a6b16c565b015654871a39809f6f9a09be998ae3c7be737778909bcca722ad1")
        generate_synthetic(self.source)
        self.assertEqual(self.source.read_bytes(), first)

    def test_generated_image_does_not_overwrite_a_different_existing_input(self):
        original = b"different existing contents"
        self.source.write_bytes(original)
        with self.assertRaises((ValueError, OSError, RuntimeError)):
            generate_synthetic(self.source)
        self.assertEqual(self.source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
