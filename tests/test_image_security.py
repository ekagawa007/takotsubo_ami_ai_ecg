"""Small image fixtures for content restrictions and repeated worker reads."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw

from takotsubo_ami_ai_ecg import inference
from takotsubo_ami_ai_ecg.image_input import ImageInputError, open_supported_image


EPS_HEADER = (
    b"%!PS-Adobe-3.0 EPSF-3.0\n"
    b"%%BoundingBox: 0 0 480 480\n"
    b"%%EndComments\nshowpage\n%%EOF\n"
)


class ImageSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.image = Image.new("RGB", (480, 480), (239, 221, 207))
        draw = ImageDraw.Draw(self.image)
        draw.line([(0, 200), (170, 200), (190, 100), (210, 300), (230, 200), (479, 200)],
                  fill=(19, 37, 67), width=3)

    def test_supported_single_frames_keep_existing_decoded_rgb_pixels(self):
        for format_name, suffix in (("PNG", ".png"), ("JPEG", ".jpg"),
                                    ("BMP", ".bmp"), ("TIFF", ".tiff"),
                                    ("WEBP", ".webp")):
            with self.subTest(format=format_name):
                path = self.directory / ("synthetic" + suffix)
                self.image.save(path, format=format_name)
                with Image.open(path) as previous:
                    expected = previous.convert("RGB").tobytes()
                self.assertEqual(inference._validate_images([path], "strict480"),
                                 [inference.sha256_file(path)])
                dataset = inference.ImageDataset([path], lambda image: image.tobytes())
                self.assertEqual(dataset[0], expected)

    def test_supported_content_is_not_inferred_from_filename(self):
        path = self.directory / "synthetic.jpg"
        self.image.save(path, format="PNG")
        with open_supported_image(path, 1) as image:
            self.assertEqual(image.format, "PNG")

    def test_palette_and_grayscale_conversion_keep_existing_pixels(self):
        for mode in ("L", "P", "RGBA"):
            with self.subTest(mode=mode):
                path = self.directory / ("synthetic_" + mode + ".png")
                self.image.convert(mode).save(path)
                with Image.open(path) as previous:
                    expected = previous.convert("RGB").tobytes()
                self.assertEqual(inference.ImageDataset([path], lambda image: image.tobytes())[0],
                                 expected)

    def test_eps_disguised_as_png_never_reaches_eps_decoder_or_subprocess(self):
        path = self.directory / "synthetic.png"
        path.write_bytes(EPS_HEADER)
        with patch("PIL.EpsImagePlugin.EpsImageFile._open",
                   side_effect=AssertionError("EPS decoder must not be reached")) as eps_open, \
                patch("subprocess.Popen",
                      side_effect=AssertionError("No external renderer may run")) as process:
            with self.assertRaisesRegex(ImageInputError, "PNG, JPEG, BMP, TIFF or WebP"):
                inference._validate_images([path], "strict480")
            with self.assertRaises(ImageInputError):
                inference.ImageDataset([path], lambda image: image.tobytes())[0]
        eps_open.assert_not_called()
        process.assert_not_called()

    def test_file_replaced_after_validation_is_rejected_at_actual_read(self):
        path = self.directory / "synthetic.png"
        self.image.save(path)
        self.assertEqual(len(inference._validate_images([path], "strict480")), 1)
        dataset = inference.ImageDataset([path], lambda image: image.tobytes())
        path.write_bytes(EPS_HEADER)
        with self.assertRaises(ImageInputError):
            dataset[0]

    def test_multiframe_supported_formats_are_rejected_at_both_reads(self):
        second = Image.new("RGB", (480, 480), (10, 20, 30))
        for format_name, suffix in (("PNG", ".png"), ("TIFF", ".tiff"), ("WEBP", ".webp")):
            with self.subTest(format=format_name):
                path = self.directory / ("multiple" + suffix)
                self.image.save(path, format=format_name, save_all=True,
                                append_images=[second], duration=50, loop=0)
                with self.assertRaisesRegex(ImageInputError, "exactly one frame"):
                    inference._validate_images([path], "strict480")
                with self.assertRaisesRegex(ImageInputError, "exactly one frame"):
                    inference.ImageDataset([path], lambda image: image.tobytes())[0]

    def test_corrupt_or_unsupported_content_has_no_private_path_in_error(self):
        path = self.directory / "private_identifier.png"
        contents = (b"not an image", b"\x89PNG\r\n\x1a\n", b"%PDF-1.7\n", b"<svg/>")
        for content in contents:
            with self.subTest(content=content):
                path.write_bytes(content)
                for operation in (lambda: inference._validate_images([path], "strict480"),
                                  lambda: inference.ImageDataset([path], lambda image: image)[0]):
                    with self.assertRaises(ValueError) as caught:
                        operation()
                    self.assertNotIn(path.name, str(caught.exception))
                    self.assertNotIn(str(self.directory), str(caught.exception))

    def test_unsupported_raster_format_is_rejected(self):
        path = self.directory / "synthetic.png"
        self.image.save(path, format="GIF")
        with self.assertRaises(ImageInputError):
            inference._validate_images([path], "strict480")

    def test_strict_size_and_explicit_resize_policy_keep_existing_behavior(self):
        path = self.directory / "synthetic.png"
        self.image.resize((320, 240)).save(path)
        with self.assertRaisesRegex(ValueError, "480x480"):
            inference._validate_images([path], "strict480")
        self.assertEqual(len(inference._validate_images([path], "resize")), 1)
        transform = inference.build_transform("vit", False, 1, "resize")
        self.assertEqual(tuple(inference.ImageDataset([path], transform)[0].shape), (3, 480, 480))


if __name__ == "__main__":
    unittest.main()
