"""Grad-CAM artifact and CLI checks using disposable synthetic fixtures."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, PngImagePlugin
from takotsubo_ami_ai_ecg import cli, gradcam_output, safe_output


class GradCAMOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ecg-cam-output-", dir=os.environ.get("ECG_TEST_TMPDIR"), ignore_cleanup_errors=True)
        self.folder = Path(self.temp.name).resolve()
        self.image = self.folder / "private_source.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("private_comment", "do not export source metadata")
        Image.new("RGB", (480, 480), (244, 240, 230)).save(self.image, pnginfo=info)
        self.digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.output = self.folder / "prediction.json"
        self.result = {"metadata": {"model": "ensemble", "number_of_images": 1,
                       "settings": {"input_policy": "strict480"}},
                       "predictions": [{"sample_id": "sample_000001", "input_sha256": self.digest,
                        "pred_id": 1, "pred_label": "AMI", "logits": [0.25, 1.25], "probabilities": [0.3, 0.7],
                        "models": {name: {"logits": [0.25, 1.25], "probabilities": [0.3, 0.7]}
                                   for name in ("vit", "convnextv2", "mlpmixer")}}]}
        values = np.repeat(np.linspace(0, 1, 480, dtype=np.float32)[None, :], 480, axis=0)
        self.cam = {"metadata": {"precision": "float32", "tta": 1}, "images": [{
            "sample_id": "sample_000001", "input_sha256": self.digest,
            "models": {name: {"map": values.copy(), "target_id": 1, "target_label": "AMI",
                       "target_source": "model-predicted", "layers": ["fixture"],
                       "unaugmented_logits": [0.2, 1.2], "normalization": {"mean": [0.485, 0.456, 0.406],
                       "std": [0.229, 0.224, 0.225]}, "map_is_constant": False}
                       for name in ("vit", "convnextv2")}}]}

    def tearDown(self):
        self.temp.cleanup()

    def serialize(self, output=None):
        return gradcam_output.serialize_gradcam([self.image], self.result, self.cam, output or self.output)

    def invoke(self, extra=(), result=None):
        args = ["--image", str(self.image), "--output", str(self.output), "--progress", "off", "--gradcam", *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch("takotsubo_ami_ai_ecg.inference.predict", return_value=copy.deepcopy(result or self.result)) as predict:
                with patch("takotsubo_ami_ai_ecg.gradcam.generate_gradcam", return_value=copy.deepcopy(self.cam)) as generate:
                    status = cli.main(args)
        return status, predict, generate

    def test_outputs_have_generic_names_and_no_source_metadata(self):
        contents = self.serialize()
        self.assertEqual(len(contents), 7)
        self.assertEqual([p for p, _ in contents], gradcam_output.artifact_paths(self.output, 1, ["vit", "convnextv2"]))
        for path, payload in contents:
            self.assertNotIn("private_source", path.name)
            if path.suffix == ".png":
                with Image.open(io.BytesIO(payload)) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (480, 480))
                    self.assertNotIn("private_comment", image.info)
                    self.assertNotIn("exif", image.info)
            else:
                values = np.load(io.BytesIO(payload), allow_pickle=False)
                self.assertEqual(values.dtype, np.float32)
                np.testing.assert_array_equal(values, self.cam["images"][0]["models"]["vit"]["map"])
        metadata = self.result["metadata"]["gradcam"]
        for record, (path, payload) in zip(metadata["artifacts"], contents):
            self.assertEqual(record, {"filename": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
        self.assertNotIn(str(self.folder), json.dumps(self.result))

    def test_overlay_opacity_and_endpoints(self):
        rgb = np.full((480, 480, 3), 200, dtype=np.uint8)
        values = self.cam["images"][0]["models"]["vit"]["map"]
        heat, overlay, constant = gradcam_output.render_map(rgb, values, 0)
        self.assertFalse(constant)
        np.testing.assert_array_equal(overlay, rgb)
        np.testing.assert_array_equal(heat[0, 0], [30, 56, 130])
        np.testing.assert_array_equal(heat[0, -1], [255, 224, 64])
        heat, overlay, _ = gradcam_output.render_map(rgb, values, 1)
        np.testing.assert_array_equal(overlay, heat)

    def test_constant_map_uses_neutral_heatmap_unchanged_overlay(self):
        rgb = np.full((480, 480, 3), 100, dtype=np.uint8)
        for value in (0.0, 0.5):
            heat, overlay, constant = gradcam_output.render_map(rgb, np.full((480, 480), value), .6)
            self.assertTrue(constant)
            self.assertTrue((heat == 128).all())
            np.testing.assert_array_equal(overlay, rgb)

    def test_nonfinite_or_out_of_range_map_and_opacity_rejected(self):
        rgb = np.full((480, 480, 3), 100, dtype=np.uint8)
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                gradcam_output.render_map(rgb, np.full((480, 480), bad), .6)
            with self.subTest(opacity=bad), self.assertRaises(ValueError):
                gradcam_output.render_map(rgb, np.zeros((480, 480)), bad)

    def test_input_drift_rejected_without_writes(self):
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.serialize()
        self.assertFalse(self.output.exists())

    def test_model_and_input_correspondence_rejected(self):
        self.cam["images"][0]["sample_id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "correspondence"):
            self.serialize()
        self.cam["images"][0]["sample_id"] = "sample_000001"
        self.cam["images"][0]["models"]["mlpmixer"] = self.cam["images"][0]["models"].pop("vit")
        with self.assertRaisesRegex(ValueError, "correspondence"):
            self.serialize()

    def test_resize_display_obeys_explicit_input_policy(self):
        Image.new("RGB", (360, 420), "white").save(self.image)
        digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.result["predictions"][0]["input_sha256"] = digest
        self.cam["images"][0]["input_sha256"] = digest
        with self.assertRaisesRegex(ValueError, "480x480"):
            self.serialize()
        self.result["metadata"]["settings"]["input_policy"] = "resize"
        contents = self.serialize()
        with Image.open(io.BytesIO(contents[0][1])) as image:
            self.assertEqual(image.size, (480, 480))

    def test_cli_keeps_prediction_values_exact(self):
        baseline = copy.deepcopy(self.result["predictions"])
        status, predict, generate = self.invoke()
        self.assertEqual(status, 0)
        predict.assert_called_once()
        generate.assert_called_once()
        result = json.loads(self.output.read_text(encoding="utf-8"))
        for row in result["predictions"]:
            row.pop("gradcam")
        self.assertEqual(result["predictions"], baseline)
        self.assertTrue(all(path.is_file() for path in gradcam_output.artifact_paths(self.output, 1, ["vit", "convnextv2"])))

    def test_cli_existing_cam_blocks_inference_and_all_writes(self):
        destination = gradcam_output.artifact_paths(self.output, 1, ["vit", "convnextv2"])[1]
        destination.write_bytes(b"keep")
        with patch("takotsubo_ami_ai_ecg.inference.predict") as predict:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                cli.main(["--image", str(self.image), "--output", str(self.output), "--gradcam", "--progress", "off"])
        self.assertEqual(caught.exception.code, 2)
        predict.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(destination.read_bytes(), b"keep")

    def test_protected_cam_alias_rejected_even_with_overwrite(self):
        destination = gradcam_output.artifact_paths(self.output, 1, ["vit", "convnextv2"])[1]
        try:
            os.link(self.image, destination)
        except (OSError, NotImplementedError):
            self.skipTest("Hard links are unavailable.")
        with patch("takotsubo_ami_ai_ecg.inference.predict") as predict:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["--image", str(self.image), "--output", str(self.output), "--gradcam", "--overwrite", "--progress", "off"])
        predict.assert_not_called()
        self.assertEqual(hashlib.sha256(self.image.read_bytes()).hexdigest(), self.digest)
        self.assertFalse(self.output.exists())

    def test_symlinked_cam_rejected(self):
        destination = gradcam_output.artifact_paths(self.output, 1, ["vit", "convnextv2"])[1]
        try:
            destination.symlink_to(self.image)
        except (OSError, NotImplementedError):
            self.skipTest("Symbolic links are unavailable.")
        with self.assertRaises(ValueError):
            safe_output.validate_outputs([destination], overwrite=True)

    def test_csv_sidecar_retains_per_image_map_manifest(self):
        self.output = self.output.with_suffix(".csv")
        self.invoke()
        metadata = json.loads(self.output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        record = metadata["gradcam"]["images"][0]
        self.assertEqual(record["sample_id"], "sample_000001")
        self.assertTrue((self.output.parent / record["models"]["vit"]["files"]["values"]).is_file())

    def test_invalid_options_fail_before_prediction(self):
        configurations = [["--model", "mlpmixer"], ["--model", "vit", "--cam-target", "ensemble-predicted"],
                          ["--cam-opacity", "nan"], ["--cam-opacity", "1.1"]]
        for extra in configurations:
            with self.subTest(extra=extra), patch("takotsubo_ami_ai_ecg.inference.predict") as predict:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    cli.main(["--image", str(self.image), "--output", str(self.output), "--gradcam", "--progress", "off", *extra])
                self.assertEqual(caught.exception.code, 2)
                predict.assert_not_called()

    def test_nondefault_cam_options_require_gradcam(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["--image", str(self.image), "--output", str(self.output), "--cam-target", "AMI", "--progress", "off"])

    def test_help_does_not_import_ml_libraries(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);from takotsubo_ami_ai_ecg.cli import build_parser;build_parser().format_help();assert not any(name in sys.modules for name in ('torch','torchvision','PIL','numpy','transformers','timm'))"
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "src")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
