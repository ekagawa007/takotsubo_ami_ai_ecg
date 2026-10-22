"""Progress regression checks with synthetic inputs and tiny fake networks.

Run from the repository root: python -m unittest discover -s tests -p test_progress.py
No trained checkpoints, patient data, network access or GPU are required.
"""

from contextlib import contextmanager
from dataclasses import replace
import io
import json
from pathlib import Path
import random
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

from takotsubo_ami_ai_ecg import cli, inference
from takotsubo_ami_ai_ecg.progress import JSON_PREFIX, ProgressReporter
from takotsubo_ami_ai_ecg.settings import InferenceSettings, ModelSettings


class ProgressTests(unittest.TestCase):
    def setUp(self):
        # Windows sync/indexing services can briefly retain a generated file
        # after it is closed; cleanup is unrelated to prediction correctness.
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.images = []
        for index, value in enumerate((32, 96, 192)):
            path = self.directory / f"synthetic_{index}.png"
            Image.new("RGB", (480, 480), (value, value, value)).save(path)
            self.images.append(path)
        self.weights = self.directory / "weights"
        for name in ("vit", "convnextv2", "mlpmixer"):
            folder = self.weights / name
            folder.mkdir(parents=True)
            # Only existence is checked: _load_model is replaced below.
            (folder / "model.safetensors").write_bytes(b"test placeholder")
        self.settings = replace(
            InferenceSettings(), device="cpu", num_workers=0, batch_size=2,
            vit=ModelSettings(2, 1), convnextv2=ModelSettings(1, 5),
            mlpmixer=ModelSettings(3, 3),
        )

    @contextmanager
    def fake_networks(self, visits=None, invalid_model=None):
        """Keep actual image validation, DataLoader, TTA and ensemble flow."""
        visits = [] if visits is None else visits

        def transform_factory(name, augment, magnitude, policy):
            def transform(image):
                value = image.getpixel((0, 0))[0] / 255.0
                if augment:
                    # Exercise all three RNGs used by the inference loop.
                    value += (random.random() + float(np.random.random())
                              + float(torch.rand(()))) / 100.0
                return torch.tensor([value], dtype=torch.float32)
            return transform

        def load_model(name, path, device):
            class TinyNetwork(torch.nn.Module):
                def forward(self, images):
                    values = images[:, 0]
                    visits.append((name, values.tolist()))
                    scale = {"vit": 1.0, "convnextv2": 2.0, "mlpmixer": 3.0}[name]
                    logits = torch.stack((values * scale, -values * scale), dim=1)
                    if name == invalid_model:
                        logits = logits[:, :1]
                    return SimpleNamespace(logits=logits) if name == "vit" else logits
            return TinyNetwork(), {"format": "synthetic-test"}

        with patch.object(inference, "build_transform", side_effect=transform_factory), \
                patch.object(inference, "_load_model", side_effect=load_model):
            yield visits

    def arguments(self, output, mode="json"):
        args = ["--weights-dir", str(self.weights), "--output", str(output),
                "--device", "cpu", "--num-workers", "0", "--batch-size", "2",
                "--vit-tta", "2", "--convnextv2-tta", "1", "--mlpmixer-tta", "3",
                "--progress", mode]
        for path in self.images:
            args.extend(["--image", str(path)])
        return args

    @staticmethod
    def json_events(stream):
        return [json.loads(line[len(JSON_PREFIX):])
                for line in stream.getvalue().splitlines() if line.startswith(JSON_PREFIX)]

    def test_progress_preserves_predictions_input_order_and_all_rng_states(self):
        results, visits, states = [], [], []
        for enabled in (False, True):
            events = []
            reporter = ProgressReporter("json", stream=io.StringIO())

            def report(event):
                events.append(event.copy())
                reporter(event)

            with self.fake_networks() as seen:
                results.append(inference.predict(
                    self.images, self.weights, settings=self.settings,
                    progress_callback=report if enabled else None))
                visits.append(seen)
                states.append((torch.get_rng_state().clone(), np.random.get_state(), random.getstate()))
            if enabled:
                self.assertTrue(any(event["event"] == "progress" for event in events))
        self.assertEqual(results[0]["predictions"], results[1]["predictions"])
        self.assertEqual(visits[0], visits[1])
        # The first pass processes both a full batch and the final single image.
        self.assertEqual([len(batch) for name, batch in visits[0]][:2], [2, 1])
        np.testing.assert_array_equal(states[0][0].numpy(), states[1][0].numpy())
        self.assertEqual(states[0][1][0], states[1][1][0])
        np.testing.assert_array_equal(states[0][1][1], states[1][1][1])
        self.assertEqual(states[0][1][2:], states[1][1][2:])
        self.assertEqual(states[0][2], states[1][2])

    def test_batch_counts_cover_remainders_tta_offsets_and_selected_models(self):
        for selection, selected in (("ensemble", ("vit", "convnextv2", "mlpmixer")),
                                    ("mlpmixer", ("mlpmixer",))):
            with self.subTest(selection=selection), self.fake_networks():
                events = []
                inference.predict(self.images, self.weights, selection, self.settings,
                                  progress_callback=events.append)
            progress = [event for event in events if event["event"] == "progress"]
            expected = []
            offset = 0
            for model_index, name in enumerate(selected, 1):
                tta = getattr(self.settings, name).tta
                for view in range(1, tta + 1):
                    for count in (2, 3):
                        expected.append((offset + (view - 1) * 3 + count,
                                         name, model_index, view, tta, count))
                offset += tta * 3
            self.assertEqual(
                [(event["completed"], event["model"], event["model_index"],
                  event["tta_index"], event["tta_total"], event["images_completed"])
                 for event in progress], expected)
            self.assertTrue(all(event["total"] == offset and event["images_total"] == 3
                                and event["model_count"] == len(selected) for event in progress))
            self.assertFalse(any(event["event"] == "complete" for event in events))

    def test_invalid_model_or_logits_never_report_completed_inference(self):
        for selection, invalid in (("unknown", None), ("ensemble", "convnextv2")):
            events = []
            with self.subTest(selection=selection), self.fake_networks(invalid_model=invalid), \
                    self.assertRaises(ValueError):
                inference.predict(self.images, self.weights, selection, self.settings,
                                  progress_callback=events.append)
            self.assertFalse(any(event["event"] == "complete" for event in events))
            batches = [event for event in events if event["event"] == "progress"]
            if invalid:
                self.assertEqual(batches[-1]["completed"], 6)
                self.assertTrue(all(event["model"] == "vit" for event in batches))
                self.assertLess(batches[-1]["completed"], batches[-1]["total"])
            else:
                self.assertEqual(batches, [])

    def test_cli_json_and_off_preserve_saved_predictions_without_private_paths(self):
        saved = []
        for mode in ("json", "off"):
            output = self.directory / f"predictions_{mode}.json"
            stderr, stdout = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), self.fake_networks(), \
                    patch("sys.stderr", stderr), patch("sys.stdout", stdout):
                self.assertEqual(cli.main(self.arguments(output, mode)), 0)
            saved.append(json.loads(output.read_text(encoding="utf-8"))["predictions"])
            self.assertIn("Saved 3 prediction(s)", stdout.getvalue())
            events = self.json_events(stderr)
            if mode == "off":
                self.assertEqual(stderr.getvalue(), "")
                continue
            self.assertEqual(events[-1]["event"], "complete")
            self.assertEqual((events[-1]["completed"], events[-1]["total"]), (18, 18))
            self.assertEqual(sum(event["event"] == "complete" for event in events), 1)
            self.assertTrue(all(line.startswith(JSON_PREFIX) for line in stderr.getvalue().splitlines()))
            for event in events:
                self.assertEqual(set(event), {
                    "event", "message", "completed", "total", "model", "model_index", "model_count",
                    "tta_index", "tta_total", "images_completed", "images_total", "elapsed_seconds"})
                self.assertGreaterEqual(event["elapsed_seconds"], 0)
            self.assertNotIn(str(self.directory), stderr.getvalue())
            self.assertNotIn("synthetic_", stderr.getvalue())
            saving = next(event for event in events if event["message"] == "Saving predictions.")
            for key in ("model", "model_index", "model_count", "tta_index", "tta_total",
                        "images_completed", "images_total"):
                self.assertIsNone(saving[key])
        self.assertEqual(saved[0], saved[1])

    def test_cli_write_failure_after_final_batch_emits_error_not_complete(self):
        output = self.directory / "cannot_save.json"
        stderr, stdout = io.StringIO(), io.StringIO()
        with self.fake_networks(), patch("sys.stderr", stderr), patch("sys.stdout", stdout), \
                patch.object(cli, "write_result", side_effect=OSError("private-file-path")), \
                self.assertRaises(SystemExit) as error:
            cli.main(self.arguments(output))
        self.assertEqual(error.exception.code, 2)
        events = self.json_events(stderr)
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["completed"], events[-1]["total"])
        self.assertFalse(any(event["event"] == "complete" for event in events))
        self.assertFalse(output.exists())
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("private-file-path", stderr.getvalue())

    def test_cli_validation_failure_and_interruption_do_not_claim_success(self):
        cases = (("validation", None, 2), ("interrupt", KeyboardInterrupt(), 130))
        for name, failure, exit_code in cases:
            output = self.directory / f"{name}.json"
            args = self.arguments(output)
            if failure is None:
                args.extend(["--batch-size", "0"])
            stderr, stdout = io.StringIO(), io.StringIO()
            with self.subTest(name=name), patch("sys.stderr", stderr), patch("sys.stdout", stdout), \
                    patch.object(inference, "predict", side_effect=failure) as predict_mock, \
                    self.assertRaises(SystemExit) as error:
                cli.main(args)
            self.assertEqual(error.exception.code, exit_code)
            if failure is None:
                predict_mock.assert_not_called()
            events = self.json_events(stderr)
            self.assertEqual(events[-1]["event"], "error")
            self.assertFalse(any(event["event"] == "complete" for event in events))
            self.assertFalse(output.exists())
            self.assertEqual(stdout.getvalue(), "")

    def test_terminal_and_redirected_auto_modes_keep_final_status_readable(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        for mode, stream, uses_bar in (("auto", io.StringIO(), False),
                                       ("auto", Terminal(), True),
                                       ("bar", io.StringIO(), True)):
            with self.subTest(mode=mode, uses_bar=uses_bar), \
                    patch("takotsubo_ami_ai_ecg.progress.shutil.get_terminal_size",
                          return_value=SimpleNamespace(columns=80)):
                reporter = ProgressReporter(mode, stream=stream, clock=lambda: 2.0)
                reporter({"event": "progress", "message": "Running inference.",
                          "completed": 3, "total": 3, "model": "vit", "model_index": 1,
                          "model_count": 1, "tta_index": 1, "tta_total": 1,
                          "images_completed": 3, "images_total": 3})
                reporter.stage("Saving predictions.")
                reporter.complete("Predictions saved.")
                reporter.close()
                text = stream.getvalue()
                self.assertIn("100.0%", text)
                self.assertIn("saved", text.lower())
                self.assertTrue(text.endswith("\n"))
                self.assertEqual("\r" in text, uses_bar)
                if uses_bar:
                    self.assertTrue(all(len(frame) <= 79 for frame in text.replace("\r", "\n").splitlines()))
                final_status = text.replace("\r", "\n").splitlines()[-1]
                self.assertNotIn("TTA", final_status)
                self.assertNotIn("ViT", final_status)


if __name__ == "__main__":
    unittest.main()
