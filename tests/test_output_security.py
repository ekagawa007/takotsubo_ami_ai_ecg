"""Filesystem regression checks with disposable, non-patient fixtures only."""

from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takotsubo_ami_ai_ecg import cli, safe_output


def example_result():
    return {"metadata": {"number_of_images": 1}, "predictions": [{
        "sample_id": "sample_000001", "input_sha256": "0" * 64,
        "pred_id": 1, "pred_label": "AMI", "logits": [0.0, 1.0],
        "probabilities": [0.25, 0.75],
        "models": {"vit": {"logits": [0.0, 1.0], "probabilities": [0.25, 0.75]}},
    }]}


class OutputSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ecg-output-test-", dir=os.environ.get("ECG_TEST_TMPDIR"),
                                               ignore_cleanup_errors=True)
        self.directory = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def symlink(self, link, target, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable in this environment.")

    def hardlink(self, link, target):
        try:
            os.link(target, link)
        except (NotImplementedError, OSError):
            self.skipTest("Hard links are unavailable in this environment.")

    def test_json_and_csv_success_preserve_public_schema(self):
        result = example_result()
        output = self.directory / "nested" / "prediction.json"
        cli.write_result(result, output)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
        output = output.with_suffix(".csv")
        cli.write_result(result, output)
        rows = list(csv.DictReader(io.StringIO(output.read_text(encoding="utf-8"))))
        self.assertEqual(rows[0]["sample_id"], "sample_000001")
        self.assertEqual(rows[0]["prob_AMI"], "0.75")
        self.assertEqual(json.loads(output.with_suffix(".metadata.json").read_text()), result["metadata"])

    def test_no_overwrite_preserves_existing_file(self):
        output = self.directory / "prediction.json"
        output.write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "already exists"):
            cli.write_result(example_result(), output)
        self.assertEqual(output.read_bytes(), b"original")

    def test_existing_sidecar_prevents_creating_csv(self):
        output = self.directory / "prediction.csv"
        sidecar = output.with_suffix(".metadata.json")
        sidecar.write_bytes(b"original metadata")
        with self.assertRaisesRegex(ValueError, "already exists"):
            cli.write_result(example_result(), output)
        self.assertFalse(output.exists())
        self.assertEqual(sidecar.read_bytes(), b"original metadata")

    def test_protected_hardlink_and_sidecar_hardlink_are_rejected(self):
        protected = self.directory / "input.png"
        protected.write_bytes(b"private image fixture")
        for filename in ("alias.json", "prediction.metadata.json"):
            alias = self.directory / filename
            self.hardlink(alias, protected)
            output = alias if filename == "alias.json" else self.directory / "prediction.csv"
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "must not overwrite"):
                cli.write_result(example_result(), output, True, protected_paths=[protected])
            self.assertEqual(protected.read_bytes(), b"private image fixture")
            self.assertEqual(alias.read_bytes(), b"private image fixture")

    def test_explicit_overwrite_replaces_entry_without_truncating_hardlink(self):
        other = self.directory / "other.json"
        other.write_bytes(b"unrelated data")
        output = self.directory / "prediction.json"
        self.hardlink(output, other)
        cli.write_result(example_result(), output, True)
        self.assertEqual(other.read_bytes(), b"unrelated data")
        self.assertEqual(json.loads(output.read_text()), example_result())
        self.assertFalse(os.path.samefile(other, output))

    def test_output_and_sidecar_symlinks_are_rejected(self):
        target = self.directory / "preserve.json"
        target.write_bytes(b"preserve")
        for name in ("alias.json", "prediction.metadata.json"):
            link = self.directory / name
            self.symlink(link, target)
            output = link if name == "alias.json" else self.directory / "prediction.csv"
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "ordinary files"):
                cli.write_result(example_result(), output, True)
            self.assertEqual(target.read_bytes(), b"preserve")

    def test_linked_parent_is_rejected(self):
        target = self.directory / "real"
        target.mkdir()
        link = self.directory / "linked"
        self.symlink(link, target, directory=True)
        with self.assertRaisesRegex(ValueError, "ordinary directories"):
            cli.write_result(example_result(), link / "prediction.json")
        self.assertEqual(list(target.iterdir()), [])

    def test_two_simultaneous_creators_do_not_overwrite_each_other(self):
        output = self.directory / "prediction.json"
        barrier = threading.Barrier(2)

        def save(value):
            barrier.wait(timeout=10)
            try:
                cli.write_result({"value": value}, output)
                return True, value
            except ValueError:
                return False, value

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(save, (1, 2)))
        successful = [value for success, value in results if success]
        self.assertEqual(len(successful), 1)
        self.assertEqual(json.loads(output.read_text()), {"value": successful[0]})

    def test_sidecar_created_during_reservation_is_not_overwritten(self):
        output = self.directory / "prediction.csv"
        sidecar = output.with_suffix(".metadata.json")
        original_create = safe_output._Directory.create

        def create(directory, name):
            if name == sidecar.name:
                sidecar.write_bytes(b"concurrent result")
            return original_create(directory, name)

        with patch.object(safe_output._Directory, "create", create), self.assertRaisesRegex(ValueError, "could not be completed"):
            cli.write_result(example_result(), output)
        self.assertEqual(sidecar.read_bytes(), b"concurrent result")
        self.assertEqual(output.read_bytes(), b"")

    def test_invalid_metadata_is_serialized_before_any_file_is_opened(self):
        result = example_result()
        result["metadata"]["invalid"] = float("nan")
        output = self.directory / "prediction.csv"
        with self.assertRaises(ValueError):
            cli.write_result(result, output)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_preparation_failure_preserves_both_existing_files(self):
        output = self.directory / "prediction.csv"
        sidecar = output.with_suffix(".metadata.json")
        output.write_bytes(b"old csv")
        sidecar.write_bytes(b"old metadata")
        original_write = safe_output._write
        calls = 0

        def fail_second(stream, payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated disk full")
            return original_write(stream, payload)

        with patch.object(safe_output, "_write", fail_second), self.assertRaisesRegex(ValueError, "could not be completed"):
            cli.write_result(example_result(), output, True)
        self.assertEqual(output.read_bytes(), b"old csv")
        self.assertEqual(sidecar.read_bytes(), b"old metadata")

    def test_second_replacement_failure_is_reported_as_incomplete(self):
        output = self.directory / "prediction.csv"
        sidecar = output.with_suffix(".metadata.json")
        output.write_bytes(b"old csv")
        sidecar.write_bytes(b"old metadata")
        original_replace = safe_output._Directory.replace
        calls = 0

        def fail_second(directory, source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated replacement failure")
            return original_replace(directory, source, destination)

        with patch.object(safe_output._Directory, "replace", fail_second), self.assertRaisesRegex(ValueError, "could not be completed"):
            cli.write_result(example_result(), output, True)
        self.assertNotEqual(output.read_bytes(), b"old csv")
        self.assertEqual(sidecar.read_bytes(), b"old metadata")

    def test_directory_change_stops_save_without_removing_foreign_file(self):
        output = self.directory / "prediction.json"
        foreign = self.directory / "foreign.json"
        foreign.write_bytes(b"unrelated")
        original_verify = safe_output._Directory.verify
        calls = 0

        def changed(directory):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise safe_output.OutputError("The output directory changed during saving.")
            original_verify(directory)

        with patch.object(safe_output._Directory, "verify", changed), self.assertRaisesRegex(ValueError, "could not be completed"):
            cli.write_result(example_result(), output)
        self.assertEqual(foreign.read_bytes(), b"unrelated")

    @unittest.skipUnless(os.name == "nt", "Windows directory sharing behavior")
    def test_windows_parent_handle_prevents_rename_during_save(self):
        parent = self.directory / "outputs"
        parent.mkdir()
        with safe_output._Directory(parent):
            with self.assertRaises(OSError):
                parent.rename(self.directory / "renamed")
        self.assertTrue(parent.is_dir())

    @unittest.skipUnless(os.name == "nt", "Windows alternate streams")
    def test_windows_alternate_stream_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported output filename"):
            cli.write_result(example_result(), self.directory / "original:stream.json")


if __name__ == "__main__":
    unittest.main()
