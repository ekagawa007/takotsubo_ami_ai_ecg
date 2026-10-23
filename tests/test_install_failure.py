"""Small local fixtures for failed weight installation; no downloads or models."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from takotsubo_ami_ai_ecg import download


class FailedInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ecg-install-failure-", ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stage = self.root / "stage"
        self.output = self.root / "output"
        (self.stage / "vit").mkdir(parents=True)
        self.payload = b"small-public-tensor-fixture"
        (self.stage / "vit/model.safetensors").write_bytes(self.payload)
        self.descriptor = {"files": {"weights/vit/model.safetensors": {
            "size_bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }}}

    def test_parent_swap_does_not_delete_unrelated_same_name_file(self):
        victim = self.root / "unrelated"
        victim.mkdir()
        victim_file = victim / "model.safetensors"
        victim_file.write_bytes(b"keep-this-unrelated-file")
        probe = self.root / "probe"
        try:
            probe.symlink_to(victim, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Directory symlinks are unavailable for this account")
        probe.unlink()
        original = download._existing_assets
        calls = 0

        def swap_before_final_verification(output, descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                model_dir = output / "vit"
                model_dir.rename(self.root / "retained")
                model_dir.symlink_to(victim, target_is_directory=True)
            return original(output, descriptor)

        try:
            with patch.object(download, "_existing_assets", side_effect=swap_before_final_verification):
                with self.assertRaisesRegex(download.DownloadError, "Partial files were retained"):
                    download._install_verified(self.stage, self.output, self.descriptor)
            self.assertEqual(victim_file.read_bytes(), b"keep-this-unrelated-file")
            self.assertEqual((self.root / "retained/model.safetensors").read_bytes(), self.payload)
        finally:
            link = self.output / "vit"
            if link.is_symlink():
                link.unlink()

    def test_incomplete_copy_is_retained_without_removing_existing_files(self):
        self.output.mkdir()
        existing = self.output / "README.md"
        existing.write_bytes(b"keep-existing-instructions")

        def interrupted_copy(source, destination, **kwargs):
            destination.write(source.read(4))
            raise OSError("private-diagnostic-not-for-output")

        with patch.object(download.shutil, "copyfileobj", side_effect=interrupted_copy):
            with self.assertRaisesRegex(download.DownloadError, "Partial files were retained") as raised:
                download._install_verified(self.stage, self.output, self.descriptor)
        self.assertNotIn("private-diagnostic", str(raised.exception))
        self.assertEqual(existing.read_bytes(), b"keep-existing-instructions")
        self.assertEqual((self.output / "vit/model.safetensors").read_bytes(), self.payload[:4])
        with self.assertRaisesRegex(download.DownloadError, "conflicts"):
            download._existing_assets(self.output, self.descriptor)

    def test_final_integrity_failure_never_claims_success(self):
        with patch.object(download, "_existing_assets", side_effect=[set(), set()]):
            with self.assertRaisesRegex(download.DownloadError, "did not complete"):
                download._install_verified(self.stage, self.output, self.descriptor)
        self.assertEqual((self.output / "vit/model.safetensors").read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
