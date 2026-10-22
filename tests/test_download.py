"""Download integrity checks with tiny synthetic ZIPs and no network access.

Run from the repository root:
    python -m unittest discover -s tests -p test_download.py -v
"""

import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from http.client import IncompleteRead
import io
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takotsubo_ami_ai_ecg import download


class WeightDownloadTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="ecg-download-test-"))
        self.addCleanup(self.cleanup_workspace)
        self.output = self.workspace / "weights"
        self.files = {
            "weights/vit/model.safetensors": b"synthetic-vit-tensor-bytes",
            "weights/vit/config.json": b'{"model_type":"vit"}\n',
            "weights/convnextv2/model.safetensors": b"synthetic-convnext-tensor-bytes",
            "weights/mlpmixer/model.safetensors": b"synthetic-mixer-tensor-bytes",
        }

    def cleanup_workspace(self):
        resolved = self.workspace.resolve()
        if resolved.parent == Path(tempfile.gettempdir()).resolve() and resolved.name.startswith("ecg-download-test-"):
            shutil.rmtree(self.workspace, ignore_errors=True)

    def archive_fixture(self, entries=None, filename="weights.zip"):
        """Create regular-file ZIP entries unless a ZipInfo is supplied."""
        target = self.workspace / filename
        if entries is None:
            entries = list(self.files.items())
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member, payload in entries:
                if isinstance(member, zipfile.ZipInfo):
                    info = member
                else:
                    info = zipfile.ZipInfo(member)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload)
        descriptor = {
            "schema_version": 1,
            "version": "0.2.0",
            "url": (
                "https://github.com/ekagawa007/takotsubo_ami_ai_ecg/"
                "releases/download/v0.2.0/synthetic-test-weights.zip"
            ),
            "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "archive_size_bytes": target.stat().st_size,
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                for name, data in self.files.items()
            },
        }
        return target, descriptor

    def assert_installed(self, output=None):
        output = self.output if output is None else output
        for member, payload in self.files.items():
            installed = output.joinpath(*member.split("/")[1:])
            self.assertEqual(installed.read_bytes(), payload, member)

    def assert_no_payload_installed(self, output=None):
        output = self.output if output is None else output
        for member in self.files:
            self.assertFalse(output.joinpath(*member.split("/")[1:]).exists(), member)

    def install(self, archive, descriptor, output=None):
        return download.download_weights(
            self.output if output is None else output,
            archive=archive,
            quiet=True,
            _descriptor=descriptor,
        )

    def test_verified_archive_installs_and_preserves_repository_readme(self):
        archive, descriptor = self.archive_fixture()
        self.output.mkdir()
        readme = self.output / "README.md"
        readme.write_text("Repository weight instructions.\n", encoding="utf-8")

        returned = self.install(archive, descriptor)

        self.assertEqual(returned, self.output.resolve())
        self.assert_installed()
        self.assertEqual(readme.read_text(encoding="utf-8"), "Repository weight instructions.\n")
        self.assertFalse((self.output / "weights").exists())

    def test_matching_installed_assets_are_reused_without_network(self):
        archive, descriptor = self.archive_fixture()
        self.install(archive, descriptor)
        previous = {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}

        with patch.object(download, "_download_archive", side_effect=AssertionError("Unexpected network access")) as network:
            returned = download.download_weights(self.output, quiet=True, _descriptor=descriptor)

        network.assert_not_called()
        self.assertEqual(returned, self.output.resolve())
        self.assert_installed()
        self.assertEqual(previous, {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()})

    def test_archive_digest_mismatch_cannot_install_any_payload(self):
        archive, descriptor = self.archive_fixture()
        descriptor["archive_sha256"] = "0" * 64

        with self.assertRaises(download.DownloadError):
            self.install(archive, descriptor)

        self.assert_no_payload_installed()

    def test_member_digest_mismatch_cannot_install_partial_payload(self):
        entries = list(self.files.items())
        name, payload = entries[-1]
        entries[-1] = (name, b"X" * len(payload))
        archive, descriptor = self.archive_fixture(entries)

        with self.assertRaises(download.DownloadError):
            self.install(archive, descriptor)

        self.assert_no_payload_installed()

    def test_unsafe_zip_members_are_rejected_before_installation(self):
        symlink = zipfile.ZipInfo("weights/redirect")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        directory = zipfile.ZipInfo("weights/directory/")
        directory.create_system = 3
        directory.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
        unsafe = [
            ("parent traversal", "../escaped.txt", b"escape"),
            ("nested traversal", "weights/../../escaped.txt", b"escape"),
            ("absolute", "/weights/escaped.txt", b"escape"),
            ("backslash", "weights\\escaped.txt", b"escape"),
            ("unknown", "weights/unexpected.txt", b"unknown"),
            ("case duplicate", "weights/VIT/model.safetensors", self.files["weights/vit/model.safetensors"]),
            ("symlink", symlink, b"../../escaped.txt"),
            ("directory", directory, b""),
        ]
        for index, (label, member, payload) in enumerate(unsafe):
            with self.subTest(label=label):
                archive, descriptor = self.archive_fixture(
                    [*self.files.items(), (member, payload)], filename=f"unsafe_{index}.zip"
                )
                target = self.workspace / f"output_{index}"
                with self.assertRaises(download.DownloadError):
                    self.install(archive, descriptor, output=target)
                self.assert_no_payload_installed(target)
                self.assertFalse((self.workspace / "escaped.txt").exists())

    def test_member_expanded_size_must_match_descriptor(self):
        entries = list(self.files.items())
        member = "weights/vit/config.json"
        entries = [(name, b"A" * 65536 if name == member else data) for name, data in entries]
        archive, descriptor = self.archive_fixture(entries)
        # The ZIP checksum is correct, but its expanded member exceeds the
        # trusted descriptor size; compressed length cannot establish safety.
        with self.assertRaises(download.DownloadError):
            self.install(archive, descriptor)
        self.assert_no_payload_installed()

    def test_conflicting_local_file_is_not_overwritten_or_downloaded_over(self):
        _archive, descriptor = self.archive_fixture()
        target = self.output / "vit" / "model.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"valuable-existing-model")

        with patch.object(download, "_download_archive", side_effect=AssertionError("Unexpected network access")) as network:
            with self.assertRaises(download.DownloadError):
                download.download_weights(self.output, quiet=True, _descriptor=descriptor)

        network.assert_not_called()
        self.assertEqual(target.read_bytes(), b"valuable-existing-model")
        self.assertFalse((self.output / "convnextv2" / "model.safetensors").exists())

    def test_output_symlink_ancestor_cannot_redirect_installation(self):
        archive, descriptor = self.archive_fixture()
        real = self.workspace / "real"
        real.mkdir()
        link = self.workspace / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("This platform or account cannot create directory symlinks")

        with self.assertRaises(download.DownloadError):
            self.install(archive, descriptor, output=link / "weights")

        self.assertFalse((real / "weights").exists())

    def test_failed_download_cleans_staging_and_preserves_existing_files(self):
        _archive, descriptor = self.archive_fixture()
        self.output.mkdir()
        readme = self.output / "README.md"
        readme.write_text("Keep these instructions.\n", encoding="utf-8")
        before = set(self.workspace.iterdir())

        failures = (
            IncompleteRead(b"private-transport-detail", 100),
            ConnectionError("private-transport-detail"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                class InterruptedResponse(io.BytesIO):
                    headers = {"Content-Length": str(descriptor["archive_size_bytes"])}

                    def geturl(self):
                        return descriptor["url"]

                    def read(self, size=-1):
                        if self.tell() == 0:
                            return super().read(min(size, 7))
                        raise failure

                response = InterruptedResponse(b"partial-network-response")
                opener = SimpleNamespace(open=lambda *args, **kwargs: response)
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(download, "_read_descriptor", return_value=descriptor), \
                        patch.object(download, "build_opener", return_value=opener), \
                        redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as stopped:
                        download.main(["--output-dir", str(self.output), "--quiet"])

                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("The weight download could not be completed", stderr.getvalue())
                self.assertNotIn("private-transport-detail", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(set(self.workspace.iterdir()), before)
                self.assertEqual(list(self.output.iterdir()), [readme])
                self.assertEqual(readme.read_text(encoding="utf-8"), "Keep these instructions.\n")
                self.assert_no_payload_installed()

    def test_downloaded_archive_is_verified_before_success(self):
        archive, descriptor = self.archive_fixture()

        def local_download(_descriptor, destination, timeout, quiet):
            self.assertEqual(timeout, 7)
            shutil.copyfile(archive, destination)

        with patch.object(download, "_download_archive", side_effect=local_download) as network:
            returned = download.download_weights(self.output, timeout=7, quiet=True, _descriptor=descriptor)

        network.assert_called_once()
        self.assertEqual(returned, self.output.resolve())
        self.assert_installed()

    def test_descriptor_cannot_replace_release_url_with_plain_http(self):
        _archive, descriptor = self.archive_fixture()
        descriptor = copy.deepcopy(descriptor)
        descriptor["url"] = descriptor["url"].replace("https://", "http://", 1)
        with patch.object(download, "_download_archive", side_effect=AssertionError("Unexpected network access")) as network:
            with self.assertRaises(download.DownloadError):
                download.download_weights(self.output, quiet=True, _descriptor=descriptor)
        network.assert_not_called()
        self.assert_no_payload_installed()

    def test_cli_rejects_bad_input_without_claiming_completion(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "download_weights.py"),
             "--output-dir", str(self.output),
             "--archive"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("expected one argument", completed.stderr)
        self.assertNotIn("Traceback (most recent call last)", completed.stderr)
        self.assertNotIn("weights are ready", completed.stdout)
        self.assert_no_payload_installed()


if __name__ == "__main__":
    unittest.main()
