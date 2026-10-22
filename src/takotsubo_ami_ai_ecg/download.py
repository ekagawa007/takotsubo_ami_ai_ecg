"""Download and verify the official tensor-only weight bundle using stdlib only."""

import argparse
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile


_CHUNK = 1024 * 1024
_DESCRIPTOR = Path(__file__).parent / "assets" / "weights_v0.2.0.json"
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


class DownloadError(ValueError):
    """An expected integrity, destination, or download error without private paths."""


def _https_url(url):
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.hostname and parsed.username is None
                 and parsed.password is None and parsed.port in (None, 443) and not parsed.fragment)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise DownloadError("Weight downloads and redirects must use HTTPS without embedded credentials.")
    return url


class _HTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _member_path(name):
    if not isinstance(name, str) or "\\" in name or any(ord(char) < 32 for char in name):
        raise DownloadError("The bundle contains an unsafe member path.")
    parts = name.split("/")
    if (len(parts) < 2 or parts[0] != "weights"
            or any(part in ("", ".", "..") or ":" in part or part.endswith((".", " "))
                   or part.split(".")[0].upper() in _RESERVED for part in parts)):
        raise DownloadError("The bundle contains an unsafe member path.")
    return Path(*parts[1:])


def _validate_descriptor(value):
    if (not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("version") != "0.2.0"
            or not isinstance(value.get("files"), dict) or not value["files"]):
        raise DownloadError("The bundled weight descriptor is invalid.")
    _https_url(value.get("url"))
    if (type(value.get("archive_size_bytes")) is not int or value["archive_size_bytes"] <= 0
            or not isinstance(value.get("archive_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", value["archive_sha256"]) is None):
        raise DownloadError("The bundled archive integrity record is invalid.")
    seen = set()
    for name, details in value["files"].items():
        _member_path(name)
        folded = name.casefold()
        if folded in seen:
            raise DownloadError("The bundled descriptor contains duplicate member paths.")
        seen.add(folded)
        if (not isinstance(details, dict) or type(details.get("size_bytes")) is not int
                or details["size_bytes"] < 0 or not isinstance(details.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", details["sha256"]) is None):
            raise DownloadError("The bundled file integrity record is invalid.")
    return value


def _read_descriptor():
    try:
        return _validate_descriptor(json.loads(_DESCRIPTOR.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise DownloadError("The packaged weight descriptor is missing or unreadable. Reinstall this release.") from None


def _say(message, quiet):
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_link(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    # Junctions and symlinks must not redirect the installation outside its root.
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", None) in (0xA0000003, 0xA000000C):
        raise DownloadError("The output path contains a symbolic link or junction.")
    return info


def _check_directory_path(path):
    for directory in (*reversed(path.parents), path):
        info = _check_link(directory)
        if info is not None and not stat.S_ISDIR(info.st_mode):
            raise DownloadError("An output directory component is not a directory.")


def _ensure_directory(path, created=None):
    for directory in (*reversed(path.parents), path):
        info = _check_link(directory)
        if info is None:
            directory.mkdir()
            if created is not None:
                created.append(directory)
        elif not stat.S_ISDIR(info.st_mode):
            raise DownloadError("An output directory component is not a directory.")


def _existing_assets(output, descriptor):
    _check_directory_path(output)
    present = set()
    for name, expected in descriptor["files"].items():
        target = output / _member_path(name)
        _check_directory_path(target.parent)
        info = _check_link(target)
        if info is None:
            continue
        if (not stat.S_ISREG(info.st_mode) or info.st_size != expected["size_bytes"]
                or _file_hash(target) != expected["sha256"]):
            raise DownloadError("An existing weight-bundle file conflicts with this release. Choose a different output directory; no files were overwritten.")
        present.add(name)
    return present


def _copy_archive(source, destination, descriptor, quiet, label):
    expected_size = descriptor["archive_size_bytes"]
    count = 0
    digest = hashlib.sha256()
    last_report = time.monotonic()
    _say(f"{label}: 0.0% (0/{expected_size} bytes)", quiet)
    with destination.open("xb") as target:
        while True:
            block = source.read(min(_CHUNK, expected_size - count + 1))
            if not block:
                break
            count += len(block)
            if count > expected_size:
                raise DownloadError("The archive exceeds the size recorded for this release.")
            digest.update(block)
            target.write(block)
            now = time.monotonic()
            if count == expected_size or now - last_report >= 0.5:
                _say(f"{label}: {100 * count / expected_size:.1f}% ({count}/{expected_size} bytes)", quiet)
                last_report = now
    if count != expected_size or digest.hexdigest() != descriptor["archive_sha256"]:
        raise DownloadError("Archive size or SHA-256 verification failed. Nothing was installed.")


def _download_archive(descriptor, destination, timeout, quiet):
    request = Request(_https_url(descriptor["url"]), headers={
        "User-Agent": "takotsubo-ami-ai-ecg/0.2.0", "Accept-Encoding": "identity"})
    try:
        with build_opener(_HTTPSRedirectHandler()).open(request, timeout=timeout) as response:
            _https_url(response.geturl())
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    valid_length = int(length) == descriptor["archive_size_bytes"]
                except ValueError:
                    valid_length = False
                if not valid_length:
                    raise DownloadError("The server returned an unexpected archive size.")
            _copy_archive(response, destination, descriptor, quiet, "Downloading weights")
    except HTTPError as error:
        if error.code == 404:
            raise DownloadError("The v0.2.0 weight release asset is not available yet (HTTP 404). Retry after it is published, or use --archive with the official ZIP.") from None
        raise DownloadError(f"The weight download failed (HTTP {error.code}). Retry later or use the official local ZIP.") from None
    except (URLError, TimeoutError, HTTPException, ConnectionError):
        raise DownloadError("The weight download could not be completed. Check the connection or use --archive with the official ZIP.") from None


def _extract_verified(archive, staging, descriptor, quiet):
    # Recheck here as well so callers/tests replacing the transport cannot bypass integrity.
    if archive.stat().st_size != descriptor["archive_size_bytes"] or _file_hash(archive) != descriptor["archive_sha256"]:
        raise DownloadError("Archive size or SHA-256 verification failed. Nothing was installed.")
    _say("Archive verified; checking every bundled file.", quiet)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            names = []
            folded = set()
            for member in members:
                name = member.filename
                _member_path(getattr(member, "orig_filename", name))
                mode = (member.external_attr >> 16) & 0xffff
                if (member.is_dir() or name.casefold() in folded or (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                        or member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise DownloadError("The archive contains a duplicate, linked, encrypted, or unsupported member.")
                names.append(name)
                folded.add(name.casefold())
            if set(names) != set(descriptor["files"]):
                raise DownloadError("The archive members do not exactly match this release descriptor.")
            for index, member in enumerate(members, 1):
                expected = descriptor["files"][member.filename]
                if member.file_size != expected["size_bytes"]:
                    raise DownloadError("A bundled file has an unexpected size.")
                target = staging / _member_path(member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                count = 0
                with bundle.open(member) as source, target.open("xb") as destination:
                    while True:
                        block = source.read(min(_CHUNK, expected["size_bytes"] - count + 1))
                        if not block:
                            break
                        count += len(block)
                        if count > expected["size_bytes"]:
                            raise DownloadError("A bundled file exceeds its recorded size.")
                        digest.update(block)
                        destination.write(block)
                if count != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
                    raise DownloadError("A bundled file failed SHA-256 verification. Nothing was installed.")
                _say(f"Verified bundled files: {index}/{len(members)}", quiet)
    except (zipfile.BadZipFile, RuntimeError, EOFError):
        raise DownloadError("The verified archive could not be read as the expected ZIP bundle.") from None


def _install_verified(staging, output, descriptor):
    present = _existing_assets(output, descriptor)
    created_files, created_dirs = [], []
    try:
        for name in descriptor["files"]:
            if name in present:
                continue
            relative = _member_path(name)
            target = output / relative
            _ensure_directory(target.parent, created_dirs)
            if _check_link(target) is not None:
                raise DownloadError("The output changed during installation. No existing files were overwritten.")
            with target.open("xb") as destination:
                created_files.append(target)
                with (staging / relative).open("rb") as source:
                    shutil.copyfileobj(source, destination, length=_CHUNK)
        if len(_existing_assets(output, descriptor)) != len(descriptor["files"]):
            raise DownloadError("The installed files could not be verified.")
    except BaseException:
        # Roll back only the files/directories created by this invocation.
        for path in reversed(created_files):
            try:
                path.unlink()
            except OSError:
                pass
        for path in reversed(created_dirs):
            try:
                path.rmdir()
            except OSError:
                pass
        raise


def download_weights(output_dir="weights", *, archive=None, quiet=False, timeout=60, _descriptor=None):
    """Return the directory containing hash-verified official v0.2.0 weights.

    ``archive`` optionally supplies a local copy of the official ZIP. Matching
    files are reused; conflicting files are never overwritten. ``_descriptor``
    is an internal fixture hook, not a CLI option or an alternative download URL.
    No model library or checkpoint deserializer is imported.
    """
    descriptor = _read_descriptor() if _descriptor is None else _validate_descriptor(_descriptor)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise DownloadError("Download timeout must be a positive finite number.")
    output = Path(os.path.abspath(Path(output_dir).expanduser()))
    existing = _existing_assets(output, descriptor)
    if len(existing) == len(descriptor["files"]) and archive is None:
        _say("All weight-bundle files already match v0.2.0; no download is needed.", quiet)
        return output
    _ensure_directory(output.parent)
    # Sync/antivirus software can briefly lock temporary files on Windows.
    # Cleanup failure must not turn a fully verified installation into failure;
    # any locked leftovers contain only this public bundle, never user inputs.
    with tempfile.TemporaryDirectory(prefix=".ecg-weights-", dir=output.parent,
                                     ignore_cleanup_errors=True) as temporary:
        temporary = Path(temporary)
        downloaded = temporary / "official.zip"
        if archive is None:
            _download_archive(descriptor, downloaded, timeout, quiet)
        else:
            try:
                with Path(archive).expanduser().open("rb") as source:
                    _copy_archive(source, downloaded, descriptor, quiet, "Reading local archive")
            except OSError:
                raise DownloadError("The local official ZIP could not be read.") from None
        staging = temporary / "verified"
        staging.mkdir()
        _extract_verified(downloaded, staging, descriptor, quiet)
        _install_verified(staging, output, descriptor)
    _say("Official weight bundle installed and verified.", quiet)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and SHA-256 verify the official v0.2.0 inference weight bundle.")
    parser.add_argument("--output-dir", default="weights", help="Destination weights directory; matching files are reused.")
    parser.add_argument("--archive", help="Verify and install a local copy of the official ZIP instead of downloading.")
    parser.add_argument("--quiet", action="store_true", help="Suppress download progress and success messages.")
    args = parser.parse_args(argv)
    try:
        download_weights(args.output_dir, archive=args.archive, quiet=args.quiet)
    except DownloadError as error:
        parser.exit(2, f"Error: {error}\n")
    except OSError:
        parser.exit(2, "Error: Cannot access the archive or destination. Check permissions and available disk space.\n")
    except KeyboardInterrupt:
        parser.exit(130, "Weight download interrupted.\n")
    if not args.quiet:
        print("Official v0.2.0 weights are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
