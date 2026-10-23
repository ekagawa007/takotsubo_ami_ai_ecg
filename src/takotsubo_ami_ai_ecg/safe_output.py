"""Conservative local result saving; no model dependencies.

New files are reserved exclusively. Explicit replacements use completed temporary
files. CSV and its metadata are two separate files, not a two-file transaction.
On failure, uncertain files are retained rather than deleted through pathnames.
Use an output directory controlled by the current user: these checks are not a
sandbox against another process with arbitrary access to the same directory.
"""

import os
from pathlib import Path
import secrets
import stat


class OutputError(ValueError):
    """Expected output error with no private path in its message."""


_PROTECTED = "Prediction output must not overwrite an input image, input CSV or model file."
_EXISTS = "Prediction output already exists; choose a new name or use --overwrite."
_INCOMPLETE = ("Prediction output could not be completed. Existing files were not truncated; "
               "some result or temporary files may remain. Review the output directory "
               "and choose a new output name before retrying.")


def absolute_path(path):
    """Normalize without resolving links that must remain visible to checks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _identity(info):
    return info.st_dev, info.st_ino


def _is_link(info):
    return stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", None) in (0xA0000003, 0xA000000C)


def _lstat(path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _check_name(path):
    # Colons on Windows can name alternate data streams rather than new files.
    if os.name == "nt":
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}
        for part in path.parts[1:]:
            if (":" in part or part.endswith((".", " "))
                    or part.split(".")[0].upper() in reserved):
                raise OutputError("Unsupported output filename or directory name.")


def _check_parents(path):
    records = []
    for directory in (*reversed(path.parents), path):
        info = _lstat(directory)
        if info is None:
            break
        if _is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise OutputError("Output directories must be ordinary directories, not links or junctions.")
        records.append((directory, _identity(info)))
    return records


def _protected(paths):
    names, identities = set(), set()
    for value in paths:
        path = absolute_path(value).resolve()
        names.add(path)
        try:
            identities.add(_identity(path.stat()))
        except FileNotFoundError:
            pass
    return names, identities


def _check_destination(path, info, protected, overwrite):
    names, identities = protected
    if path in names or (info is not None and _identity(info) in identities):
        raise OutputError(_PROTECTED)
    if info is not None:
        if _is_link(info) or not stat.S_ISREG(info.st_mode):
            raise OutputError("Prediction outputs must be ordinary files, not links or special files.")
        if not overwrite:
            raise OutputError(_EXISTS)


def validate_outputs(paths, overwrite=False, protected_paths=()):
    """Early preflight; write_files repeats checks at the actual save."""
    protected = _protected(protected_paths)
    for value in paths:
        path = absolute_path(value)
        _check_name(path)
        _check_parents(path.parent)
        _check_destination(path, _lstat(path), protected, overwrite)


class _Directory:
    """Anchor leaf operations to a checked parent during a short save."""

    def __init__(self, path):
        self.path = path
        self.fd = None
        self.handles = []
        self.records = []

    def __enter__(self):
        try:
            if os.name == "nt":
                self._open_windows()
            else:
                self._open_posix()
            self.records = _check_parents(self.path)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _open_windows(self):
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        self._close_handle = kernel.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL
        # Hold every parent without FILE_SHARE_DELETE, preventing renames or
        # link/junction replacement while the following path-based calls run.
        for directory in (*reversed(self.path.parents), self.path):
            info = _lstat(directory)
            if info is None:
                try:
                    directory.mkdir()
                except FileExistsError:
                    pass
            _check_parents(directory)
            handle = create(str(directory), 0x0001 | 0x0080, 0x0001 | 0x0002,
                            None, 3, 0x02000000 | 0x00200000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise OutputError("Cannot hold the output directory safely; choose a private local directory.")
            self.handles.append(handle)
            _check_parents(directory)

    def _open_posix(self):
        if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            raise OutputError("This platform cannot safely anchor output directories.")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.path.anchor, flags)
        try:
            for part in self.path.parts[1:]:
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            self.fd = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def verify(self):
        if _check_parents(self.path) != self.records:
            raise OutputError("The output directory changed during saving.")

    def info(self, name):
        try:
            if self.fd is not None:
                return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            return (self.path / name).lstat()
        except FileNotFoundError:
            return None

    def create(self, name):
        self.verify()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if self.fd is not None:
            descriptor = os.open(name, flags, 0o600, dir_fd=self.fd)
        else:
            descriptor = os.open(self.path / name, flags, 0o600)
        return os.fdopen(descriptor, "wb")

    def replace(self, source, destination):
        self.verify()
        if self.fd is not None:
            os.replace(source, destination, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        else:
            os.replace(self.path / source, self.path / destination)

    def __exit__(self, *unused):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        for handle in reversed(self.handles):
            self._close_handle(handle)
        self.handles.clear()


def _write(stream, payload):
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())


def _require_identity(info, expected):
    if info is None or _is_link(info) or not stat.S_ISREG(info.st_mode) or _identity(info) != expected:
        raise OutputError("An output file changed during saving.")


def write_files(files, overwrite=False, protected_paths=()):
    """Write pre-serialized (path, bytes) pairs without truncating existing files.

    Both CSV and sidecar are checked/reserved before new contents are written.
    If a save fails, report it and retain uncertain artifacts; do not unlink a
    pathname which might now refer to someone else's file.
    """
    files = [(absolute_path(path), payload) for path, payload in files]
    if not files or any(not isinstance(payload, bytes) for _, payload in files):
        raise ValueError("Output contents must be serialized to bytes before saving.")
    if len({path for path, _ in files}) != len(files) or len({path.parent for path, _ in files}) != 1:
        raise ValueError("Prediction output files must have distinct names in one directory.")
    validate_outputs([path for path, _ in files], overwrite, protected_paths)
    protected = _protected(protected_paths)
    streams = []
    touched = False
    try:
        with _Directory(files[0][0].parent) as directory:
            before = {}
            for path, _ in files:
                info = directory.info(path.name)
                _check_destination(path, info, protected, overwrite)
                before[path.name] = None if info is None else _identity(info)
            if not overwrite:
                # O_EXCL, rather than an earlier exists() call, decides ownership.
                for path, payload in files:
                    stream = directory.create(path.name)
                    touched = True
                    streams.append((path.name, stream, payload, _identity(os.fstat(stream.fileno()))))
                for name, stream, payload, identity in streams:
                    directory.verify()
                    _require_identity(directory.info(name), identity)
                    _write(stream, payload)
                for name, stream, _, identity in streams:
                    directory.verify()
                    _require_identity(directory.info(name), identity)
            else:
                prepared = []
                for path, payload in files:
                    name = f".{path.name}.ecg-{secrets.token_hex(16)}.tmp"
                    with directory.create(name) as stream:
                        touched = True
                        _write(stream, payload)
                        prepared.append((path, name, _identity(os.fstat(stream.fileno()))))
                # All contents are complete before replacing either file.
                for path, name, identity in prepared:
                    current = directory.info(path.name)
                    _check_destination(path, current, protected, True)
                    if (None if current is None else _identity(current)) != before[path.name]:
                        raise OutputError("A prediction output changed before replacement.")
                    _require_identity(directory.info(name), identity)
                    directory.replace(name, path.name)
                    _require_identity(directory.info(path.name), identity)
                directory.verify()
    except (OSError, OutputError) as error:
        if touched:
            raise OutputError(_INCOMPLETE) from None
        if isinstance(error, FileExistsError):
            raise OutputError(_EXISTS) from None
        if isinstance(error, OutputError):
            raise
        raise OutputError("Cannot safely access the prediction output directory.") from None
    finally:
        for _, stream, _, _ in streams:
            stream.close()
