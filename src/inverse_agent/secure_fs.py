"""Race-resistant, read-only filesystem primitives for untrusted workspaces.

The high-level read tools validate policy. This module makes that policy true at
the OS boundary: every child is opened relative to a retained parent handle,
links/reparse points are never followed, and file bytes are read from the same
handle whose metadata was validated.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import stat
import time
from collections.abc import Iterator
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


class SecureFsError(OSError):
    """A filesystem object could not be opened or read safely."""


class SecureFsPolicyError(SecureFsError):
    """A link, reparse point, alias, or identity change was refused."""


class SecureFsDeadlineError(SecureFsError):
    """The bounded filesystem operation exceeded its deadline."""


@dataclass(frozen=True)
class SecureEntry:
    name: str
    is_dir: bool
    is_file: bool
    size: int
    link_count: int
    identity: tuple[int, int]
    change_token: tuple[int, int]


@dataclass(frozen=True)
class SecureRead:
    data: bytes
    entry: SecureEntry


@dataclass(frozen=True)
class SecureListing:
    entries: tuple[SecureEntry, ...]
    visited: int
    truncated: bool


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise SecureFsDeadlineError("filesystem operation exceeded its deadline")


def _same_entry(left: SecureEntry, right: SecureEntry) -> bool:
    return (
        left.identity == right.identity
        and left.size == right.size
        and left.link_count == right.link_count
        and left.is_dir == right.is_dir
        and left.is_file == right.is_file
        and left.change_token == right.change_token
    )


class SecureWorkspace:
    """Workspace root identity plus handle-relative read/list operations."""

    def __init__(self, workspace: Path, root_identity: tuple[int, int]) -> None:
        self.workspace = workspace
        self.root_identity = root_identity

    @classmethod
    def open(cls, workspace: Path) -> SecureWorkspace:
        try:
            resolved = workspace.resolve(strict=True)
        except OSError as exc:
            raise SecureFsError("workspace is not an existing directory") from exc
        if not resolved.is_dir():
            raise SecureFsError("workspace is not an existing directory")
        if os.name == "nt":
            with _windows_root(resolved) as handle:
                entry = _windows_entry(handle, resolved.name)
        else:
            with _posix_root(resolved) as descriptor:
                entry = _posix_entry(descriptor, resolved.name)
        return cls(resolved, entry.identity)

    def read_bytes(
        self,
        parts: tuple[str, ...],
        *,
        maximum_bytes: int,
        deadline: float,
    ) -> SecureRead:
        if not parts:
            raise SecureFsError("file path is empty")
        if os.name == "nt":
            return self._read_windows(parts, maximum_bytes=maximum_bytes, deadline=deadline)
        return self._read_posix(parts, maximum_bytes=maximum_bytes, deadline=deadline)

    def list_directory(
        self,
        parts: tuple[str, ...],
        *,
        maximum_visits: int,
        deadline: float,
    ) -> SecureListing:
        if maximum_visits < 1:
            raise SecureFsError("directory visit limit must be positive")
        if os.name == "nt":
            return self._list_windows(parts, maximum_visits=maximum_visits, deadline=deadline)
        return self._list_posix(parts, maximum_visits=maximum_visits, deadline=deadline)

    def _read_posix(
        self,
        parts: tuple[str, ...],
        *,
        maximum_bytes: int,
        deadline: float,
    ) -> SecureRead:
        with self._posix_target(parts, expect_directory=False, deadline=deadline) as descriptor:
            before = _posix_entry(descriptor, parts[-1])
            if not before.is_file:
                raise SecureFsError("path is not a regular file")
            if before.link_count > 1:
                raise SecureFsPolicyError("file has multiple hard links")
            if before.size > maximum_bytes:
                raise SecureFsError("file exceeds the maximum readable size")
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                _check_deadline(deadline)
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > maximum_bytes:
                raise SecureFsError("file exceeds the maximum readable size")
            after = _posix_entry(descriptor, parts[-1])
            if not _same_entry(before, after):
                raise SecureFsError("file changed while being read")
            return SecureRead(data=data, entry=after)

    def _list_posix(
        self,
        parts: tuple[str, ...],
        *,
        maximum_visits: int,
        deadline: float,
    ) -> SecureListing:
        with self._posix_target(parts, expect_directory=True, deadline=deadline) as descriptor:
            result: list[SecureEntry] = []
            visited = 0
            truncated = False
            try:
                with os.scandir(descriptor) as scan:
                    for scanned in scan:
                        _check_deadline(deadline)
                        visited += 1
                        if visited > maximum_visits:
                            visited = maximum_visits
                            truncated = True
                            break
                        name = scanned.name
                        if name in {".", ".."}:
                            continue
                        try:
                            flags = (
                                os.O_RDONLY
                                | getattr(os, "O_CLOEXEC", 0)
                                | getattr(os, "O_NOFOLLOW", 0)
                                | getattr(os, "O_NONBLOCK", 0)
                            )
                            child = os.open(name, flags, dir_fd=descriptor)
                        except OSError as exc:
                            try:
                                refused = os.stat(
                                    name,
                                    dir_fd=descriptor,
                                    follow_symlinks=False,
                                )
                            except OSError as stat_exc:
                                raise SecureFsError(
                                    "directory entry could not be opened and verified"
                                ) from stat_exc
                            if stat.S_ISLNK(refused.st_mode) or not (
                                stat.S_ISDIR(refused.st_mode) or stat.S_ISREG(refused.st_mode)
                            ):
                                continue
                            raise SecureFsError(
                                "directory entry could not be opened and verified"
                            ) from exc
                        try:
                            entry = _posix_entry(child, name)
                            if not (entry.is_dir or entry.is_file):
                                continue
                            result.append(entry)
                        finally:
                            os.close(child)
            except SecureFsError:
                raise
            except OSError as exc:
                raise SecureFsError("directory could not be listed") from exc
            return SecureListing(tuple(result), visited, truncated)

    @contextlib.contextmanager
    def _posix_target(
        self,
        parts: tuple[str, ...],
        *,
        expect_directory: bool,
        deadline: float,
    ) -> Iterator[int]:
        with _posix_root(self.workspace) as root:
            root_entry = _posix_entry(root, self.workspace.name)
            if root_entry.identity != self.root_identity:
                raise SecureFsPolicyError("workspace root was replaced")
            current = root
            owned: list[int] = []
            try:
                for index, part in enumerate(parts):
                    _check_deadline(deadline)
                    final = index == len(parts) - 1
                    wants_directory = not final or expect_directory
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0)
                    )
                    if wants_directory:
                        flags |= getattr(os, "O_DIRECTORY", 0)
                    try:
                        child = os.open(part, flags, dir_fd=current)
                    except OSError as exc:
                        raise SecureFsPolicyError(
                            "path component is unavailable or is a link"
                        ) from exc
                    owned.append(child)
                    entry = _posix_entry(child, part)
                    if wants_directory and not entry.is_dir:
                        raise SecureFsPolicyError("path traverses a non-directory")
                    current = child
                yield current
            finally:
                for descriptor in reversed(owned):
                    os.close(descriptor)

    def _read_windows(
        self,
        parts: tuple[str, ...],
        *,
        maximum_bytes: int,
        deadline: float,
    ) -> SecureRead:
        with self._windows_target(parts, expect_directory=False, deadline=deadline) as handle:
            before = _windows_entry(handle, parts[-1])
            if not before.is_file:
                raise SecureFsError("path is not a regular file")
            if before.link_count > 1:
                raise SecureFsPolicyError("file has multiple hard links")
            if before.size > maximum_bytes:
                raise SecureFsError("file exceeds the maximum readable size")
            data = _windows_read(handle, maximum_bytes=maximum_bytes, deadline=deadline)
            after = _windows_entry(handle, parts[-1])
            if not _same_entry(before, after):
                raise SecureFsError("file changed while being read")
            return SecureRead(data=data, entry=after)

    def _list_windows(
        self,
        parts: tuple[str, ...],
        *,
        maximum_visits: int,
        deadline: float,
    ) -> SecureListing:
        with self._windows_target(parts, expect_directory=True, deadline=deadline) as directory:
            result: list[SecureEntry] = []
            visited = 0
            truncated = False
            for name, attributes in _windows_directory_names(directory, deadline=deadline):
                _check_deadline(deadline)
                visited += 1
                if visited > maximum_visits:
                    visited = maximum_visits
                    truncated = True
                    break
                if name in {".", ".."} or attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                    continue
                expect_directory = bool(attributes & _FILE_ATTRIBUTE_DIRECTORY)
                try:
                    with _windows_child(
                        directory,
                        name,
                        expect_directory=expect_directory,
                    ) as child:
                        entry = _windows_entry(child, name)
                except SecureFsError as exc:
                    raise SecureFsError("directory entry could not be opened and verified") from exc
                if not (entry.is_dir or entry.is_file):
                    continue
                result.append(entry)
            return SecureListing(tuple(result), visited, truncated)

    @contextlib.contextmanager
    def _windows_target(
        self,
        parts: tuple[str, ...],
        *,
        expect_directory: bool,
        deadline: float,
    ) -> Iterator[int]:
        with _windows_root(self.workspace) as root:
            root_entry = _windows_entry(root, self.workspace.name)
            if root_entry.identity != self.root_identity:
                raise SecureFsPolicyError("workspace root was replaced")
            current = root
            owned: list[int] = []
            try:
                for index, part in enumerate(parts):
                    _check_deadline(deadline)
                    final = index == len(parts) - 1
                    wants_directory = not final or expect_directory
                    child = _windows_open_child(
                        current,
                        part,
                        expect_directory=wants_directory,
                        deny_writers=final and not expect_directory,
                    )
                    owned.append(child)
                    entry = _windows_entry(child, part)
                    if wants_directory and not entry.is_dir:
                        raise SecureFsPolicyError("path traverses a non-directory")
                    if not _windows_name_is_canonical(child, part):
                        raise SecureFsPolicyError(
                            "path uses a short-name alias or non-canonical form"
                        )
                    current = child
                yield current
            finally:
                for handle in reversed(owned):
                    _close_windows(handle)


def _posix_entry(descriptor: int, name: str) -> SecureEntry:
    info = os.fstat(descriptor)
    return SecureEntry(
        name=name,
        is_dir=stat.S_ISDIR(info.st_mode),
        is_file=stat.S_ISREG(info.st_mode),
        size=info.st_size,
        link_count=info.st_nlink,
        identity=(info.st_dev, info.st_ino),
        change_token=(info.st_mtime_ns, info.st_ctime_ns),
    )


@contextlib.contextmanager
def _posix_root(workspace: Path) -> Iterator[int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(workspace, flags)
    except OSError as exc:
        raise SecureFsError("workspace root is no longer accessible") from exc
    try:
        yield descriptor
    finally:
        os.close(descriptor)


_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_SHARE_READ = 0x1
_FILE_SHARE_ALL = 0x7
_FILE_OPEN = 1
_FILE_DIRECTORY_FILE = 0x1
_FILE_SYNCHRONOUS_IO_NONALERT = 0x20
_FILE_NON_DIRECTORY_FILE = 0x40
_FILE_OPEN_REPARSE_POINT = 0x00200000
_FILE_READ_ATTRIBUTES = 0x80
_SYNCHRONIZE = 0x00100000
_OBJ_CASE_INSENSITIVE = 0x40
_OBJ_DONT_REPARSE = 0x1000
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_GENERIC_READ = 0x80000000
_ERROR_NO_MORE_FILES = 18
_FILE_ID_BOTH_DIRECTORY_INFO = 10
_FILE_ID_BOTH_DIRECTORY_RESTART_INFO = 11
_VOLUME_NAME_DOS = 0
_INVALID_HANDLE_VALUE = cast(int, ctypes.c_void_p(-1).value)


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", _FILETIME),
        ("ftLastAccessTime", _FILETIME),
        ("ftLastWriteTime", _FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", wintypes.LPWSTR),
    ]


class _OBJECT_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.ULONG),
        ("RootDirectory", wintypes.HANDLE),
        ("ObjectName", ctypes.POINTER(_UNICODE_STRING)),
        ("Attributes", wintypes.ULONG),
        ("SecurityDescriptor", wintypes.LPVOID),
        ("SecurityQualityOfService", wintypes.LPVOID),
    ]


class _IO_STATUS_BLOCK_UNION(ctypes.Union):
    _fields_ = [("Status", wintypes.LONG), ("Pointer", wintypes.LPVOID)]


class _IO_STATUS_BLOCK(ctypes.Structure):
    _anonymous_ = ("result",)
    _fields_ = [("result", _IO_STATUS_BLOCK_UNION), ("Information", ctypes.c_size_t)]


class _FILE_ID_BOTH_DIR_INFO(ctypes.Structure):
    _fields_ = [
        ("NextEntryOffset", wintypes.DWORD),
        ("FileIndex", wintypes.DWORD),
        ("CreationTime", ctypes.c_longlong),
        ("LastAccessTime", ctypes.c_longlong),
        ("LastWriteTime", ctypes.c_longlong),
        ("ChangeTime", ctypes.c_longlong),
        ("EndOfFile", ctypes.c_longlong),
        ("AllocationSize", ctypes.c_longlong),
        ("FileAttributes", wintypes.DWORD),
        ("FileNameLength", wintypes.DWORD),
        ("EaSize", wintypes.DWORD),
        ("ShortNameLength", ctypes.c_byte),
        ("ShortName", wintypes.WCHAR * 12),
        ("FileId", ctypes.c_longlong),
        ("FileName", wintypes.WCHAR * 1),
    ]


def _windows_api() -> tuple[Any, Any]:
    if os.name != "nt":
        raise SecureFsError("Windows filesystem API is unavailable")
    return ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("ntdll")


@contextlib.contextmanager
def _windows_root(workspace: Path) -> Iterator[int]:
    kernel32, _ntdll = _windows_api()
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(workspace),
        _GENERIC_READ,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    value = cast(int, handle) if handle is not None else _INVALID_HANDLE_VALUE
    if value == _INVALID_HANDLE_VALUE:
        raise SecureFsError("workspace root is no longer accessible")
    try:
        entry = _windows_entry(value, workspace.name)
        if not entry.is_dir:
            raise SecureFsPolicyError("workspace root is not a directory")
        yield value
    finally:
        _close_windows(value)


@contextlib.contextmanager
def _windows_child(parent: int, name: str, *, expect_directory: bool) -> Iterator[int]:
    handle = _windows_open_child(
        parent,
        name,
        expect_directory=expect_directory,
        deny_writers=False,
    )
    try:
        yield handle
    finally:
        _close_windows(handle)


def _windows_open_child(
    parent: int,
    name: str,
    *,
    expect_directory: bool,
    deny_writers: bool,
) -> int:
    _kernel32, ntdll = _windows_api()
    name_buffer = ctypes.create_unicode_buffer(name)
    encoded_length = len(name.encode("utf-16-le"))
    unicode_name = _UNICODE_STRING(
        Length=encoded_length,
        MaximumLength=encoded_length + 2,
        Buffer=ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = _OBJECT_ATTRIBUTES(
        Length=ctypes.sizeof(_OBJECT_ATTRIBUTES),
        RootDirectory=wintypes.HANDLE(parent),
        ObjectName=ctypes.pointer(unicode_name),
        Attributes=_OBJ_CASE_INSENSITIVE | _OBJ_DONT_REPARSE,
        SecurityDescriptor=None,
        SecurityQualityOfService=None,
    )
    io_status = _IO_STATUS_BLOCK()
    handle = wintypes.HANDLE()
    nt_create = ntdll.NtCreateFile
    nt_create.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_OBJECT_ATTRIBUTES),
        ctypes.POINTER(_IO_STATUS_BLOCK),
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
    ]
    nt_create.restype = wintypes.LONG
    options = _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
    options |= _FILE_DIRECTORY_FILE if expect_directory else _FILE_NON_DIRECTORY_FILE
    status = int(
        nt_create(
            ctypes.byref(handle),
            _GENERIC_READ | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
            ctypes.byref(attributes),
            ctypes.byref(io_status),
            None,
            0,
            _FILE_SHARE_READ if deny_writers else _FILE_SHARE_ALL,
            _FILE_OPEN,
            options,
            None,
            0,
        )
    )
    if status < 0:
        raise SecureFsPolicyError(
            "path component is unavailable or is a symlink, junction, or reparse point "
            f"(NTSTATUS 0x{status & 0xFFFFFFFF:08X})"
        )
    value = handle.value if handle.value is not None else _INVALID_HANDLE_VALUE
    if value == _INVALID_HANDLE_VALUE:
        raise SecureFsError("Windows returned an invalid file handle")
    try:
        entry = _windows_entry(value, name)
        if entry.is_dir != expect_directory or entry.is_file == expect_directory:
            raise SecureFsPolicyError("path component type changed while opening")
    except BaseException:
        _close_windows(value)
        raise
    return value


def _windows_entry(handle: int, name: str) -> SecureEntry:
    kernel32, _ntdll = _windows_api()
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION)]
    get_info.restype = wintypes.BOOL
    info = _BY_HANDLE_FILE_INFORMATION()
    if not get_info(wintypes.HANDLE(handle), ctypes.byref(info)):
        raise SecureFsError("file metadata could not be read")
    if info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise SecureFsPolicyError("path component is a symlink, junction, or reparse point")
    identity = (
        int(info.dwVolumeSerialNumber),
        (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow),
    )
    size = (int(info.nFileSizeHigh) << 32) | int(info.nFileSizeLow)
    last_write = (int(info.ftLastWriteTime.dwHighDateTime) << 32) | int(
        info.ftLastWriteTime.dwLowDateTime
    )
    is_dir = bool(info.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY)
    return SecureEntry(
        name=name,
        is_dir=is_dir,
        is_file=not is_dir,
        size=size,
        link_count=int(info.nNumberOfLinks),
        identity=identity,
        change_token=(last_write, 0),
    )


def _windows_read(handle: int, *, maximum_bytes: int, deadline: float) -> bytes:
    kernel32, _ntdll = _windows_api()
    read_file = kernel32.ReadFile
    read_file.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    read_file.restype = wintypes.BOOL
    chunks: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining:
        _check_deadline(deadline)
        size = min(64 * 1024, remaining)
        buffer = ctypes.create_string_buffer(size)
        count = wintypes.DWORD()
        if not read_file(handle, buffer, size, ctypes.byref(count), None):
            raise SecureFsError("file bytes could not be read")
        if count.value == 0:
            break
        chunks.append(buffer.raw[: count.value])
        remaining -= count.value
    data = b"".join(chunks)
    if len(data) > maximum_bytes:
        raise SecureFsError("file exceeds the maximum readable size")
    return data


def _windows_directory_names(handle: int, *, deadline: float) -> Iterator[tuple[str, int]]:
    kernel32, _ntdll = _windows_api()
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    restart = True
    while True:
        _check_deadline(deadline)
        buffer = ctypes.create_string_buffer(64 * 1024)
        info_class = (
            _FILE_ID_BOTH_DIRECTORY_RESTART_INFO if restart else _FILE_ID_BOTH_DIRECTORY_INFO
        )
        restart = False
        if not get_info(handle, info_class, buffer, len(buffer)):
            error = ctypes.get_last_error()
            if error == _ERROR_NO_MORE_FILES:
                return
            raise SecureFsError(f"directory could not be listed (WinError {error})")
        offset = 0
        while True:
            header = _FILE_ID_BOTH_DIR_INFO.from_buffer_copy(
                buffer.raw[offset : offset + ctypes.sizeof(_FILE_ID_BOTH_DIR_INFO)]
            )
            start = offset + _FILE_ID_BOTH_DIR_INFO.FileName.offset
            end = start + int(header.FileNameLength)
            name = buffer.raw[start:end].decode("utf-16-le", errors="strict")
            yield name, int(header.FileAttributes)
            if header.NextEntryOffset == 0:
                break
            offset += int(header.NextEntryOffset)


def _windows_name_is_canonical(handle: int, requested: str) -> bool:
    kernel32, _ntdll = _windows_api()
    get_name = kernel32.GetFinalPathNameByHandleW
    get_name.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_name.restype = wintypes.DWORD
    size = get_name(handle, None, 0, _VOLUME_NAME_DOS)
    if size == 0:
        raise SecureFsError("canonical file name could not be read")
    buffer = ctypes.create_unicode_buffer(size + 1)
    written = get_name(handle, buffer, len(buffer), _VOLUME_NAME_DOS)
    if written == 0 or written >= len(buffer):
        raise SecureFsError("canonical file name could not be read")
    canonical_name = Path(buffer.value).name
    return canonical_name.casefold() == requested.casefold()


def _close_windows(handle: int) -> None:
    kernel32, _ntdll = _windows_api()
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(handle))
