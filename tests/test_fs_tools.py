"""Adversarial tests for the safe workspace read tier."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from inverse_agent.fs_tools import (
    FILE_MAX_BYTES,
    READ_MAX_LINES,
    FsToolError,
    PolicyViolationError,
    WorkspaceReader,
)
from inverse_agent.secure_fs import SecureFsPolicyError


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("line one\nline two\nline three\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Title\nbody text\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / ".env").write_text("api_key=sk_live_abcdefghijklmnop\n", encoding="utf-8")
    return tmp_path


def test_read_file_returns_numbered_lines(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    obs = reader.read_file("src/app.py")
    assert obs.path == "src/app.py"
    assert obs.lines[0] == "1: line one"
    assert obs.content_hash
    assert not obs.incomplete


def test_read_file_rejects_absolute_path(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="absolute"):
        reader.read_file(str(workspace / "src" / "app.py"))


def test_read_file_rejects_traversal(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="traversal"):
        reader.read_file("../secret.txt")


def test_secure_backend_revalidates_relative_components(workspace: Path) -> None:
    secure = WorkspaceReader.open(workspace).secure
    with pytest.raises(SecureFsPolicyError):
        secure.read_bytes(
            ("..", "secret.txt"),
            maximum_bytes=1024,
            deadline=float("inf"),
        )
    with pytest.raises(SecureFsPolicyError):
        secure.list_directory(("src/app.py/child",), maximum_visits=10, deadline=float("inf"))


def test_read_file_rejects_git_internals(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="denied directory"):
        reader.read_file(".git/config")


def test_read_file_rejects_sensitive_env(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="sensitive-file policy"):
        reader.read_file(".env")


def test_read_file_rejects_ads_syntax(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="alternate-data-stream"):
        reader.read_file("src/app.py:secret")


def test_read_file_rejects_device_name(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="reserved device name"):
        reader.read_file("NUL")


def test_read_file_rejects_trailing_dot_alias(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="trailing dot or space"):
        reader.read_file("src/app.py.")


def test_read_file_redacts_secrets_line_preserving(tmp_path: Path) -> None:
    (tmp_path / "conf.py").write_text(
        "safe = 1\napi_key=sk_live_0123456789abcdef\nsafe = 2\n", encoding="utf-8"
    )
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.read_file("conf.py")
    assert obs.incomplete
    assert obs.redacted
    assert "sk_live_0123456789abcdef" not in obs.text
    # Line count preserved: line 3 stays line 3.
    assert obs.lines[2] == "3: safe = 2"


def test_read_file_refuses_non_utf8(tmp_path: Path) -> None:
    (tmp_path / "blob.txt").write_bytes(b"valid\n\xff\xfe not utf8\n")
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError, match="not valid UTF-8"):
        reader.read_file("blob.txt")


def test_read_file_flags_binary(tmp_path: Path) -> None:
    (tmp_path / "image.bin").write_bytes(bytes(range(256)) * 8)
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.read_file("image.bin")
    assert obs.metadata.get("binary") is True
    assert obs.text == ""


def test_read_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("secret\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted in this environment")
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError, match="symlink, junction, or reparse"):
        reader.read_file("link.txt")


def test_read_file_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "file.txt").write_text("data\n", encoding="utf-8")
    link_dir = tmp_path / "linkdir"
    try:
        os.symlink(real, link_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted in this environment")
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError, match="symlink, junction, or reparse"):
        reader.read_file("linkdir/file.txt")


def test_read_file_rejects_hard_link(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside secret\n", encoding="utf-8")
    linked = tmp_path / "linked.txt"
    try:
        os.link(outside, linked)
    except OSError:
        pytest.skip("hard links not permitted in this environment")
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError, match="multiple hard links"):
        reader.read_file("linked.txt")


def test_read_file_uses_validated_handle_not_path_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "safe.txt"
    target.write_text("retained handle\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)

    def path_reopen_forbidden(_path: Path) -> bytes:
        raise AssertionError("validated path was reopened")

    monkeypatch.setattr(Path, "read_bytes", path_reopen_forbidden)
    assert reader.read_file("safe.txt").text == "retained handle\n"


def test_reader_rejects_replaced_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("old root\n", encoding="utf-8")
    reader = WorkspaceReader.open(workspace)
    moved = tmp_path / "moved-workspace"
    workspace.rename(moved)
    workspace.mkdir()
    (workspace / "safe.txt").write_text("replacement root\n", encoding="utf-8")
    with pytest.raises(FsToolError, match="workspace root was replaced"):
        reader.read_file("safe.txt")


def test_read_file_enforces_operation_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inverse_agent.fs_tools as fs_tools

    (tmp_path / "safe.txt").write_text("content\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    monkeypatch.setattr(fs_tools, "FS_OPERATION_TIMEOUT_SECONDS", -1.0)
    with pytest.raises(FsToolError, match="deadline"):
        reader.read_file("safe.txt")


def test_recursive_list_and_search_propagate_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inverse_agent.fs_tools as fs_tools

    (tmp_path / "safe.py").write_text("needle\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    monkeypatch.setattr(fs_tools, "FS_OPERATION_TIMEOUT_SECONDS", -1.0)
    with pytest.raises(FsToolError, match="deadline"):
        reader.list_files(".", glob="**/*.py")
    with pytest.raises(FsToolError, match="deadline"):
        reader.search_text("needle")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX special-file behavior")
def test_posix_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError, match="regular file"):
        reader.read_file("pipe")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission behavior")
def test_posix_unreadable_regular_file_makes_listing_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "unreadable.txt"
    target.write_text("hidden evidence\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    target.chmod(0)
    try:
        listing = reader.list_files(".")
        assert listing.incomplete and listing.truncated
        assert "unreadable.txt" not in listing.lines
        search = reader.search_text("hidden evidence")
        assert search.incomplete and search.truncated
        assert search.lines == ()
    finally:
        target.chmod(0o600)


def test_read_file_max_lines_bounds(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="max_lines"):
        reader.read_file("src/app.py", max_lines=READ_MAX_LINES + 1)


def test_list_files_excludes_denied_dirs(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    obs = reader.list_files(".")
    assert "src/" in obs.lines
    assert "README.md" in obs.lines
    assert ".git/" not in obs.lines


def test_search_text_finds_and_skips_sensitive(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    obs = reader.search_text("line two")
    assert any("src/app.py:2" in match for match in obs.lines)
    # The .env file must never be scanned.
    assert all(".env" not in match for match in obs.lines)


def test_search_text_query_bounds(workspace: Path) -> None:
    reader = WorkspaceReader.open(workspace)
    with pytest.raises(FsToolError, match="query is empty"):
        reader.search_text("")


def test_reader_open_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FsToolError, match="existing directory"):
        WorkspaceReader.open(tmp_path / "does-not-exist")


def test_missing_file_is_retryable_not_a_policy_violation(tmp_path: Path) -> None:
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError) as error:
        reader.read_file("missing.txt")
    assert not isinstance(error.value, PolicyViolationError)


def test_query_and_glob_reject_surrogate_text(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("content\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError, match="non-UTF-8"):
        reader.search_text("\udcff")
    with pytest.raises(FsToolError, match="non-UTF-8"):
        reader.list_files(".", glob="*\udcff")


def test_read_file_redacts_multiline_private_key(tmp_path: Path) -> None:
    # Regression: per-line redaction leaked the key body; span redaction must
    # remove the whole block while preserving line numbers.
    body1 = "MIIEvQIBADANBgkqhkiGSECRETLINE1AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    body2 = "9w0BAQEFAASCBKcwggSECRETLINE2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=="
    pem = (
        "config = 1\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        f"{body1}\n"
        f"{body2}\n"
        "-----END RSA PRIVATE KEY-----\n"
        "after = 2\n"
    )
    (tmp_path / "notes.txt").write_text(pem, encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.read_file("notes.txt")
    assert obs.redacted and obs.incomplete
    assert body1 not in obs.text
    assert body2 not in obs.text
    assert "BEGIN RSA PRIVATE KEY" not in obs.text
    # Line numbers preserved: "after = 2" stays on line 6.
    assert "6: after = 2" in obs.lines


def test_redacted_content_hash_is_not_a_raw_secret_oracle(tmp_path: Path) -> None:
    first = "before\n-----BEGIN PRIVATE KEY-----\nSECRET_A\n-----END PRIVATE KEY-----\nafter\n"
    second = first.replace("SECRET_A", "SECRET_B")
    (tmp_path / "first.txt").write_text(first, encoding="utf-8")
    (tmp_path / "second.txt").write_text(second, encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    first_obs = reader.read_file("first.txt")
    second_obs = reader.read_file("second.txt")
    assert first_obs.redacted and second_obs.redacted
    assert first_obs.text == second_obs.text
    assert first_obs.content_hash == second_obs.content_hash


def test_search_skips_oversized_file_and_marks_result_incomplete(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "needle\n" + "x" * FILE_MAX_BYTES,
        encoding="utf-8",
    )
    (tmp_path / "small.txt").write_text("other\n", encoding="utf-8")
    observation = WorkspaceReader.open(tmp_path).search_text("needle")
    assert observation.lines == ()
    assert observation.incomplete and observation.truncated
    assert observation.metadata["oversized_skipped"] == 1


def test_read_file_past_eof_yields_no_citable_lines(tmp_path: Path) -> None:
    # Regression: a start past EOF must not manufacture a phantom line.
    (tmp_path / "small.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.read_file("small.py", start_line=9999)
    assert obs.lines == ()
    assert obs.text == ""


def test_read_file_window_ids_are_unique_per_window(tmp_path: Path) -> None:
    content = "\n".join(f"line {i}" for i in range(100)) + "\n"
    (tmp_path / "f.py").write_text(content, encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    a = reader.read_file("f.py", start_line=1, max_lines=10)
    b = reader.read_file("f.py", start_line=40, max_lines=10)
    assert a.observation_id != b.observation_id


def test_read_file_byte_ceiling_on_wide_chars(tmp_path: Path) -> None:
    # Multi-byte characters must be bounded by encoded bytes, not char count.
    wide = ("界" * 200 + "\n") * 200
    (tmp_path / "cjk.txt").write_text(wide, encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.read_file("cjk.txt")
    assert len(obs.text.encode("utf-8")) <= 3_000 * 4
    assert obs.truncated


def test_read_file_rejects_denied_dir_case_insensitively(tmp_path: Path) -> None:
    reader = WorkspaceReader.open(tmp_path)
    with pytest.raises(FsToolError, match="denied directory"):
        reader.read_file(".GIT/config")


def test_search_text_response_ceiling(tmp_path: Path) -> None:
    # Many matching lines must not exceed the serialized ceiling.
    lines = "\n".join(f"needle line {i} " + "x" * 400 for i in range(200))
    (tmp_path / "big.txt").write_text(lines + "\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.search_text("needle")
    assert obs.truncated
    assert len(obs.text.encode()) <= 16 * 1024


def test_list_files_response_ceiling(tmp_path: Path) -> None:
    for i in range(400):
        (tmp_path / (f"file_{i:03d}_" + "y" * 60 + ".txt")).write_text("x", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".")
    assert len(obs.text.encode()) <= 16 * 1024


def test_search_skips_denied_dir_case_insensitively(tmp_path: Path) -> None:
    # A physically upper-cased denied dir must not be recursively searched.
    denied = tmp_path / ".SSH"
    denied.mkdir()
    (denied / "notes.txt").write_text("secret cluster token here\n", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("cluster is fine here\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.search_text("cluster")
    assert any("ok.txt" in line for line in obs.lines)
    assert all(".SSH" not in line and ".ssh" not in line for line in obs.lines)


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short-name aliasing is windows-only")
def test_read_file_rejects_short_name_alias(tmp_path: Path) -> None:
    secret = tmp_path / "google-services.json"
    secret.write_text('{"k":"secret-value"}\n', encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    # The 8.3 alias of google-services.json is typically GOOGLE~1.JSO.
    import subprocess

    listing = subprocess.run(
        ["cmd", "/c", "dir", "/x", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    if "GOOGLE~1.JSO" not in listing.upper():
        pytest.skip("8.3 short names not enabled on this volume")
    with pytest.raises(FsToolError):
        reader.read_file("GOOGLE~1.JSO")


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only reparse attribute path")
def test_windows_reparse_attribute_detected(tmp_path: Path) -> None:
    # On Windows a directory junction carries the reparse attribute; if we cannot
    # create one, the portable symlink tests already cover the rejection path.
    reader = WorkspaceReader.open(tmp_path)
    (tmp_path / "plain.txt").write_text("ok\n", encoding="utf-8")
    obs = reader.read_file("plain.txt")
    assert obs.path == "plain.txt"


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are windows-only")
def test_windows_directory_junction_is_never_traversed(tmp_path: Path) -> None:
    import subprocess

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside secret\n", encoding="utf-8")
    junction = tmp_path / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("directory junctions not permitted in this environment")
    try:
        reader = WorkspaceReader.open(tmp_path)
        with pytest.raises(FsToolError, match="symlink, junction, or reparse"):
            reader.read_file("junction/secret.txt")
        assert "outside secret" not in reader.search_text("outside secret").text
    finally:
        junction.rmdir()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_read_handle_denies_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import inverse_agent.secure_fs as secure_fs

    target = tmp_path / "stable.txt"
    target.write_bytes(b"A" * (128 * 1024))
    reader = WorkspaceReader.open(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    original_read = secure_fs._windows_read

    def delayed_read(handle: int, *, maximum_bytes: int, deadline: float) -> bytes:
        entered.set()
        if not release.wait(5):
            raise AssertionError("writer probe did not release the read")
        return original_read(handle, maximum_bytes=maximum_bytes, deadline=deadline)

    def run_read() -> None:
        try:
            reader.read_file("stable.txt")
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(secure_fs, "_windows_read", delayed_read)
    thread = threading.Thread(target=run_read)
    thread.start()
    assert entered.wait(5)
    try:
        with pytest.raises(OSError):
            target.open("r+b")
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert failures == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_writer_cannot_hide_file_from_listing_or_search(tmp_path: Path) -> None:
    target = tmp_path / "visible.txt"
    target.write_text("visible evidence\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    with target.open("r+b"):
        assert "visible.txt" in reader.list_files(".").lines
        search = reader.search_text("visible evidence")
        assert search.incomplete and search.truncated
        assert search.lines == ()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_exclusive_writer_makes_listing_and_search_fail_closed(tmp_path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    target = tmp_path / "exclusive.txt"
    target.write_text("exclusive evidence\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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
    handle = create_file(str(target), 0xC0000000, 0, None, 3, 0, None)
    assert handle not in {None, ctypes.c_void_p(-1).value}
    try:
        listing = reader.list_files(".")
        assert listing.incomplete and listing.truncated
        assert "exclusive.txt" not in listing.lines
        search = reader.search_text("exclusive evidence")
        assert search.incomplete and search.truncated
        assert search.lines == ()
    finally:
        kernel32.CloseHandle(handle)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_unverifiable_denied_name_is_not_disclosed(tmp_path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    target = tmp_path / ".env"
    target.write_text("SECRET=value\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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
    handle = create_file(str(target), 0xC0000000, 0, None, 3, 0, None)
    assert handle not in {None, ctypes.c_void_p(-1).value}
    try:
        listing = reader.list_files(".")
        assert listing.incomplete and listing.truncated
        assert all(".env" not in line for line in listing.lines)
    finally:
        kernel32.CloseHandle(handle)
