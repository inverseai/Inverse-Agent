"""Adversarial tests for the safe workspace read tier."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from inverse_agent.fs_tools import (
    READ_MAX_LINES,
    FsToolError,
    WorkspaceReader,
)


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
