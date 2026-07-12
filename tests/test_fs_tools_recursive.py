"""Recursive-glob listing and ``**`` glob matching in the read tier."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.fs_tools import WorkspaceReader, _glob_match


def test_glob_strips_recursive_prefix() -> None:
    assert _glob_match("**/*.py", "app.py")
    assert _glob_match("*.py", "app.py")
    assert not _glob_match("**/*.py", "app.xml")


def test_list_files_recursive_glob_finds_nested(tmp_path: Path) -> None:
    nested = tmp_path / "app" / "src" / "main"
    nested.mkdir(parents=True)
    (nested / "AndroidManifest.xml").write_text("<manifest/>\n", encoding="utf-8")
    (tmp_path / "top.xml").write_text("<x/>\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# hi\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".", glob="**/*.xml")
    assert "app/src/main/AndroidManifest.xml" in obs.lines
    assert "top.xml" in obs.lines
    assert all(".md" not in line for line in obs.lines)
    assert obs.metadata.get("recursive") is True


def test_list_files_recursive_excludes_denied_dirs(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("secret\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("ok\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".", glob="**/*.py")
    assert "keep.py" in obs.lines
    assert all(".git" not in line for line in obs.lines)


def test_list_files_recursive_scoped_to_subdir(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.py").write_text("1\n", encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "y.py").write_text("2\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files("a", glob="**/*.py")
    assert "a/x.py" in obs.lines
    assert all("b/y.py" not in line for line in obs.lines)


def test_recursive_listing_omits_sensitive_file_paths(tmp_path: Path) -> None:
    # A recursive listing must not enumerate the PATHS of sensitive files.
    (tmp_path / ".env").write_text("api_key=sk_live_xxxxxxxxxxxx\n", encoding="utf-8")
    nested = tmp_path / "app"
    nested.mkdir()
    (nested / "id_rsa").write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    (nested / "google-services.json").write_text("{}\n", encoding="utf-8")
    (nested / "main.py").write_text("ok\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".", glob="**/*")
    joined = "\n".join(obs.lines)
    assert "app/main.py" in obs.lines
    assert ".env" not in joined
    assert "id_rsa" not in joined
    assert "google-services.json" not in joined


def test_flat_listing_omits_sensitive_files(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("secret\n", encoding="utf-8")
    (tmp_path / "keystore.jks").write_text("x\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("ok\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".")
    assert "app.py" in obs.lines
    assert ".env" not in obs.lines
    assert "keystore.jks" not in obs.lines


def test_interior_recursive_glob_matches_nested(tmp_path: Path) -> None:
    deep = tmp_path / "src" / "pkg"
    deep.mkdir(parents=True)
    (deep / "mod.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("y\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".", glob="src/**/*.py")
    assert "src/pkg/mod.py" in obs.lines
    assert "other.py" not in obs.lines


def test_recursive_listing_reports_entry_cap_truncation(tmp_path: Path) -> None:
    # More than LIST_MAX_ENTRIES files must report truncated=True, not hide it.
    from inverse_agent.fs_tools import LIST_MAX_ENTRIES

    for i in range(LIST_MAX_ENTRIES + 5):
        (tmp_path / f"f{i:04d}.py").write_text("x\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".", glob="**/*.py")
    assert len(obs.lines) == LIST_MAX_ENTRIES
    assert obs.truncated is True


def test_recursive_listing_path_is_the_directory_not_last_file(tmp_path: Path) -> None:
    # Regression: the observation path must be the listing root, not a matched file.
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.py").write_text("1\n", encoding="utf-8")
    (tmp_path / "a" / "two.py").write_text("2\n", encoding="utf-8")
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files("a", glob="**/*.py")
    assert obs.path == "a"


def test_walk_visit_limit_bounds_a_single_large_directory(
    tmp_path: Path, monkeypatch: object
) -> None:
    # A single directory with many files must not be fully materialized: with a
    # small visit cap, the walk stops early instead of enumerating everything.
    import inverse_agent.fs_tools as fs_tools

    for i in range(50):
        (tmp_path / f"f{i:03d}.py").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(fs_tools, "WALK_VISIT_LIMIT", 10)  # type: ignore[attr-defined]
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.list_files(".", glob="**/*.py")
    # Capped well below the 50 present files.
    assert len(obs.lines) <= 10
    assert obs.truncated is True


def test_search_reports_truncation_when_walk_capped(
    tmp_path: Path, monkeypatch: object
) -> None:
    # search_text must not report a capped scan as complete.
    import inverse_agent.fs_tools as fs_tools

    for i in range(50):
        (tmp_path / f"f{i:03d}.py").write_text("needle here\n", encoding="utf-8")
    monkeypatch.setattr(fs_tools, "WALK_VISIT_LIMIT", 5)  # type: ignore[attr-defined]
    reader = WorkspaceReader.open(tmp_path)
    obs = reader.search_text("needle")
    assert obs.truncated is True
