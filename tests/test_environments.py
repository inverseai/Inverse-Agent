from __future__ import annotations

from pathlib import Path

import pytest

import inverse_agent.environments as environments


def test_trusted_git_ignores_programfiles_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted-program-files"
    attacker = tmp_path / "attacker-controlled"
    trusted_git = trusted / "Git" / "cmd" / "git.exe"
    attacker_git = attacker / "Git" / "cmd" / "git.exe"
    trusted_git.parent.mkdir(parents=True)
    attacker_git.parent.mkdir(parents=True)
    trusted_git.write_bytes(b"trusted")
    attacker_git.write_bytes(b"attacker")
    monkeypatch.setattr(environments.sys, "platform", "win32")
    monkeypatch.setattr(environments, "_windows_program_files_roots", lambda: (trusted,))
    monkeypatch.setenv("PROGRAMFILES", str(attacker))

    discovered = environments.discover_trusted_git()

    assert discovered == trusted_git.resolve()
    assert discovered != attacker_git.resolve()


def test_trusted_git_rejects_linked_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted-program-files"
    outside = tmp_path / "outside"
    (outside / "cmd").mkdir(parents=True)
    (outside / "cmd" / "git.exe").write_bytes(b"untrusted")
    trusted.mkdir()
    try:
        (trusted / "Git").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    monkeypatch.setattr(environments.sys, "platform", "win32")
    monkeypatch.setattr(environments, "_windows_program_files_roots", lambda: (trusted,))

    assert environments.discover_trusted_git() is None
