"""Target-workspace toolchain discovery."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Executable:
    path: Path
    source: str


def discover_python(root: Path) -> Executable:
    root = root.resolve()
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        root / "venv" / "Scripts" / "python.exe",
        root / "venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return Executable(candidate.resolve(), "workspace-virtualenv")
    return Executable(Path(sys.executable).resolve(), "inverse-agent-runtime")


def discover_system_executable(name: str) -> Path | None:
    suffixes = (".exe", ".cmd", ".bat", "") if sys.platform == "win32" else ("",)
    runtime_bin = Path(sys.executable).resolve().parent
    for suffix in suffixes:
        candidate = runtime_bin / f"{name}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    value = shutil.which(name)
    return Path(value).resolve() if value else None


def discover_trusted_git() -> Path | None:
    """Find Git only in fixed operating-system installation locations."""

    if sys.platform == "win32":
        candidates = tuple(
            root / "Git" / location / "git.exe"
            for root in _windows_program_files_roots()
            for location in ("cmd", "bin")
        )
    else:
        candidates = (
            Path("/usr/bin/git"),
            Path("/usr/local/bin/git"),
            Path("/opt/homebrew/bin/git"),
            Path("/opt/local/bin/git"),
        )
    for candidate in candidates:
        if _is_unlinked_file(candidate):
            try:
                return candidate.resolve(strict=True)
            except OSError:
                continue
    return None


def _windows_program_files_roots() -> tuple[Path, ...]:
    """Read Program Files locations from the machine registry, never process environment."""

    if sys.platform != "win32":
        return ()
    try:
        import winreg

        access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion",
            access=access,
        ) as key:
            values: list[object] = []
            for name in ("ProgramFilesDir", "ProgramFilesDir (x86)"):
                try:
                    values.append(winreg.QueryValueEx(key, name)[0])
                except OSError:
                    continue
    except OSError:
        return ()
    roots: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        root = Path(value)
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _is_unlinked_file(path: Path) -> bool:
    """Reject a candidate when it or any existing parent is a link or junction."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink() or bool(getattr(current, "is_junction", lambda: False)()):
            return False
    return path.is_file()


def discover_gradle_wrapper(root: Path) -> Path | None:
    root = root.resolve()
    names = ("gradlew.bat", "gradlew") if sys.platform == "win32" else ("gradlew",)
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    return None
