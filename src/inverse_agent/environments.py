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


def discover_gradle_wrapper(root: Path) -> Path | None:
    root = root.resolve()
    names = ("gradlew.bat", "gradlew") if sys.platform == "win32" else ("gradlew",)
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    return None
