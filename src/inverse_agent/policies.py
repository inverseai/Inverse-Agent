"""Policy factories for exact, trusted command execution."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.environments import (
    discover_gradle_wrapper,
    discover_python,
    discover_system_executable,
)
from inverse_agent.models import CommandRule, Domain, RunnerPolicy

GIT_SAFE_PREFIX = (
    "git",
    "--no-optional-locks",
    "-c",
    "core.fsmonitor=",
    "-c",
    "core.pager=cat",
    "-c",
    "pager.status=false",
)
GIT_STATUS_ARGV = (*GIT_SAFE_PREFIX, "status", "--short", "--branch", "--untracked-files=no")
GIT_LS_FILES_ARGV = (*GIT_SAFE_PREFIX, "ls-files")
GIT_HEAD_COMMIT_ARGV = (*GIT_SAFE_PREFIX, "rev-parse", "--verify", "HEAD^{commit}")
GIT_PARENT_COMMIT_ARGV = (*GIT_SAFE_PREFIX, "rev-parse", "--verify", "HEAD^1^{commit}")


def default_policy(workspace_root: Path) -> RunnerPolicy:
    """Create a default-deny policy with exact argv and trusted executable paths."""

    root = workspace_root.resolve()
    python = discover_python(root).path
    gradle = discover_gradle_wrapper(root)
    trusted: dict[str, tuple[Path, ...]] = {}
    workspace_executables: list[Path] = []

    _register_executable(trusted, workspace_executables, root, "python", python)
    if gradle:
        _register_executable(trusted, workspace_executables, root, "gradlew", gradle)
    for name in ("cmake", "git", "pytest", "ruff", "xcodebuild"):
        _register_executable(
            trusted,
            workspace_executables,
            root,
            name,
            discover_system_executable(name),
        )

    rules = [
        CommandRule(
            "git-status",
            GIT_STATUS_ARGV,
            Domain.GENERIC,
            requires_approval=True,
            reason=(
                "Git status executes an operator-selected system binary and may invoke "
                "repository-configured clean/filter helpers"
            ),
        ),
        CommandRule(
            "git-ls-files",
            GIT_LS_FILES_ARGV,
            Domain.GENERIC,
            requires_approval=True,
            reason="Git inspection executes an operator-selected system binary",
        ),
        CommandRule(
            "git-head-commit",
            GIT_HEAD_COMMIT_ARGV,
            Domain.GENERIC,
            requires_approval=True,
            reason="Git inspection executes an operator-selected system binary",
        ),
        CommandRule(
            "git-parent-commit",
            GIT_PARENT_COMMIT_ARGV,
            Domain.GENERIC,
            requires_approval=True,
            reason="Git inspection executes an operator-selected system binary",
        ),
        CommandRule(
            "django-check",
            ("python", "manage.py", "check"),
            Domain.DJANGO,
            requires_approval=True,
            reason="Django checks import and execute workspace code",
            workspace_path_args=(1,),
        ),
        CommandRule(
            "django-test",
            ("python", "manage.py", "test"),
            Domain.DJANGO,
            requires_approval=True,
            reason="Django tests import and execute workspace code",
            workspace_path_args=(1,),
        ),
        CommandRule(
            "django-makemigrations-dry-run",
            ("python", "manage.py", "makemigrations", "--check", "--dry-run"),
            Domain.DJANGO,
            requires_approval=True,
            reason="Django management commands import workspace code",
            workspace_path_args=(1,),
        ),
        CommandRule(
            "django-migrate-plan",
            ("python", "manage.py", "migrate", "--plan"),
            Domain.DJANGO,
            requires_approval=True,
            reason="Django management commands import workspace code",
            workspace_path_args=(1,),
        ),
        CommandRule(
            "pytest",
            ("pytest",),
            Domain.GENERIC,
            requires_approval=True,
            reason="pytest imports and executes workspace code",
        ),
        CommandRule(
            "ruff-check",
            ("ruff", "check", "."),
            Domain.GENERIC,
            workspace_path_args=(2,),
        ),
        CommandRule(
            "pytorch-smoke",
            ("python", "train.py", "--smoke"),
            Domain.PYTORCH,
            requires_approval=True,
            reason="PyTorch smoke jobs execute workspace code",
            workspace_path_args=(1,),
        ),
        CommandRule(
            "pytorch-eval",
            ("python", "eval.py"),
            Domain.PYTORCH,
            requires_approval=True,
            reason="Evaluation jobs execute workspace code",
            workspace_path_args=(1,),
        ),
        CommandRule(
            "gradle-tasks",
            ("gradlew", "--offline", "tasks"),
            Domain.ANDROID,
            requires_approval=True,
            reason="Gradle configuration executes workspace build scripts",
        ),
        CommandRule(
            "gradle-test",
            ("gradlew", "--offline", "test"),
            Domain.ANDROID,
            requires_approval=True,
            reason="Gradle tests execute workspace code",
        ),
        CommandRule(
            "gradle-lint",
            ("gradlew", "--offline", "lint"),
            Domain.ANDROID,
            requires_approval=True,
            reason="Android lint executes workspace build scripts",
        ),
        CommandRule(
            "gradle-assemble-debug",
            ("gradlew", "--offline", "assembleDebug"),
            Domain.ANDROID,
            requires_approval=True,
            reason="Android builds execute workspace build scripts",
        ),
        CommandRule(
            "cmake-build",
            ("cmake", "--build", "build"),
            Domain.ANDROID_NDK,
            requires_approval=True,
            reason="Native builds execute project build scripts",
            workspace_path_args=(2,),
        ),
        CommandRule(
            "xcode-list",
            ("xcodebuild", "-list"),
            Domain.IOS,
            requires_approval=True,
            reason="Xcode project loading may resolve packages and execute project configuration",
        ),
        CommandRule(
            "xcode-build",
            ("xcodebuild", "build"),
            Domain.IOS,
            requires_approval=True,
            reason="Xcode builds execute project build phases",
        ),
        CommandRule(
            "xcode-test",
            ("xcodebuild", "test"),
            Domain.IOS,
            requires_approval=True,
            reason="Xcode tests execute workspace code",
        ),
    ]
    return RunnerPolicy(
        workspace_root=root,
        allowed_commands=rules,
        trusted_executables=trusted,
        allowed_workspace_executables=tuple(workspace_executables),
    )


def _register_executable(
    trusted: dict[str, tuple[Path, ...]],
    workspace_executables: list[Path],
    root: Path,
    alias: str,
    executable: Path | None,
) -> None:
    if executable is None:
        return
    resolved = executable.resolve()
    if resolved.is_relative_to(root):
        workspace_executables.append(resolved)
        return
    trusted[alias] = (*trusted.get(alias, ()), resolved)
