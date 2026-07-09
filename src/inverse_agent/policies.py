"""Policy factories and validation."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.models import CommandRule, Domain, RunnerPolicy

SHELL_METACHARS = {
    "!",
    "$(",
    "%",
    "&",
    "&&",
    ";",
    "<",
    ">",
    ">>",
    "^",
    "`",
    "|",
    "||",
}


def has_shell_metachar(value: str) -> bool:
    return any(token in value for token in SHELL_METACHARS)


def default_policy(workspace_root: Path) -> RunnerPolicy:
    """Create the default-deny command policy with narrow allowlists."""

    return RunnerPolicy(
        workspace_root=workspace_root,
        allowed_commands=[
            CommandRule("git-status", ("git", "status"), Domain.GENERIC),
            CommandRule("git-diff", ("git", "diff"), Domain.GENERIC),
            CommandRule("git-log", ("git", "log"), Domain.GENERIC),
            CommandRule("git-show", ("git", "show"), Domain.GENERIC),
            CommandRule("git-ls-files", ("git", "ls-files"), Domain.GENERIC),
            CommandRule("django-check", ("python", "manage.py", "check"), Domain.DJANGO),
            CommandRule(
                "django-test",
                ("python", "manage.py", "test"),
                Domain.DJANGO,
                requires_approval=True,
                reason="Django tests import and execute workspace code",
            ),
            CommandRule(
                "django-makemigrations-dry-run",
                ("python", "manage.py", "makemigrations", "--check", "--dry-run"),
                Domain.DJANGO,
            ),
            CommandRule("django-migrate-plan", ("python", "manage.py", "migrate", "--plan"), Domain.DJANGO),
            CommandRule(
                "pytest",
                ("pytest",),
                Domain.GENERIC,
                requires_approval=True,
                reason="pytest imports and executes workspace code",
            ),
            CommandRule("ruff-check", ("ruff", "check"), Domain.GENERIC),
            CommandRule(
                "pytorch-smoke",
                ("python", "train.py", "--smoke"),
                Domain.PYTORCH,
                requires_approval=True,
                reason="PyTorch smoke jobs execute workspace code",
            ),
            CommandRule(
                "pytorch-eval",
                ("python", "eval.py"),
                Domain.PYTORCH,
                requires_approval=True,
                reason="Evaluation jobs execute workspace code",
            ),
            CommandRule("gradle-tasks", ("gradlew", "tasks"), Domain.ANDROID),
            CommandRule(
                "gradle-test",
                ("gradlew", "test"),
                Domain.ANDROID,
                requires_approval=True,
                reason="Gradle tests execute workspace code",
            ),
            CommandRule(
                "gradle-lint",
                ("gradlew", "lint"),
                Domain.ANDROID,
                requires_approval=True,
                network_required=True,
                reason="Gradle may resolve dependencies or plugins",
            ),
            CommandRule(
                "gradle-assemble-debug",
                ("gradlew", "assembleDebug"),
                Domain.ANDROID,
                requires_approval=True,
                network_required=True,
                reason="Android builds may resolve dependencies and execute build scripts",
            ),
            CommandRule(
                "cmake-build",
                ("cmake", "--build"),
                Domain.ANDROID_NDK,
                requires_approval=True,
                reason="Native builds execute project build scripts",
            ),
            CommandRule("xcode-list", ("xcodebuild", "-list"), Domain.IOS),
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
        ],
    )
