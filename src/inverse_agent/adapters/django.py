"""Django toolchain adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool, ToolResult
from inverse_agent.models import Domain, WorkspaceProfile
from inverse_agent.runner import LocalRunner


class DjangoAdapter(CommandAdapter):
    domain = Domain.DJANGO

    def detect(self, root: Path) -> bool:
        return (root / "manage.py").exists()

    def profile(self, root: Path) -> WorkspaceProfile:
        python = sys.executable
        return WorkspaceProfile(
            root=root,
            domains={Domain.DJANGO},
            commands={
                "check": [python, "manage.py", "check"],
                "test": [python, "manage.py", "test"],
                "makemigrations_check": [
                    python,
                    "manage.py",
                    "makemigrations",
                    "--check",
                    "--dry-run",
                ],
                "migrate_plan": [python, "manage.py", "migrate", "--plan"],
            },
            test_targets=["manage.py test"],
            toolchain={"python": python, "framework": "django"},
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("django.check", "Run Django system checks", "safe-read", self.domain),
            Tool("django.test", "Run Django tests", "safe-read", self.domain),
            Tool("django.migration_plan", "Inspect pending migrations", "safe-read", self.domain),
        ]

    def run_checks(self, runner: LocalRunner, root: Path) -> ToolResult:
        profile = self.profile(root)
        result = self.run_command(runner, root, profile.commands["check"])
        return ToolResult(
            name="django.check",
            ok=result.status.value == "succeeded",
            summary=result.stdout or result.stderr or result.reason,
            command=result,
        )

    def run_tests(self, runner: LocalRunner, root: Path, *, approved: bool = False) -> ToolResult:
        profile = self.profile(root)
        result = self.run_command(runner, root, profile.commands["test"], approved=approved)
        return ToolResult(
            name="django.test",
            ok=result.status.value == "succeeded",
            summary=result.stdout or result.stderr or result.reason,
            command=result,
        )
