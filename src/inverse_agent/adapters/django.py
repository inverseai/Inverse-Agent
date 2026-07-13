"""Django toolchain adapter."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool, ToolResult
from inverse_agent.environments import discover_python
from inverse_agent.models import Domain, WorkspaceProfile
from inverse_agent.runner import LocalRunner


class DjangoAdapter(CommandAdapter):
    domain = Domain.DJANGO

    def detect(self, root: Path) -> bool:
        return (root / "manage.py").exists()

    def profile(self, root: Path) -> WorkspaceProfile:
        environment = discover_python(root)
        python = str(environment.path)
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
            toolchain={
                "python": python,
                "python_source": environment.source,
                "framework": "django",
            },
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("django.check", "Run Django system checks", "approval-required", self.domain),
            Tool("django.test", "Run Django tests", "approval-required", self.domain),
            Tool(
                "django.migration_plan",
                "Inspect pending migrations",
                "approval-required",
                self.domain,
            ),
        ]

    def run_checks(
        self,
        runner: LocalRunner,
        root: Path,
        *,
        approval_token: str | None = None,
        approval_challenge_id: str | None = None,
    ) -> ToolResult:
        profile = self.profile(root)
        result = self.run_command(
            runner,
            root,
            profile.commands["check"],
            approval_token=approval_token,
            approval_challenge_id=approval_challenge_id,
        )
        return ToolResult(
            name="django.check",
            ok=result.status.value == "succeeded",
            summary=result.stdout or result.stderr or result.reason,
            command=result,
        )

    def run_tests(
        self,
        runner: LocalRunner,
        root: Path,
        *,
        approval_token: str | None = None,
        approval_challenge_id: str | None = None,
    ) -> ToolResult:
        profile = self.profile(root)
        result = self.run_command(
            runner,
            root,
            profile.commands["test"],
            approval_token=approval_token,
            approval_challenge_id=approval_challenge_id,
        )
        return ToolResult(
            name="django.test",
            ok=result.status.value == "succeeded",
            summary=result.stdout or result.stderr or result.reason,
            command=result,
        )
