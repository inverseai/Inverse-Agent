"""Generic Git repository inspection adapter."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool
from inverse_agent.environments import discover_system_executable
from inverse_agent.models import Domain, WorkspaceProfile
from inverse_agent.policies import (
    GIT_HEAD_COMMIT_ARGV,
    GIT_LS_FILES_ARGV,
    GIT_PARENT_COMMIT_ARGV,
    GIT_STATUS_ARGV,
)


class GenericGitAdapter(CommandAdapter):
    domain = Domain.GENERIC

    def detect(self, root: Path) -> bool:
        return (root / ".git").exists()

    def profile(self, root: Path) -> WorkspaceProfile:
        git = discover_system_executable("git")
        commands: dict[str, list[str]] = {}
        unavailable: dict[str, str] = {}
        if git:
            commands = {
                "status": [str(git), *GIT_STATUS_ARGV[1:]],
                "tracked_files": [str(git), *GIT_LS_FILES_ARGV[1:]],
                "head_commit": [str(git), *GIT_HEAD_COMMIT_ARGV[1:]],
                "parent_commit": [str(git), *GIT_PARENT_COMMIT_ARGV[1:]],
            }
        else:
            unavailable["git"] = "Git executable not found"
        return WorkspaceProfile(
            root=root,
            domains={Domain.GENERIC},
            commands=commands,
            test_targets=[],
            toolchain={"git": str(git) if git else "unavailable"},
            unavailable_tools=unavailable,
        )

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "generic.status",
                "Inspect branch and tracked working-tree status",
                "approval-required",
                self.domain,
            ),
            Tool(
                "generic.tracked_files",
                "List files tracked by Git",
                "approval-required",
                self.domain,
            ),
            Tool(
                "generic.head_commit",
                "Resolve the current HEAD commit",
                "approval-required",
                self.domain,
            ),
            Tool(
                "generic.parent_commit",
                "Resolve HEAD's first parent when one exists",
                "approval-required",
                self.domain,
            ),
        ]
