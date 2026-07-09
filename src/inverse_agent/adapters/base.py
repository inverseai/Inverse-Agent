"""MCP-style adapter base classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from inverse_agent.models import Artifact, Domain, WorkspaceProfile
from inverse_agent.runner import CommandRequest, CommandResult, LocalRunner


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    safety: str
    domain: Domain


@dataclass
class ToolResult:
    name: str
    ok: bool
    summary: str
    command: CommandResult | None = None
    artifacts: list[Artifact] = field(default_factory=list)


class ToolchainAdapter(Protocol):
    domain: Domain

    def detect(self, root: Path) -> bool:
        ...

    def profile(self, root: Path) -> WorkspaceProfile:
        ...

    def tools(self) -> list[Tool]:
        ...


class CommandAdapter:
    domain: Domain = Domain.GENERIC

    def run_command(
        self,
        runner: LocalRunner,
        workspace: Path,
        argv: list[str],
        *,
        approved: bool = False,
    ) -> CommandResult:
        return runner.run(
            CommandRequest(
                argv=tuple(argv),
                cwd=workspace,
                domain=self.domain,
                approved=approved,
            )
        )

