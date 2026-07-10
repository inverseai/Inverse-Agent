"""Executable MCP surface for policy-enforced agent operations."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from mcp.server.fastmcp import FastMCP

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.eval import json_default
from inverse_agent.models import AutonomyLevel, Domain
from inverse_agent.service import AgentService


def create_mcp_server(service: AgentService) -> FastMCP:
    server = FastMCP(
        "Inverse-Agent",
        instructions=(
            "Profile engineering workspaces and start policy-enforced runs. "
            "Human approvals are intentionally unavailable over MCP."
        ),
    )

    @server.tool(structured_output=True)
    def profile_workspace(path: str) -> dict[str, Any]:
        """Detect toolchains under the configured workspace root."""

        workspace = Path(path).resolve()
        if not workspace.is_relative_to(service.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        return _json_safe(asdict(detect_workspace(workspace)))

    @server.tool(structured_output=True)
    def create_run(
        goal: str,
        workspace: str,
        domain: str,
        autonomy_level: int = AutonomyLevel.ASSISTED.value,
    ) -> dict[str, Any]:
        """Create a durable run without executing any workspace code."""

        record = service.create_run(
            goal=goal,
            workspace=Path(workspace),
            domain=Domain(domain),
            autonomy_level=AutonomyLevel(autonomy_level),
        )
        return asdict(record)

    @server.tool(structured_output=True)
    def start_run(run_id: str) -> dict[str, Any]:
        """Plan a run and stop at its first human-approval checkpoint."""

        return asdict(service.start(run_id))

    @server.tool(structured_output=True)
    def get_run(run_id: str) -> dict[str, Any]:
        """Read current durable run state."""

        return asdict(service.get(run_id))

    return server


def _json_safe(value: Any) -> dict[str, Any]:
    import json

    return cast(dict[str, Any], json.loads(json.dumps(value, default=json_default)))
