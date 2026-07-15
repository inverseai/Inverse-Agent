"""Executable MCP surface for policy-enforced agent operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.models import AutonomyLevel, Domain, RunKind, WorkspaceProfile
from inverse_agent.redaction import redact_text
from inverse_agent.service import AgentService, RunRecord


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
        return _mcp_profile_view(
            service,
            detect_workspace(workspace),
        )

    @server.tool(structured_output=True)
    def create_run(
        goal: str,
        workspace: str,
        domain: str,
        kind: str = RunKind.VERIFICATION.value,
        autonomy_level: int = AutonomyLevel.ASSISTED.value,
    ) -> dict[str, Any]:
        """Create a durable run without executing any workspace code."""

        record = service.create_run(
            goal=goal,
            workspace=Path(workspace),
            domain=Domain(domain),
            kind=RunKind(kind),
            autonomy_level=AutonomyLevel(autonomy_level),
        )
        return _mcp_run_view(service, record)

    @server.tool(structured_output=True)
    def start_run(run_id: str) -> dict[str, Any]:
        """Queue a run; poll get_run for progress or an approval checkpoint."""

        return _mcp_run_view(service, service.start(run_id, wait=False))

    @server.tool(structured_output=True)
    def get_run(run_id: str) -> dict[str, Any]:
        """Read current durable run state."""

        return _mcp_run_view(service, service.get(run_id))

    @server.tool(structured_output=True)
    def list_runs(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List bounded durable run lifecycle records."""

        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("run pagination is out of range")
        return [
            _mcp_run_view(service, record) for record in service.list(limit=limit, offset=offset)
        ]

    @server.tool(structured_output=True)
    def get_plan(run_id: str) -> dict[str, Any]:
        """Read a run's bounded plan without approval or source content."""

        return _mcp_plan_view(service.plan_view(run_id))

    @server.tool(structured_output=True)
    def get_trace(run_id: str) -> dict[str, Any]:
        """Read bounded lifecycle metadata; command output is intentionally omitted."""

        return _mcp_trace_view(service.trace_preview(run_id))

    return server


def _workspace_ref(service: AgentService, workspace: Path) -> str:
    resolved = workspace.resolve()
    if not resolved.is_relative_to(service.workspace_root):
        raise ValueError("workspace is outside configured workspace root")
    relative = resolved.relative_to(service.workspace_root)
    return relative.as_posix() or "."


def _mcp_profile_view(service: AgentService, profile: WorkspaceProfile) -> dict[str, Any]:
    """Project metadata without absolute executables, roots, or toolchain paths."""

    return {
        "workspace": _workspace_ref(service, profile.root),
        "domains": sorted(domain.value for domain in profile.domains),
        "tools": sorted(profile.commands),
        "unavailable_tools": sorted(profile.unavailable_tools),
        "toolchain_kinds": sorted(profile.toolchain),
        "inference_mode": profile.inference_mode.value,
        "autonomy": {
            domain.value: level.value
            for domain, level in sorted(profile.autonomy.items(), key=lambda item: item[0].value)
        },
    }


def _mcp_run_view(service: AgentService, record: RunRecord) -> dict[str, Any]:
    """Project lifecycle state without approvals, absolute paths, source, or answers."""

    return {
        "run_id": record.run_id,
        "workspace": _workspace_ref(service, Path(record.workspace)),
        "domain": record.domain,
        "kind": record.kind,
        "autonomy_level": record.autonomy_level,
        "status": record.status,
        "plan": list(record.plan),
        "completed_actions": record.completed_actions,
        "stop_reason": record.stop_reason,
        "budget": dict(record.budget or {}),
        "usage": dict(record.usage or {}),
        "has_pending_approval": record.pending_approval is not None,
        "has_answer": record.answer is not None,
        "has_error": record.error is not None,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "attempt": record.attempt,
    }


def _mcp_plan_view(payload: dict[str, Any]) -> dict[str, Any]:
    rationale = redact_text(str(payload.get("rationale", ""))).text[:4096]
    plan = payload.get("plan", [])
    return {
        "run_id": str(payload.get("run_id", "")),
        "status": str(payload.get("status", "")),
        "plan": [str(item)[:256] for item in plan] if isinstance(plan, list) else [],
        "rationale": rationale,
        "completed_actions": (
            payload.get("completed_actions")
            if isinstance(payload.get("completed_actions"), int)
            and not isinstance(payload.get("completed_actions"), bool)
            else 0
        ),
    }


def _mcp_trace_view(payload: dict[str, Any]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    raw_actions = payload.get("actions", [])
    if isinstance(raw_actions, list):
        for raw in raw_actions[:100]:
            if not isinstance(raw, dict):
                continue
            returncode = raw.get("returncode")
            actions.append(
                {
                    "name": str(raw.get("name", ""))[:256],
                    "status": str(raw.get("status", ""))[:128],
                    "rule": str(raw.get("rule", ""))[:256],
                    "returncode": (
                        returncode
                        if isinstance(returncode, int) and not isinstance(returncode, bool)
                        else None
                    ),
                }
            )

    duration = payload.get("duration_seconds")
    return {
        "run_id": str(payload.get("run_id", "")),
        "status": str(payload.get("status", "")),
        "duration_seconds": (
            float(duration)
            if isinstance(duration, int | float) and not isinstance(duration, bool)
            else 0.0
        ),
        "actions": actions,
        "actions_truncated": bool(payload.get("actions_truncated")),
        "output_omitted": True,
    }
