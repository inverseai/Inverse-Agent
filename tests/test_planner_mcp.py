import asyncio
from pathlib import Path
from typing import Any

import pytest

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.mcp_server import create_mcp_server
from inverse_agent.models import Domain
from inverse_agent.planner import StructuredPlanner
from inverse_agent.service import AgentService

FIXTURES = Path(__file__).parent / "fixtures"
SECRET = b"test-mcp-secret-that-is-at-least-32-bytes"


class FakeModel:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.prompt = ""

    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        assert "Never emit commands" in system
        assert "available_tools" in prompt
        self.prompt = prompt
        return self.response


def test_structured_planner_accepts_only_registered_tools() -> None:
    profile = detect_workspace(FIXTURES / "django_project")
    model = FakeModel({"actions": ["django.check"], "rationale": "api_key=supersecretvalue"})
    planner = StructuredPlanner(model)
    plan = planner.plan(
        goal="verify with token=anothersecretvalue",
        domain=Domain.DJANGO,
        profile=profile,
        available_tools=("django.check", "django.test"),
    )
    assert [action.tool_name for action in plan.actions] == ["django.check"]
    assert "anothersecretvalue" not in model.prompt
    assert "supersecretvalue" not in plan.rationale
    assert "[REDACTED_SECRET]" in plan.rationale


def test_structured_planner_rejects_raw_or_unknown_action() -> None:
    profile = detect_workspace(FIXTURES / "django_project")
    planner = StructuredPlanner(FakeModel({"actions": ["python manage.py test"]}))
    with pytest.raises(ValueError, match="unknown tool"):
        planner.plan(
            goal="verify",
            domain=Domain.DJANGO,
            profile=profile,
            available_tools=("django.check",),
        )


def test_structured_planner_rejects_duplicate_actions() -> None:
    profile = detect_workspace(FIXTURES / "django_project")
    planner = StructuredPlanner(FakeModel({"actions": ["django.check", "django.check"]}))
    with pytest.raises(ValueError, match="duplicate tool"):
        planner.plan(
            goal="verify",
            domain=Domain.DJANGO,
            profile=profile,
            available_tools=("django.check",),
        )


def test_mcp_lists_and_calls_policy_backed_tools(tmp_path: Path) -> None:
    service = AgentService(
        workspace_root=FIXTURES,
        state_dir=tmp_path / "state",
        approval_secret=SECRET,
    )
    server = create_mcp_server(service)

    async def exercise() -> None:
        tools = await server.list_tools()
        names = {tool.name for tool in tools}
        assert {"profile_workspace", "create_run", "start_run", "get_run"} <= names
        assert "approve_run" not in names
        result = await server.call_tool(
            "profile_workspace",
            {"path": str(FIXTURES / "django_project")},
        )
        assert result

    try:
        asyncio.run(exercise())
    finally:
        service.close()
