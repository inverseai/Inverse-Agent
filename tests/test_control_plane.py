from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inverse_agent.control_plane import create_app
from inverse_agent.models import RunStatus
from inverse_agent.service import AgentService

FIXTURES = Path(__file__).parent / "fixtures"
SECRET = b"test-control-secret-that-is-at-least-32-bytes"
HEADERS = {"X-Inverse-Agent-Token": "api-secret"}
APPROVER_HEADERS = {"X-Inverse-Agent-Approval-Token": "approver-secret"}
APPROVERS = {"approver-secret": "human@example.test"}


def _service(tmp_path: Path) -> AgentService:
    return AgentService(
        workspace_root=FIXTURES,
        state_dir=tmp_path / "state",
        approval_secret=SECRET,
    )


def test_control_plane_refuses_empty_api_token(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        with pytest.raises(ValueError, match="API token is required"):
            create_app(service=service, api_token="", approver_tokens=APPROVERS)
    finally:
        service.close()


def test_control_plane_requires_distinct_approver_credentials(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        with pytest.raises(ValueError, match="approver token"):
            create_app(service=service, api_token="api-secret", approver_tokens={})
        with pytest.raises(ValueError, match="must be distinct"):
            create_app(
                service=service,
                api_token="same-secret",
                approver_tokens={"same-secret": "human@example.test"},
            )
    finally:
        service.close()


def test_every_endpoint_except_health_requires_auth(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = TestClient(
            create_app(service=service, api_token="api-secret", approver_tokens=APPROVERS)
        )
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/profile", params={"path": str(FIXTURES)}).status_code == 401
        assert client.get("/runs").status_code == 401
        assert client.post("/runs", json={}).status_code == 401
        assert client.post("/workspaces/trust", json={}).status_code == 401
        assert client.get("/runs/unknown").status_code == 401
        assert client.post("/runs/unknown/start").status_code == 401
        assert client.post("/runs/unknown/approvals", json={}).status_code == 401
    finally:
        service.close()


def test_control_plane_end_to_end_approval_flow(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = TestClient(
            create_app(service=service, api_token="api-secret", approver_tokens=APPROVERS)
        )
        created = client.post(
            "/runs",
            headers=HEADERS,
            json={
                "goal": "Verify fixture",
                "workspace": str(FIXTURES / "django_project"),
                "domain": "django",
                "autonomy_level": 1,
            },
        )
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        trusted = client.post(
            "/workspaces/trust",
            headers=APPROVER_HEADERS,
            json={
                "workspace": str(FIXTURES / "django_project"),
            },
        )
        assert trusted.status_code == 200
        assert trusted.json()["trusted_by"] == "human@example.test"
        waiting = client.post(f"/runs/{run_id}/start", headers=HEADERS)
        assert waiting.json()["status"] == RunStatus.WAITING_FOR_APPROVAL.value
        waiting_again = client.post(
            f"/runs/{run_id}/approvals",
            headers=APPROVER_HEADERS,
            json={"action_digest": waiting.json()["pending_approval"]["action_digest"]},
        )
        assert waiting_again.json()["status"] == RunStatus.WAITING_FOR_APPROVAL.value
        final = client.post(
            f"/runs/{run_id}/approvals",
            headers=APPROVER_HEADERS,
            json={
                "action_digest": waiting_again.json()["pending_approval"]["action_digest"]
            },
        )
        assert final.json()["status"] == RunStatus.SUCCEEDED.value
    finally:
        service.close()


def test_control_plane_restricts_profile_to_workspace_root(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = TestClient(
            create_app(service=service, api_token="api-secret", approver_tokens=APPROVERS)
        )
        ok = client.get(
            "/profile",
            params={"path": str(FIXTURES)},
            headers=HEADERS,
        )
        forbidden = client.get(
            "/profile",
            params={"path": str(FIXTURES.parent)},
            headers=HEADERS,
        )
        assert ok.status_code == 200
        assert forbidden.status_code == 403
    finally:
        service.close()
