import json
import sqlite3
import time
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


def _client(service: AgentService) -> TestClient:
    app = create_app(
        service=service,
        api_token="api-secret",
        approver_tokens=APPROVERS,
        planner_summary={
            "kind": "openai-compatible",
            "model": "test-model",
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key_set": True,
        },
    )
    return TestClient(app, base_url="http://127.0.0.1")


def _wait_for_status(
    client: TestClient,
    run_id: str,
    *statuses: RunStatus,
) -> dict[str, object]:
    expected = {status.value for status in statuses}
    deadline = time.monotonic() + 5
    while True:
        response = client.get(f"/runs/{run_id}", headers=HEADERS)
        assert response.status_code == 200
        record = response.json()
        if record["status"] in expected:
            return record
        if time.monotonic() >= deadline:
            raise AssertionError(f"run did not reach {sorted(expected)}: {record}")
        time.sleep(0.01)


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
        client = _client(service)
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.css").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "api_version": "2026-07-15.v3"}
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/profile", params={"path": str(FIXTURES)}).status_code == 401
        assert client.get("/runtime").status_code == 401
        assert (
            client.get("/workspaces/trust-status", params={"path": str(FIXTURES)}).status_code
            == 401
        )
        assert client.get("/runs").status_code == 401
        assert client.post("/runs", json={}).status_code == 401
        assert client.post("/workspaces/trust", json={}).status_code == 401
        assert client.request("DELETE", "/workspaces/trust", json={}).status_code == 401
        assert client.get("/runs/unknown").status_code == 401
        assert client.get("/runs/unknown/plan").status_code == 401
        assert client.get("/runs/unknown/trace").status_code == 401
        assert client.get("/runs/unknown/events").status_code == 401
        assert client.post("/runs/unknown/start").status_code == 401
        assert client.post("/runs/unknown/cancel").status_code == 401
        assert client.post("/runs/unknown/approvals", json={}).status_code == 401
        assert client.post("/runs/unknown/decline", json={}).status_code == 401
        assert client.get("/approver/session").status_code == 401
        assert (
            client.get(
                "/runs",
                headers=[(b"x-inverse-agent-token", b"non-ascii-\xff")],
            ).status_code
            == 401
        )
    finally:
        service.close()


def test_operator_token_cannot_use_approver_endpoints(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
        trust = client.post(
            "/workspaces/trust",
            headers=HEADERS,
            json={"workspace": str(FIXTURES / "django_project")},
        )
        revoke = client.request(
            "DELETE",
            "/workspaces/trust",
            headers=HEADERS,
            json={"workspace": str(FIXTURES / "django_project")},
        )
        approval = client.post(
            "/runs/unknown/approvals",
            headers=HEADERS,
            json={
                "action_digest": "0" * 64,
                "challenge_id": "0" * 32,
            },
        )
        decline = client.post(
            "/runs/unknown/decline",
            headers=HEADERS,
            json={"action_digest": "0" * 64, "challenge_id": "0" * 32},
        )
        approver_session = client.get("/approver/session", headers=HEADERS)
        assert trust.status_code == 401
        assert revoke.status_code == 401
        assert approval.status_code == 401
        assert decline.status_code == 401
        assert approver_session.status_code == 401
    finally:
        service.close()


def test_control_plane_end_to_end_approval_flow(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
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
        accepted = client.post(f"/runs/{run_id}/start", headers=HEADERS)
        assert accepted.status_code == 202
        assert accepted.json()["status"] == RunStatus.QUEUED.value
        waiting = _wait_for_status(client, run_id, RunStatus.WAITING_FOR_APPROVAL)
        assert waiting["plan"] == ["django.check", "django.test"]
        assert waiting["plan_rationale"]
        assert "trace_path" not in waiting
        stale = client.post(
            f"/runs/{run_id}/approvals",
            headers=APPROVER_HEADERS,
            json={
                "action_digest": "0" * 64,
                "challenge_id": waiting["pending_approval"]["challenge_id"],
            },
        )
        assert stale.status_code == 409
        queued_again = client.post(
            f"/runs/{run_id}/approvals",
            headers=APPROVER_HEADERS,
            json={
                "action_digest": waiting["pending_approval"]["action_digest"],
                "challenge_id": waiting["pending_approval"]["challenge_id"],
            },
        )
        assert queued_again.status_code == 202
        assert queued_again.json()["status"] == RunStatus.QUEUED.value
        waiting_again = _wait_for_status(client, run_id, RunStatus.WAITING_FOR_APPROVAL)
        queued_final = client.post(
            f"/runs/{run_id}/approvals",
            headers=APPROVER_HEADERS,
            json={
                "action_digest": waiting_again["pending_approval"]["action_digest"],
                "challenge_id": waiting_again["pending_approval"]["challenge_id"],
            },
        )
        assert queued_final.status_code == 202
        final = _wait_for_status(client, run_id, RunStatus.SUCCEEDED)
        assert final["has_trace"] is True
        trace = client.get(f"/runs/{run_id}/trace", headers=HEADERS)
        assert trace.status_code == 200
        assert [action["name"] for action in trace.json()["actions"]] == [
            "django.check",
            "django.test",
        ]
    finally:
        service.close()


def test_control_plane_restricts_profile_to_workspace_root(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
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


def test_ui_assets_and_security_headers_are_strict(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
        responses = [
            client.get("/"),
            client.get("/assets/app.css"),
            client.get("/assets/app.js"),
            client.get("/health"),
            client.get("/runtime", headers=HEADERS),
        ]
        for response in responses:
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["referrer-policy"] == "no-referrer"
            assert response.headers["cross-origin-opener-policy"] == "same-origin"
            assert response.headers["cross-origin-embedder-policy"] == "require-corp"
            assert response.headers["cross-origin-resource-policy"] == "same-origin"
            assert "unsafe-inline" not in response.headers["content-security-policy"]
            assert "unsafe-eval" not in response.headers["content-security-policy"]
            assert (
                "require-trusted-types-for 'script'" in response.headers["content-security-policy"]
            )
            assert "access-control-allow-origin" not in response.headers
            assert "set-cookie" not in response.headers

        assert responses[0].headers["content-type"].startswith("text/html")
        assert responses[1].headers["content-type"].startswith("text/css")
        assert responses[2].headers["content-type"].startswith("text/javascript")
        assert client.get("/assets/unknown.js").status_code == 404
        assert client.get("/assets/../pyproject.toml").status_code == 404
        assert client.get("/", headers={"host": "evil.example"}).status_code == 400

        javascript = responses[2].text
        for banned in (
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "document.write",
            "new Function",
            "localStorage",
            "inverse-agent.approver",
        ):
            assert banned not in javascript
        assert "sessionStorage" in javascript
        assert 'return "Plan ready"' in javascript
        assert 'stateLabel = "planned"' in javascript
        assert "navigationEpoch" in javascript
        assert "startExistingRun" in javascript
        assert '"Start run"' in javascript
        assert "kind-select" in responses[0].text
        assert "consent-list" in responses[0].text
        assert "source_read" in javascript
        assert "code_execution" in javascript
        assert "/events?after=" in javascript
        assert "/cancel" in javascript
        assert 'method: "DELETE"' in javascript
        assert "resetEvents" in javascript
        assert "answerView" in javascript
        assert "budgetView" in javascript
        assert "mobileInspectorQuery" in javascript
        assert "syncResponsiveInspector" in javascript
        assert ".inspector-toggle {\n    display: none" not in responses[1].text
        assert "linear-gradient" not in responses[1].text
        assert "api-secret" not in responses[0].text + javascript
        assert "approver-secret" not in responses[0].text + javascript
    finally:
        service.close()


def test_runtime_trust_and_approver_read_models_are_separated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
        runtime = client.get("/runtime", headers=HEADERS)
        assert runtime.status_code == 200
        assert runtime.json()["api_version"] == "2026-07-15.v3"
        assert runtime.json()["planner"] == {
            "kind": "openai-compatible",
            "model": "test-model",
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key_set": True,
        }
        assert "api_key" not in runtime.json()["planner"]

        before = client.get(
            "/workspaces/trust-status",
            params={"path": str(FIXTURES / "django_project")},
            headers=HEADERS,
        )
        assert before.json()["trusted"] is False
        approver = client.get("/approver/session", headers=APPROVER_HEADERS)
        assert approver.json() == {"approver": "human@example.test"}
        client.post(
            "/workspaces/trust",
            headers=APPROVER_HEADERS,
            json={"workspace": str(FIXTURES / "django_project")},
        )
        after = client.get(
            "/workspaces/trust-status",
            params={"path": str(FIXTURES / "django_project")},
            headers=HEADERS,
        )
        assert after.json()["trusted"] is True
        assert after.json()["trusted_by"] == "human@example.test"
    finally:
        service.close()


def test_scoped_source_trust_can_be_granted_and_revoked(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = FIXTURES / "django_project"
    try:
        client = _client(service)
        granted = client.post(
            "/workspaces/trust",
            headers=APPROVER_HEADERS,
            json={"workspace": str(workspace), "scope": "source_read"},
        )
        assert granted.status_code == 200
        assert granted.json()["scope"] == "source_read"
        status = client.get(
            "/workspaces/trust-status",
            params={"path": str(workspace)},
            headers=HEADERS,
        ).json()
        assert status["trusted"] is False
        assert status["scopes"]["source_read"]["granted_by"] == "human@example.test"

        revoked = client.request(
            "DELETE",
            "/workspaces/trust",
            headers=APPROVER_HEADERS,
            json={"workspace": str(workspace), "scope": "source_read"},
        )
        assert revoked.status_code == 200
        assert revoked.json() == {
            "workspace": str(workspace.resolve()),
            "scope": "source_read",
            "revoked": True,
        }
        after = client.get(
            "/workspaces/trust-status",
            params={"path": str(workspace)},
            headers=HEADERS,
        ).json()
        assert "source_read" not in after["scopes"]
    finally:
        service.close()


def test_run_kind_events_and_cancel_are_exposed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
        created = client.post(
            "/runs",
            headers=HEADERS,
            json={
                "goal": "Inspect fixture",
                "workspace": str(FIXTURES / "django_project"),
                "domain": "django",
                "kind": "investigation",
                "autonomy_level": 0,
            },
        )
        assert created.status_code == 200
        record = created.json()
        assert record["kind"] == "investigation"
        assert record["budget"]["max_decisions"] == 20
        invalid_budget = client.post(
            "/runs",
            headers=HEADERS,
            json={
                "goal": "Invalid budget",
                "workspace": str(FIXTURES / "django_project"),
                "domain": "django",
                "kind": "investigation",
                "autonomy_level": 0,
                "budget": {"unknown_cap": 1},
            },
        )
        assert invalid_budget.status_code == 400
        assert invalid_budget.json()["detail"] == "investigation budget contains unknown fields"

        events = client.get(
            f"/runs/{record['run_id']}/events",
            headers=HEADERS,
            params={"after": 0, "wait_seconds": 0, "limit": 1},
        )
        assert events.status_code == 200
        payload = events.json()
        assert len(payload["events"]) == 1
        assert payload["events"][0]["sequence"] == payload["next_cursor"]
        assert payload["has_more"] is True
        follow_up = client.get(
            f"/runs/{record['run_id']}/events",
            headers=HEADERS,
            params={"after": payload["next_cursor"], "wait_seconds": 0},
        ).json()
        assert follow_up["events"] == []
        assert follow_up["next_cursor"] == payload["next_cursor"]

        service.runs.append_event(
            record["run_id"],
            "investigation.observation",
            {
                "observation": {
                    "path": "command/django.check",
                    "metadata": {
                        "command_name": "django.check",
                        "status": "succeeded",
                        "approval_id": "private-approval-id",
                        "action_digest": "private-action-digest",
                        "challenge_id": "private-challenge-id",
                        "evidence_identity": "private-evidence-identity",
                        "rule": "private-rule",
                        "argv": ["private", "argv"],
                    },
                },
                "grant_expires_at": 9999999999,
            },
        )
        projected = client.get(
            f"/runs/{record['run_id']}/events",
            headers=HEADERS,
            params={"after": payload["next_cursor"], "wait_seconds": 0},
        ).json()["events"][0]
        projected_text = json.dumps(projected, sort_keys=True)
        assert projected["payload"]["observation"]["metadata"] == {
            "command_name": "django.check",
            "status": "succeeded",
        }
        for private_value in (
            "private-approval-id",
            "private-action-digest",
            "private-challenge-id",
            "private-evidence-identity",
            "private-rule",
            "private",
            "9999999999",
        ):
            assert private_value not in projected_text

        cancelled = client.post(f"/runs/{record['run_id']}/cancel", headers=HEADERS)
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == RunStatus.CANCELLED.value
    finally:
        service.close()


def test_decline_is_digest_bound_and_terminal(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
        client.post(
            "/workspaces/trust",
            headers=APPROVER_HEADERS,
            json={"workspace": str(FIXTURES / "django_project")},
        )
        created = client.post(
            "/runs",
            headers=HEADERS,
            json={
                "goal": "Do not execute this check",
                "workspace": str(FIXTURES / "django_project"),
                "domain": "django",
                "autonomy_level": 1,
            },
        ).json()
        accepted = client.post(f"/runs/{created['run_id']}/start", headers=HEADERS)
        assert accepted.status_code == 202
        waiting = _wait_for_status(
            client,
            created["run_id"],
            RunStatus.WAITING_FOR_APPROVAL,
        )
        stale = client.post(
            f"/runs/{created['run_id']}/decline",
            headers=APPROVER_HEADERS,
            json={
                "action_digest": "0" * 64,
                "challenge_id": waiting["pending_approval"]["challenge_id"],
            },
        )
        assert stale.status_code == 409
        declined = client.post(
            f"/runs/{created['run_id']}/decline",
            headers=APPROVER_HEADERS,
            json={
                "action_digest": waiting["pending_approval"]["action_digest"],
                "challenge_id": waiting["pending_approval"]["challenge_id"],
            },
        )
        assert declined.json()["status"] == RunStatus.REFUSED.value
        assert declined.json()["error"] == "declined by human@example.test"
        replay = client.post(
            f"/runs/{created['run_id']}/approvals",
            headers=APPROVER_HEADERS,
            json={
                "action_digest": waiting["pending_approval"]["action_digest"],
                "challenge_id": waiting["pending_approval"]["challenge_id"],
            },
        )
        assert replay.status_code == 409
    finally:
        service.close()


def test_trace_preview_is_derived_redacted_allowlisted_and_bounded(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
        created = client.post(
            "/runs",
            headers=HEADERS,
            json={
                "goal": "Trace preview",
                "workspace": str(FIXTURES / "django_project"),
                "domain": "django",
                "autonomy_level": 0,
            },
        ).json()
        run_id = created["run_id"]
        assert client.get(f"/runs/{run_id}/trace", headers=HEADERS).status_code == 404

        trace_dir = service.state_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        canary = "</script><script>window.compromised=true</script>"
        secret = "token=super-secret-preview-value"
        payload = {
            "duration_seconds": 1.25,
            "actions": [
                {
                    "name": "django.check",
                    "metadata": {
                        "status": "succeeded",
                        "rule": "django-check",
                        "reason": secret + ("r" * 10_000),
                        "returncode": "must-not-leak",
                        "stdout": canary + secret + " " + ("x" * 30_000),
                        "stderr": "",
                        "argv": ["must-not-leak"],
                        "approval_id": "must-not-leak",
                    },
                    "at": "2026-07-10T00:00:00Z",
                }
            ],
        }
        (trace_dir / f"{run_id}.trace.json").write_text(json.dumps(payload), encoding="utf-8")
        outside = tmp_path / "outside.trace.json"
        outside.write_text('{"actions": []}', encoding="utf-8")
        with sqlite3.connect(service.runs.path) as connection:
            connection.execute(
                "UPDATE runs SET trace_path=? WHERE run_id=?",
                (str(outside), run_id),
            )

        response = client.get(f"/runs/{run_id}/trace", headers=HEADERS)
        assert response.status_code == 200
        action = response.json()["actions"][0]
        assert set(action) == {
            "name",
            "status",
            "rule",
            "reason",
            "returncode",
            "stdout",
            "stdout_truncated",
            "stderr",
            "stderr_truncated",
        }
        assert canary in action["stdout"]
        assert "super-secret-preview-value" not in action["stdout"]
        assert "[REDACTED_SECRET]" in action["stdout"]
        assert len(action["stdout"]) <= 16_384
        assert action["stdout_truncated"] is True
        assert len(action["reason"]) <= 4_096
        assert "super-secret-preview-value" not in action["reason"]
        assert action["returncode"] is None
        assert str(outside) not in response.text
        assert "must-not-leak" not in response.text
    finally:
        service.close()


def test_run_listing_is_paginated_and_hides_state_paths(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        client = _client(service)
        for index in range(2):
            client.post(
                "/runs",
                headers=HEADERS,
                json={
                    "goal": f"Task {index}",
                    "workspace": str(FIXTURES / "django_project"),
                    "domain": "django",
                    "autonomy_level": 0,
                },
            )
        first_page = client.get("/runs", params={"limit": 1, "offset": 0}, headers=HEADERS)
        second_page = client.get("/runs", params={"limit": 1, "offset": 1}, headers=HEADERS)
        assert len(first_page.json()) == 1
        assert len(second_page.json()) == 1
        assert first_page.json()[0]["run_id"] != second_page.json()[0]["run_id"]
        assert "trace_path" not in first_page.json()[0]
        assert client.get("/runs", params={"limit": 501}, headers=HEADERS).status_code == 422
    finally:
        service.close()
