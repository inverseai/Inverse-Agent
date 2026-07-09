from pathlib import Path

from inverse_agent.control_plane import create_app


def test_control_plane_health_and_approvals() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    approval = client.post("/runs/run-1/approvals", json={"approved": True, "by": "tester"})
    assert approval.status_code == 200
    assert approval.json() == {"run_id": "run-1", "approval_count": 1}
    assert client.get("/runs/run-1").json()["approvals"] == [{"approved": True, "by": "tester"}]


def test_control_plane_restricts_profile_to_workspace_root(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    app = create_app(workspace_root=tmp_path, api_token="secret")
    client = TestClient(app)

    assert client.get("/profile", params={"path": str(tmp_path)}).status_code == 401
    ok = client.get(
        "/profile",
        params={"path": str(tmp_path)},
        headers={"X-Inverse-Agent-Token": "secret"},
    )
    assert ok.status_code == 200
    forbidden = client.get(
        "/profile",
        params={"path": str(tmp_path.parent)},
        headers={"X-Inverse-Agent-Token": "secret"},
    )
    assert forbidden.status_code == 403
