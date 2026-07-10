import json
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from inverse_agent.cli import main
from inverse_agent.models import AutonomyLevel, Domain, RunStatus
from inverse_agent.service import AgentService

FIXTURES = Path(__file__).parent / "fixtures"
SECRET = b"test-workflow-secret-that-is-at-least-32-bytes"


def _service(state_dir: Path) -> AgentService:
    return AgentService(
        workspace_root=FIXTURES,
        state_dir=state_dir,
        approval_secret=SECRET,
    )


def _approve(service: AgentService, record, approved_by: str):
    return service.approve_and_resume(
        record.run_id,
        approved_by=approved_by,
        expected_action_digest=record.pending_approval["action_digest"],
    )


def test_django_workflow_pauses_for_each_action_then_succeeds(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
        created = service.create_run(
            goal="Verify Django fixture",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
        )
        first = service.start(created.run_id)
        second = _approve(service, first, "tester")
        final = _approve(service, second, "tester")
    finally:
        service.close()

    assert first.status == RunStatus.WAITING_FOR_APPROVAL.value
    assert first.pending_approval["rule"] == "django-check"
    assert second.status == RunStatus.WAITING_FOR_APPROVAL.value
    assert second.pending_approval["rule"] == "django-test"
    assert final.status == RunStatus.SUCCEEDED.value
    assert final.trace_path
    payload = json.loads(Path(final.trace_path).read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert len(payload["approvals"]) == 2


def test_workflow_resumes_after_service_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    first_service = _service(state_dir)
    first_service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = first_service.create_run(
        goal="Verify restart",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    waiting = first_service.start(created.run_id)
    first_service.runs.claim_pending(
        created.run_id,
        waiting.pending_approval["action_digest"],
    )
    first_service.close()
    assert waiting.status == RunStatus.WAITING_FOR_APPROVAL.value

    resumed_service = _service(state_dir)
    try:
        recovered = resumed_service.get(created.run_id)
        assert recovered.status == RunStatus.WAITING_FOR_APPROVAL.value
        resumed = _approve(resumed_service, recovered, "tester")
    finally:
        resumed_service.close()
    assert resumed.status == RunStatus.WAITING_FOR_APPROVAL.value
    assert resumed.pending_approval["rule"] == "django-test"


def test_pytorch_workflow_is_executable_and_checkpointed(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        service.trust_workspace(FIXTURES / "pytorch_project", trusted_by="researcher")
        created = service.create_run(
            goal="Smoke train and evaluate",
            workspace=FIXTURES / "pytorch_project",
            domain=Domain.PYTORCH,
        )
        first = service.start(created.run_id)
        second = _approve(service, first, "researcher")
        final = _approve(service, second, "researcher")
    finally:
        service.close()
    assert first.pending_approval["rule"] == "pytorch-smoke"
    assert second.pending_approval["rule"] == "pytorch-eval"
    assert final.status == RunStatus.SUCCEEDED.value


def test_android_workflow_uses_approval_gated_offline_wrapper(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        service.trust_workspace(FIXTURES / "android_project", trusted_by="android-engineer")
        created = service.create_run(
            goal="Verify Android project",
            workspace=FIXTURES / "android_project",
            domain=Domain.ANDROID,
        )
        record = service.start(created.run_id)
        rules: list[str] = []
        while record.status == RunStatus.WAITING_FOR_APPROVAL.value:
            rules.append(record.pending_approval["rule"])
            record = _approve(service, record, "android-engineer")
    finally:
        service.close()
    assert rules == ["gradle-tasks", "gradle-test", "gradle-lint"]
    assert record.status == RunStatus.SUCCEEDED.value


def test_ios_workflow_fails_closed_off_macos(tmp_path: Path) -> None:
    if platform.system() == "Darwin":
        return
    service = _service(tmp_path / "state")
    try:
        created = service.create_run(
            goal="Inspect iOS project",
            workspace=FIXTURES / "ios_project",
            domain=Domain.IOS,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        result = service.start(created.run_id)
    finally:
        service.close()
    assert result.status == RunStatus.FAILED.value
    assert result.error and "xcodebuild" in result.error


def test_advisory_mode_plans_without_executing(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        created = service.create_run(
            goal="Plan only",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        result = service.start(created.run_id)
    finally:
        service.close()
    assert result.status == RunStatus.SUCCEEDED.value
    assert result.pending_approval is None


def test_untrusted_workspace_refuses_execution(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        created = service.create_run(
            goal="Do not execute",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
        )
        try:
            service.start(created.run_id)
        except ValueError as exc:
            assert "not trusted" in str(exc)
        else:
            raise AssertionError("untrusted workspace execution was not refused")
    finally:
        service.close()


def test_state_directory_inside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="outside the workspace root"):
        AgentService(
            workspace_root=workspace,
            state_dir=workspace / ".state",
            approval_secret=SECRET,
        )


def test_concurrent_stale_approvals_cannot_advance_two_actions(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
        created = service.create_run(
            goal="Verify once",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
        )
        waiting = service.start(created.run_id)
        digest = waiting.pending_approval["action_digest"]

        def approve() -> str:
            try:
                service.approve_and_resume(
                    created.run_id,
                    approved_by="tester",
                    expected_action_digest=digest,
                )
                return "accepted"
            except ValueError:
                return "refused"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _index: approve(), range(2)))
        current = service.get(created.run_id)
    finally:
        service.close()
    assert sorted(outcomes) == ["accepted", "refused"]
    assert current.status == RunStatus.WAITING_FOR_APPROVAL.value
    assert current.pending_approval["rule"] == "django-test"


def test_cli_profile_outputs_json(capsys) -> None:
    code = main(["profile", str(FIXTURES / "django_project")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "django" in payload["domains"]


def test_cli_start_returns_waiting_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("INVERSE_AGENT_APPROVAL_SECRET", SECRET.decode())
    state_dir = tmp_path / "state"
    assert (
        main(
            [
                "trust-workspace",
                str(FIXTURES / "django_project"),
                "--trusted-by",
                "tester",
                "--workspace-root",
                str(FIXTURES),
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    code = main(
        [
            "start",
            str(FIXTURES / "django_project"),
            "--domain",
            "django",
            "--state-dir",
            str(state_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == RunStatus.WAITING_FOR_APPROVAL.value
