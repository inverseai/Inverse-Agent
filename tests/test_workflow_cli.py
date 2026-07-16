import argparse
import json
import platform
import shutil
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from inverse_agent.attestations import AttestationScope
from inverse_agent.cli import approve_command, main
from inverse_agent.models import AutonomyLevel, Domain, RunStatus
from inverse_agent.planner import DeterministicPlanner, ExecutionPlan, PlannedAction, Planner
from inverse_agent.runner import ApprovalChallenge
from inverse_agent.service import AgentService, RunRecord, RunStore
from inverse_agent.workflow import DurableAgentWorkflow

FIXTURES = Path(__file__).parent / "fixtures"
SECRET = b"test-workflow-secret-that-is-at-least-32-bytes"


def test_workbench_launcher_clears_omitted_calibration_pair() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "start-workbench.ps1").read_text(
        encoding="utf-8"
    )
    assert "$hasContextCalibration -ne $hasEstimatorCalibration" in script
    assert "$env:INVERSE_AGENT_MODEL_CONTEXT_TOKENS = $null" in script
    assert "$env:INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN = $null" in script
    assert "$env:INVERSE_AGENT_MODEL_REASONING_EFFORT = $null" in script


def _service(
    state_dir: Path,
    planner_fingerprint: str = "deterministic",
    planner: Planner | None = None,
) -> AgentService:
    return AgentService(
        workspace_root=FIXTURES,
        state_dir=state_dir,
        approval_secret=SECRET,
        planner=planner,
        planner_fingerprint=planner_fingerprint,
    )


def _approve(service: AgentService, record, approved_by: str):
    return service.approve_and_resume(
        record.run_id,
        approved_by=approved_by,
        expected_action_digest=record.pending_approval["action_digest"],
        expected_challenge_id=record.pending_approval["challenge_id"],
    )


def test_approve_command_returns_failure_for_failed_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = RunRecord(
        run_id="failed-run",
        goal="Inspect",
        workspace=str(FIXTURES),
        domain=Domain.GENERIC.value,
        autonomy_level=AutonomyLevel.ASSISTED.value,
        status=RunStatus.FAILED.value,
        pending_approval=None,
        trace_path=None,
        error="command failed",
        created_at=1.0,
        updated_at=2.0,
        planner_fingerprint="deterministic",
    )

    class FailedService:
        def approve_and_resume(self, *_args: object, **_kwargs: object) -> RunRecord:
            return record

        def close(self) -> None:
            return

    monkeypatch.setattr("inverse_agent.cli._service", lambda *_args, **_kwargs: FailedService())
    args = argparse.Namespace(
        workspace_root=str(FIXTURES),
        run_id=record.run_id,
        approved_by="tester",
        action_digest="digest",
        challenge_id="0" * 32,
    )

    assert approve_command(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == RunStatus.FAILED.value


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
    assert payload["planner_fingerprint"] == "deterministic"


def test_delayed_approval_cannot_authorize_an_identical_next_action(tmp_path: Path) -> None:
    class RepeatedActionPlanner:
        def plan(self, **_kwargs: object) -> ExecutionPlan:
            return ExecutionPlan(
                (PlannedAction("django.check"), PlannedAction("django.check")),
                "repeat one action to exercise challenge identity",
            )

    service = _service(tmp_path / "state", planner=RepeatedActionPlanner())
    try:
        service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
        created = service.create_run(
            goal="Run the same check twice",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
        )
        first = service.start(created.run_id)
        first_digest = first.pending_approval["action_digest"]
        first_challenge = first.pending_approval["challenge_id"]
        second = _approve(service, first, "tester")

        assert second.pending_approval["action_digest"] == first_digest
        assert second.pending_approval["challenge_id"] != first_challenge
        with pytest.raises(ValueError, match="stale or does not match"):
            service.approve_and_resume(
                created.run_id,
                approved_by="tester",
                expected_action_digest=first_digest,
                expected_challenge_id=first_challenge,
            )
        still_waiting = service.get(created.run_id)
        assert still_waiting.status == RunStatus.WAITING_FOR_APPROVAL.value
        final = _approve(service, still_waiting, "tester")
        assert final.status == RunStatus.SUCCEEDED.value
        assert final.completed_actions == 2
    finally:
        service.close()


def test_goal_is_redacted_before_persistence_and_trace_output(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        created = service.create_run(
            goal="Inspect token=super-secret-goal-value",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        completed = service.start(created.run_id)
    finally:
        service.close()

    assert "super-secret-goal-value" not in created.goal
    assert "[REDACTED_SECRET]" in created.goal
    assert completed.trace_path
    trace = Path(completed.trace_path).read_text(encoding="utf-8")
    assert "super-secret-goal-value" not in trace
    assert "[REDACTED_SECRET]" in trace


def test_run_refuses_planner_change_before_start(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    first = _service(state_dir, "openai-compatible|model-a|http://127.0.0.1:1234/v1|8")
    created = first.create_run(
        goal="Verify planner provenance",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    first.close()

    second = _service(state_dir, "openai-compatible|model-b|http://127.0.0.1:1234/v1|8")
    try:
        with pytest.raises(ValueError, match="planner configuration changed"):
            second.start(created.run_id)
    finally:
        second.close()


def test_waiting_run_resumes_without_replanning_after_config_change(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    first = _service(state_dir, "openai-compatible|model-a|http://127.0.0.1:1234/v1|8")
    first.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = first.create_run(
        goal="Verify durable model plan",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    waiting = first.start(created.run_id)
    first.close()

    class ReplanningFailsTest:
        def plan(self, **_kwargs):
            pytest.fail("resume invoked the planner")

    second = _service(
        state_dir,
        "openai-compatible|model-b|http://127.0.0.1:1234/v1|8",
        planner=ReplanningFailsTest(),
    )
    try:
        resumed = _approve(second, waiting, "tester")
    finally:
        second.close()
    assert resumed.status == RunStatus.WAITING_FOR_APPROVAL.value
    assert resumed.pending_approval and resumed.pending_approval["rule"] == "django-test"


def test_legacy_waiting_run_binds_current_execution_scope_generation(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    first = _service(state_dir)
    first.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = first.create_run(
        goal="Resume after scope migration",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    first.start(created.run_id)
    first.close()
    with sqlite3.connect(state_dir / "runs.sqlite") as connection:
        connection.execute(
            "UPDATE runs SET scope_generations=? WHERE run_id=?",
            ('{"__legacy_v01__":1}', created.run_id),
        )

    restarted = _service(state_dir)
    try:
        rebound = restarted.get(created.run_id)
        resumed = _approve(restarted, rebound, "tester")
    finally:
        restarted.close()

    assert rebound.scope_generations and "code_execution" in rebound.scope_generations
    assert resumed.status == RunStatus.WAITING_FOR_APPROVAL.value


def test_state_vector_versions_every_application_database(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    service = _service(state_dir)
    service.close()

    versions: dict[str, int] = {}
    for name in (
        "runs.sqlite",
        "workspace-trust.sqlite",
        "approval-replay.sqlite",
        "checkpoints.sqlite",
    ):
        with sqlite3.connect(state_dir / name) as connection:
            versions[name] = int(connection.execute("PRAGMA user_version").fetchone()[0])

    assert versions == {
        "runs.sqlite": 3,
        "workspace-trust.sqlite": 2,
        "approval-replay.sqlite": 1,
        "checkpoints.sqlite": 1,
    }


def test_run_store_migrates_planner_fingerprint_column(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, goal TEXT NOT NULL, workspace TEXT NOT NULL,
                domain TEXT NOT NULL, autonomy_level INTEGER NOT NULL, status TEXT NOT NULL,
                pending_approval TEXT, trace_path TEXT, error TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            )
            """
        )
    RunStore(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    assert "planner_fingerprint" in columns
    assert "plan" in columns
    assert "plan_rationale" in columns
    assert "completed_actions" in columns


def test_concurrent_double_start_invokes_workflow_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path / "state")
    service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = service.create_run(
        goal="Start only once",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    original_start = service.workflow.start
    counter = 0
    counter_lock = threading.Lock()

    def counted_start(spec):
        nonlocal counter
        with counter_lock:
            counter += 1
        time.sleep(0.1)
        return original_start(spec)

    monkeypatch.setattr(service.workflow, "start", counted_start)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            records = list(executor.map(lambda _index: service.start(created.run_id), range(2)))
        current = service.get(created.run_id)
    finally:
        service.close()

    assert counter == 1
    assert {record.status for record in records} <= {
        RunStatus.STARTING.value,
        RunStatus.WAITING_FOR_APPROVAL.value,
    }
    assert current.status == RunStatus.WAITING_FOR_APPROVAL.value


def test_expired_queued_approval_returns_to_a_fresh_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path / "state")
    service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    target = service.create_run(
        goal="Wait for an approval",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    waiting = service.start(target.run_id)
    old_challenge = waiting.pending_approval["challenge_id"]

    blocker_entered = threading.Event()
    release_blocker = threading.Event()
    original_start = service.workflow.start

    def blocking_start(spec):
        if spec.run_id == blocker.run_id:
            blocker_entered.set()
            if not release_blocker.wait(timeout=10):
                raise RuntimeError("test blocker was not released")
        return original_start(spec)

    blocker = service.create_run(
        goal="Occupy the worker",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
        autonomy_level=AutonomyLevel.ADVISORY,
    )
    monkeypatch.setattr(service.workflow, "start", blocking_start)
    try:
        service.start(blocker.run_id, wait=False)
        assert blocker_entered.wait(timeout=5)
        queued = service.approve_and_resume(
            target.run_id,
            approved_by="tester",
            expected_action_digest=waiting.pending_approval["action_digest"],
            expected_challenge_id=old_challenge,
            wait=False,
        )
        assert queued.status == RunStatus.QUEUED.value
        with sqlite3.connect(service.runs.path) as connection:
            connection.execute(
                "UPDATE run_work_items SET grant_expires_at=? WHERE run_id=? AND state='pending'",
                (time.time() - 1, target.run_id),
            )

        release_blocker.set()
        deadline = time.monotonic() + 5
        refreshed = service.get(target.run_id)
        while (
            refreshed.status != RunStatus.WAITING_FOR_APPROVAL.value and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            refreshed = service.get(target.run_id)
    finally:
        release_blocker.set()
        service.close()

    assert refreshed.status == RunStatus.WAITING_FOR_APPROVAL.value
    assert refreshed.pending_approval is not None
    assert refreshed.pending_approval["challenge_id"] != old_challenge
    assert refreshed.completed_actions == 0
    assert any(event.kind == "approval.refreshed" for event in service.runs.events(target.run_id))


def test_event_cursor_returns_only_newer_ordered_events(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    try:
        created = service.create_run(
            goal="Record ordered events",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        initial = service.events(created.run_id)
        completed = service.start(created.run_id)
        later = service.events(created.run_id, after=initial[-1].sequence)
    finally:
        service.close()

    assert completed.status == RunStatus.SUCCEEDED.value
    assert later
    assert all(event.sequence > initial[-1].sequence for event in later)
    assert [event.sequence for event in later] == sorted(event.sequence for event in later)


def test_revocation_cancels_active_runs_beyond_default_listing_page(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    workspace = FIXTURES / "django_project"
    grant = service.trust_workspace(workspace, trusted_by="tester")
    service._stop_worker.set()
    service._wake_worker.set()
    service._worker.join(timeout=5)
    created_ids: list[str] = []
    try:
        for index in range(501):
            record = service.create_run(
                goal=f"Queued run {index}",
                workspace=workspace,
                domain=Domain.DJANGO,
            )
            created_ids.append(record.run_id)
            service.runs.enqueue_start(
                record.run_id,
                scope_generations={"code_execution": grant["generation"]},
                endpoint_fingerprint=service.planner_fingerprint,
            )

        assert service.revoke_workspace(
            workspace,
            scope=AttestationScope.CODE_EXECUTION,
        )
    finally:
        service.close()

    assert all(
        service.runs.require(run_id).status == RunStatus.CANCELLED.value for run_id in created_ids
    )


def test_regrant_cancels_runs_bound_to_the_previous_generation(tmp_path: Path) -> None:
    service = _service(tmp_path / "state")
    service._stop_worker.set()
    service._wake_worker.set()
    service._worker.join(timeout=5)
    try:
        first_grant = service.trust_workspace(
            FIXTURES / "django_project",
            trusted_by="first-operator",
        )
        created = service.create_run(
            goal="Queued under the first consent generation",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
        )
        queued = service.start(created.run_id, wait=False)
        second_grant = service.trust_workspace(
            FIXTURES / "django_project",
            trusted_by="second-operator",
        )
        cancelled = service.get(created.run_id)
    finally:
        service.close()

    assert queued.status == RunStatus.QUEUED.value
    assert second_grant["generation"] > first_grant["generation"]
    assert cancelled.status == RunStatus.CANCELLED.value


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
        waiting.pending_approval["challenge_id"],
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


def test_restart_requeues_claimed_start_before_first_checkpoint(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    first = _service(state_dir)
    first.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    first._stop_worker.set()
    first._wake_worker.set()
    first._worker.join(timeout=5)
    created = first.create_run(
        goal="Recover a claimed start",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    generations = first.trust.capture_generations(
        FIXTURES / "django_project",
        (AttestationScope.CODE_EXECUTION,),
    )
    first.runs.enqueue_start(
        created.run_id,
        scope_generations=generations,
        endpoint_fingerprint=first.planner_fingerprint,
    )
    item = first.runs.claim_next_work()
    assert item is not None
    first.runs.set_running(created.run_id, expected_status=RunStatus.STARTING)
    first.runs.mark_work_started(item.work_id)
    first.close()

    restarted = _service(state_dir)
    try:
        deadline = time.monotonic() + 5
        recovered = restarted.get(created.run_id)
        while (
            recovered.status
            not in {
                RunStatus.WAITING_FOR_APPROVAL.value,
                RunStatus.FAILED.value,
            }
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            recovered = restarted.get(created.run_id)
    finally:
        restarted.close()

    assert recovered.status == RunStatus.WAITING_FOR_APPROVAL.value


def test_legacy_checkpoint_without_challenge_id_resumes_after_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"

    def legacy_payload(
        challenge: ApprovalChallenge,
        _action_ordinal: int,
    ) -> dict[str, object]:
        return {
            "action_digest": challenge.action_digest,
            "rule": challenge.rule,
            "argv": challenge.argv,
            "workspace": challenge.workspace,
            "domain": challenge.domain,
            "reason": challenge.reason,
        }

    with monkeypatch.context() as patch:
        patch.setattr(
            DurableAgentWorkflow,
            "_approval_interrupt_payload",
            staticmethod(legacy_payload),
        )
        legacy_service = _service(state_dir)
        try:
            legacy_service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
            created = legacy_service.create_run(
                goal="Resume a pre-challenge-id checkpoint",
                workspace=FIXTURES / "django_project",
                domain=Domain.DJANGO,
            )
            waiting = legacy_service.start(created.run_id)
            checkpoint_pending = legacy_service.workflow.current(created.run_id).pending_approval
            assert checkpoint_pending is not None
            assert "challenge_id" not in checkpoint_pending
            assert waiting.pending_approval is not None
            projected_challenge_id = waiting.pending_approval["challenge_id"]
        finally:
            legacy_service.close()

    upgraded_service = _service(state_dir)
    try:
        recovered = upgraded_service.get(created.run_id)
        assert recovered.pending_approval is not None
        assert recovered.pending_approval["challenge_id"] == projected_challenge_id
        resumed = _approve(upgraded_service, recovered, "tester")
    finally:
        upgraded_service.close()

    assert resumed.status == RunStatus.WAITING_FOR_APPROVAL.value
    assert resumed.pending_approval is not None
    assert resumed.pending_approval["rule"] == "django-test"


def test_restart_reconciles_incomplete_run_older_than_default_page(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    first_service = _service(state_dir)
    oldest = first_service.create_run(
        goal="Recover this old run",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
        autonomy_level=AutonomyLevel.ADVISORY,
    )
    for index in range(100):
        first_service.create_run(
            goal=f"Newer run {index}",
            workspace=FIXTURES / "django_project",
            domain=Domain.DJANGO,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
    first_service.close()
    with sqlite3.connect(state_dir / "runs.sqlite") as connection:
        connection.execute(
            "UPDATE runs SET status=?, created_at=0 WHERE run_id=?",
            (RunStatus.STARTING.value, oldest.run_id),
        )

    restarted = _service(state_dir)
    try:
        recovered = restarted.get(oldest.run_id)
    finally:
        restarted.close()

    assert recovered.status == RunStatus.FAILED.value
    assert recovered.error and "workflow recovery failed" in recovered.error


def test_restart_fails_mid_graph_checkpoint_without_replaying_work(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    planner_entered = threading.Event()
    release_planner = threading.Event()
    delegate = DeterministicPlanner()

    class BlockingPlanner:
        def plan(self, **kwargs):
            planner_entered.set()
            if not release_planner.wait(timeout=10):
                raise RuntimeError("test planner was not released")
            return delegate.plan(**kwargs)

    first_service = _service(state_dir, planner=BlockingPlanner())
    first_service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = first_service.create_run(
        goal="Pause during planning",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(first_service.start, created.run_id)
    restarted = None
    try:
        assert planner_entered.wait(timeout=5)
        first_service._state_lease.close()
        restarted = _service(state_dir)
        recovered = restarted.get(created.run_id)
        assert recovered.status == RunStatus.FAILED.value
        assert recovered.error and "outcome may be unknown" in recovered.error
    finally:
        if restarted is not None:
            restarted.close()
        release_planner.set()
        stale_result = future.result(timeout=10)
        final_record = first_service.get(created.run_id)
        executor.shutdown()
        first_service.close()

    assert stale_result.status == RunStatus.FAILED.value
    assert final_record.status == RunStatus.FAILED.value
    assert final_record.error and "outcome may be unknown" in final_record.error


def test_state_directory_refuses_concurrent_service_writer(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    first_service = _service(state_dir)
    try:
        with pytest.raises(RuntimeError, match="state directory is already in use"):
            _service(state_dir)
    finally:
        first_service.close()

    replacement = _service(state_dir)
    replacement.close()


def test_missing_checkpoint_database_fails_mixed_vector_closed(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    first_service = _service(state_dir)
    first_service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = first_service.create_run(
        goal="Verify recovery failure",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    waiting = first_service.start(created.run_id)
    first_service.close()
    assert waiting.status == RunStatus.WAITING_FOR_APPROVAL.value
    (state_dir / "checkpoints.sqlite").unlink()

    with pytest.raises(RuntimeError, match="mixed schema-version vector"):
        _service(state_dir)


def test_malformed_checkpoint_marks_only_affected_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    first_service = _service(state_dir)
    first_service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = first_service.create_run(
        goal="Verify malformed recovery",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    first_service.start(created.run_id)
    first_service.close()

    def malformed_checkpoint(*_args, **_kwargs):
        raise TypeError("malformed checkpoint value")

    monkeypatch.setattr(
        "inverse_agent.workflow.DurableAgentWorkflow.current",
        malformed_checkpoint,
    )
    restarted = _service(state_dir)
    try:
        recovered = restarted.get(created.run_id)
    finally:
        restarted.close()
    assert recovered.status == RunStatus.FAILED.value
    assert recovered.error and "malformed checkpoint value" in recovered.error


def test_recovery_projection_failure_marks_only_affected_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    first_service = _service(state_dir)
    first_service.trust_workspace(FIXTURES / "django_project", trusted_by="tester")
    created = first_service.create_run(
        goal="Verify projection recovery",
        workspace=FIXTURES / "django_project",
        domain=Domain.DJANGO,
    )
    first_service.start(created.run_id)
    first_service.close()

    def failed_projection(*_args, **_kwargs):
        raise TypeError("recovery projection failed")

    monkeypatch.setattr(
        "inverse_agent.service.RunStore.update_from_result",
        failed_projection,
    )
    restarted = _service(state_dir)
    try:
        recovered = restarted.get(created.run_id)
    finally:
        restarted.close()
    assert recovered.status == RunStatus.FAILED.value
    assert recovered.error and "recovery projection failed" in recovered.error


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


def test_generic_git_workflow_requires_approval_for_each_inspection(tmp_path: Path) -> None:
    git = shutil.which("git")
    if not git:
        pytest.skip("Git is unavailable")
    repository = tmp_path / "workspace"
    repository.mkdir()
    (repository / "README.md").write_text("# Generic fixture\n", encoding="utf-8")
    subprocess.run(
        [git, "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [git, "-C", str(repository), "add", "README.md"],
        check=True,
        capture_output=True,
    )
    service = AgentService(
        workspace_root=repository,
        state_dir=tmp_path / "state",
        approval_secret=SECRET,
    )
    try:
        service.trust_workspace(repository, trusted_by="tester")
        created = service.create_run(
            goal="Inspect repository status and tracked files",
            workspace=repository,
            domain=Domain.GENERIC,
        )
        first = service.start(created.run_id)
        second = _approve(service, first, "tester")
        result = _approve(service, second, "tester")
    finally:
        service.close()

    assert result.status == RunStatus.SUCCEEDED.value
    assert result.plan == ("generic.status", "generic.tracked_files")
    assert result.completed_actions == 2
    assert result.pending_approval is None
    assert first.pending_approval and first.pending_approval["rule"] == "git-status"
    assert second.pending_approval and second.pending_approval["rule"] == "git-ls-files"


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
        challenge_id = waiting.pending_approval["challenge_id"]

        def approve() -> str:
            try:
                service.approve_and_resume(
                    created.run_id,
                    approved_by="tester",
                    expected_action_digest=digest,
                    expected_challenge_id=challenge_id,
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


def test_scope_revocation_linearizes_against_approved_command_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path / "state")
    workspace = FIXTURES / "django_project"
    service.trust_workspace(workspace, trusted_by="tester")
    created = service.create_run(
        goal="Race revocation with dispatch",
        workspace=workspace,
        domain=Domain.DJANGO,
    )
    waiting = service.start(created.run_id)
    dispatch_boundary = threading.Event()
    release_dispatch = threading.Event()
    original_invoke = service._invoke_verification_workflow

    def blocked_invoke(running, item, *, grant_expires_at):
        dispatch_boundary.set()
        if not release_dispatch.wait(timeout=10):
            raise RuntimeError("test dispatch was not released")
        return original_invoke(
            running,
            item,
            grant_expires_at=grant_expires_at,
        )

    monkeypatch.setattr(service, "_invoke_verification_workflow", blocked_invoke)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        service.approve_and_resume(
            created.run_id,
            approved_by="tester",
            expected_action_digest=waiting.pending_approval["action_digest"],
            expected_challenge_id=waiting.pending_approval["challenge_id"],
            wait=False,
        )
        assert dispatch_boundary.wait(timeout=5)
        revocation = executor.submit(
            service.revoke_workspace,
            workspace,
            scope=AttestationScope.CODE_EXECUTION,
        )
        time.sleep(0.1)
        assert not revocation.done()
        release_dispatch.set()
        assert revocation.result(timeout=10)
        current = service.get(created.run_id)
    finally:
        release_dispatch.set()
        executor.shutdown()
        service.close()

    assert current.status == RunStatus.CANCELLED.value


def test_cli_profile_outputs_json(capsys) -> None:
    code = main(["profile", str(FIXTURES / "django_project")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "django" in payload["domains"]


def test_cli_start_returns_waiting_status(tmp_path: Path, monkeypatch, capsys) -> None:
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

    assert (
        main(
            [
                "list-runs",
                "--workspace-root",
                str(FIXTURES),
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )
    listed = json.loads(capsys.readouterr().out)
    assert [record["run_id"] for record in listed] == [payload["run_id"]]

    assert (
        main(
            [
                "get-plan",
                payload["run_id"],
                "--workspace-root",
                str(FIXTURES),
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["plan"] == ["django.check", "django.test"]

    assert (
        main(
            [
                "events",
                payload["run_id"],
                "--workspace-root",
                str(FIXTURES),
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )
    events = json.loads(capsys.readouterr().out)
    assert events["events"]
    assert events["next_cursor"] == events["events"][-1]["sequence"]

    assert (
        main(
            [
                "decline",
                payload["run_id"],
                "--declined-by",
                "tester",
                "--action-digest",
                payload["pending_approval"]["action_digest"],
                "--challenge-id",
                payload["pending_approval"]["challenge_id"],
                "--workspace-root",
                str(FIXTURES),
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )
    declined = json.loads(capsys.readouterr().out)
    assert declined["status"] == RunStatus.REFUSED.value


def test_cli_start_allows_state_directory_beside_workspace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("INVERSE_AGENT_APPROVAL_SECRET", SECRET.decode())
    workspace = tmp_path / "django_project"
    state_dir = tmp_path / "state"
    shutil.copytree(FIXTURES / "django_project", workspace)
    assert (
        main(
            [
                "trust-workspace",
                str(workspace),
                "--trusted-by",
                "tester",
                "--workspace-root",
                str(workspace),
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
            str(workspace),
            "--domain",
            "django",
            "--state-dir",
            str(state_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == RunStatus.WAITING_FOR_APPROVAL.value


def test_cli_scoped_trust_revoke_cancel_and_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("INVERSE_AGENT_APPROVAL_SECRET", SECRET.decode())
    state_dir = tmp_path / "state"
    workspace = FIXTURES / "django_project"
    common = [
        "--workspace-root",
        str(FIXTURES),
        "--state-dir",
        str(state_dir),
    ]
    assert (
        main(
            [
                "trust-workspace",
                str(workspace),
                "--trusted-by",
                "tester",
                "--scope",
                AttestationScope.SOURCE_READ.value,
                *common,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["scope"] == "source_read"
    assert (
        main(
            [
                "revoke-workspace",
                str(workspace),
                "--scope",
                AttestationScope.SOURCE_READ.value,
                *common,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["revoked"] is True

    service = _service(state_dir)
    try:
        planned = service.create_run(
            goal="Cancel before start",
            workspace=workspace,
            domain=Domain.DJANGO,
        )
        completed = service.create_run(
            goal="Create a trace",
            workspace=workspace,
            domain=Domain.DJANGO,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        completed = service.start(completed.run_id)
        assert completed.status == RunStatus.SUCCEEDED.value
    finally:
        service.close()

    assert main(["cancel", planned.run_id, *common]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == RunStatus.CANCELLED.value
    assert main(["get-trace", completed.run_id, *common]) == 0
    trace = json.loads(capsys.readouterr().out)
    assert trace["run_id"] == completed.run_id
    assert trace["actions"] == []


def test_direct_cli_no_wait_is_refused_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="running control plane"):
        main(
            [
                "start",
                str(FIXTURES / "django_project"),
                "--domain",
                "django",
                "--state-dir",
                str(tmp_path / "state"),
                "--no-wait",
            ]
        )
