"""Production service integration for queued read-only investigation runs."""

from __future__ import annotations

import hmac
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from inverse_agent.attestations import AttestationScope
from inverse_agent.fs_tools import WorkspaceReader
from inverse_agent.investigation import (
    AgentAnswer,
    Decision,
    InvestigationLoop,
    ModelCallRecord,
    ScriptedInvestigationPlanner,
    SourceCitation,
    ToolCall,
    ToolObservation,
)
from inverse_agent.investigation_model import ModelInvestigationPlanner
from inverse_agent.models import AutonomyLevel, Domain, RunKind, RunStatus
from inverse_agent.planner import PlannerTransportError
from inverse_agent.service import AgentService, RunRecord

SECRET = b"test-investigation-service-secret-at-least-32-bytes"


def _answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
    observation = catalog[-1]
    return AgentAnswer(
        summary="app.py returns 42",
        findings=("The return statement yields 42.",),
        next_actions=("Keep the implementation.",),
        citations=(
            SourceCitation(
                observation_id=observation.observation_id,
                path=observation.path,
                start_line=2,
                end_line=2,
            ),
        ),
    )


def test_service_executes_investigation_and_persists_events(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "project"
    workspace.mkdir(parents=True)
    (workspace / "app.py").write_text("def value():\n    return 42\n", encoding="utf-8")

    def planner_factory(_record: RunRecord) -> ScriptedInvestigationPlanner:
        return ScriptedInvestigationPlanner(
            steps=(ToolCall(tool="read_file", path="app.py"),),
            build_answer=_answer,
        )

    service = AgentService(
        workspace_root=workspace_root,
        state_dir=tmp_path / "state",
        approval_secret=SECRET,
        planner_fingerprint="local-model|127.0.0.1|test",
        investigation_planner_factory=planner_factory,
    )
    try:
        service.trust_workspace(
            workspace,
            trusted_by="tester",
            scope=AttestationScope.SOURCE_READ,
        )
        created = service.create_run(
            goal="What does app.py return?",
            workspace=workspace,
            domain=Domain.GENERIC,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        completed = service.start(created.run_id)
        events = service.events(created.run_id)
    finally:
        service.close()

    assert completed.status == RunStatus.SUCCEEDED.value
    assert completed.answer is not None
    assert completed.answer["summary"] == "app.py returns 42"
    assert completed.usage and completed.usage["decisions_used"] == 2
    observation = next(event for event in events if event.kind == "investigation.observation")
    assert observation.payload["observation"]["path"] == "app.py"
    assert events[-1].kind == "investigation.finished"


def test_source_revocation_cancels_active_investigation_at_boundary(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "project"
    workspace.mkdir(parents=True)
    (workspace / "app.py").write_text("value = 42\n", encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()

    class BlockingPlanner:
        def decide(
            self,
            *,
            goal: str,
            catalog: tuple[ToolObservation, ...],
        ) -> Decision:
            del goal, catalog
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test planner was not released")
            return ToolCall(tool="read_file", path="app.py")

    service = AgentService(
        workspace_root=workspace_root,
        state_dir=tmp_path / "state",
        approval_secret=SECRET,
        planner_fingerprint="local-model|127.0.0.1|test",
        investigation_planner_factory=lambda _record: BlockingPlanner(),
    )
    try:
        service.trust_workspace(
            workspace,
            trusted_by="tester",
            scope=AttestationScope.SOURCE_READ,
        )
        created = service.create_run(
            goal="Read app.py",
            workspace=workspace,
            domain=Domain.GENERIC,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        service.start(created.run_id, wait=False)
        assert entered.wait(timeout=5)
        assert service.revoke_workspace(workspace, scope=AttestationScope.SOURCE_READ)
        release.set()

        deadline = time.monotonic() + 5
        cancelled = service.get(created.run_id)
        while cancelled.status != RunStatus.CANCELLED.value and time.monotonic() < deadline:
            time.sleep(0.01)
            cancelled = service.get(created.run_id)
    finally:
        release.set()
        service.close()

    assert cancelled.status == RunStatus.CANCELLED.value
    assert cancelled.stop_reason == "cancelled"


def test_restart_resumes_durable_answer_without_rereading_or_replanning(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "project"
    workspace.mkdir(parents=True)
    (workspace / "app.py").write_text("def value():\n    return 42\n", encoding="utf-8")
    state_dir = tmp_path / "state"

    class NeverPlanner:
        def decide(
            self,
            *,
            goal: str,
            catalog: tuple[ToolObservation, ...],
        ) -> Decision:
            del goal, catalog
            raise AssertionError("a durable pending answer must not be replanned")

    first = AgentService(
        workspace_root=workspace_root,
        state_dir=state_dir,
        approval_secret=SECRET,
        planner_fingerprint="local-model|127.0.0.1|test",
        investigation_planner_factory=lambda _record: NeverPlanner(),
    )
    try:
        first.trust_workspace(
            workspace,
            trusted_by="tester",
            scope=AttestationScope.SOURCE_READ,
        )
        created = first.create_run(
            goal="What does app.py return?",
            workspace=workspace,
            domain=Domain.GENERIC,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        first._stop_worker.set()
        first._wake_worker.set()
        first._worker.join(timeout=5)
        first.start(created.run_id, wait=False)
        item = first.runs.claim_next_work()
        assert item is not None
        running = first.runs.set_running(
            created.run_id,
            expected_status=RunStatus.STARTING,
        )
        assert running.status == RunStatus.RUNNING.value
        first.runs.mark_work_started(item.work_id)

        identity_key = hmac.digest(
            first._evidence_identity_root,
            created.run_id.encode("utf-8"),
            "sha256",
        )
        reader = WorkspaceReader.open(workspace, identity_key=identity_key)
        call = ToolCall(tool="read_file", path="app.py")
        observation = reader.read_file("app.py")
        evidence_identity = reader.evidence_identity(observation.observation_id)
        assert evidence_identity is not None
        usage = {
            "decisions_used": 1,
            "tool_calls_used": 1,
            "physical_requests_used": 1,
            "command_calls_used": 0,
            "completion_tokens_used": 0,
            "completion_tokens_charged": 0,
            "completion_tokens_requested": 0,
            "observation_bytes_used": len(observation.text.encode("utf-8")),
            "active_seconds": 0.25,
            "transport_retries": 0,
            "schema_retries": 0,
        }
        first.runs.update_investigation_progress(
            created.run_id,
            usage=usage,
            event_kind="investigation.decision",
            event_payload={
                "decisions_used": 1,
                "physical_requests_used": 1,
                "decision_kind": "tool",
                "decision": asdict(call),
                "model_calls": [],
            },
        )
        first.runs.update_investigation_progress(
            created.run_id,
            usage=usage,
            event_kind="investigation.observation",
            event_payload={
                "observation": asdict(observation),
                "evidence_identity": evidence_identity,
                "model_calls": [],
                **usage,
            },
        )

        class AnswerWithLedger:
            requests_made = 0
            completion_tokens_requested = 0
            completion_tokens_charged = 0
            completion_tokens_reported = 0
            transport_retries = 0
            schema_retries = 0
            model_calls: list[ModelCallRecord] = []

            def decide(
                self,
                *,
                goal: str,
                catalog: tuple[ToolObservation, ...],
            ) -> Decision:
                del goal
                self.requests_made = 2
                self.completion_tokens_requested = 4096
                self.completion_tokens_charged = 3584
                self.completion_tokens_reported = 1536
                self.transport_retries = 1
                self.model_calls = [
                    ModelCallRecord(
                        request_index=1,
                        logical_decision=1,
                        requested_completion_tokens=2048,
                        charged_completion_tokens=2048,
                        reported_prompt_tokens=None,
                        reported_completion_tokens=None,
                        reported_model=None,
                        latency_seconds=0.1,
                        outcome="transport_error",
                    ),
                    ModelCallRecord(
                        request_index=2,
                        logical_decision=1,
                        requested_completion_tokens=2048,
                        charged_completion_tokens=1536,
                        reported_prompt_tokens=256,
                        reported_completion_tokens=1536,
                        reported_model="test-model",
                        latency_seconds=0.2,
                        outcome="success",
                    ),
                ]
                return _answer(catalog)

        def persist_decision_then_crash(kind: str, payload: dict[str, object]) -> None:
            assert kind == "investigation.decision"
            progress = dict(usage)
            for name, value in payload.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    progress[name] = value
            first.runs.update_investigation_progress(
                created.run_id,
                usage=progress,
                event_kind=kind,
                event_payload=payload,
            )
            raise RuntimeError("simulated crash after durable decision")

        with pytest.raises(RuntimeError, match="simulated crash"):
            InvestigationLoop(
                planner=AnswerWithLedger(),
                trust=first.trust,
                event_sink=persist_decision_then_crash,
                evidence_identity_key=identity_key,
            ).run(
                run_id=created.run_id,
                goal=created.goal,
                workspace=workspace,
                initial_catalog=(observation,),
                prior_usage=usage,
                initial_evidence_identities={observation.observation_id: evidence_identity},
                prior_tool_calls=(call,),
            )
    finally:
        first.close()

    second = AgentService(
        workspace_root=workspace_root,
        state_dir=state_dir,
        approval_secret=SECRET,
        planner_fingerprint="local-model|127.0.0.1|test",
        investigation_planner_factory=lambda _record: NeverPlanner(),
    )
    try:
        deadline = time.monotonic() + 5
        completed = second.get(created.run_id)
        while completed.status != RunStatus.SUCCEEDED.value and time.monotonic() < deadline:
            time.sleep(0.01)
            completed = second.get(created.run_id)
        events = second.events(created.run_id)
    finally:
        second.close()

    assert completed.status == RunStatus.SUCCEEDED.value
    assert completed.answer and completed.answer["summary"] == "app.py returns 42"
    assert completed.usage and completed.usage["decisions_used"] == 2
    assert completed.usage["tool_calls_used"] == 1
    assert completed.usage["physical_requests_used"] == 3
    assert completed.usage["completion_tokens_charged"] == 3584
    assert completed.usage["active_seconds"] >= 0.25
    assert completed.usage["transport_retries"] == 1
    assert sum(event.kind == "investigation.observation" for event in events) == 1
    result = next(event for event in events if event.kind == "investigation.result")
    assert [call["request_index"] for call in result.payload["model_calls"]] == [2, 3]


def test_restart_finalizes_a_durable_result_before_requeuing(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "project"
    workspace.mkdir(parents=True)
    state_dir = tmp_path / "state"

    class NeverPlanner:
        def decide(
            self,
            *,
            goal: str,
            catalog: tuple[ToolObservation, ...],
        ) -> Decision:
            del goal, catalog
            raise AssertionError("a durable result must not be replanned")

    first = AgentService(
        workspace_root=workspace_root,
        state_dir=state_dir,
        approval_secret=SECRET,
        investigation_planner_factory=lambda _record: NeverPlanner(),
    )
    try:
        first.trust_workspace(
            workspace,
            trusted_by="tester",
            scope=AttestationScope.SOURCE_READ,
        )
        created = first.create_run(
            goal="Inspect the project",
            workspace=workspace,
            domain=Domain.GENERIC,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        first._stop_worker.set()
        first._wake_worker.set()
        first._worker.join(timeout=5)
        first.start(created.run_id, wait=False)
        item = first.runs.claim_next_work()
        assert item is not None
        first.runs.set_running(created.run_id, expected_status=RunStatus.STARTING)
        first.runs.mark_work_started(item.work_id)
        usage: dict[str, int | float] = {
            "decisions_used": 0,
            "tool_calls_used": 0,
            "physical_requests_used": 0,
            "command_calls_used": 0,
            "completion_tokens_used": 0,
            "completion_tokens_charged": 0,
            "completion_tokens_requested": 0,
            "observation_bytes_used": 0,
            "active_seconds": 600.0,
            "transport_retries": 0,
            "schema_retries": 0,
        }
        first.runs.update_investigation_progress(
            created.run_id,
            usage=usage,
            event_kind="investigation.result",
            event_payload={
                "verdict": "incomplete",
                "stop_reason": "budget_exhausted",
                "answer": None,
                "error": "active-time budget exhausted",
                "model_calls": [],
                **usage,
            },
        )
    finally:
        first.close()

    second = AgentService(
        workspace_root=workspace_root,
        state_dir=state_dir,
        approval_secret=SECRET,
        investigation_planner_factory=lambda _record: NeverPlanner(),
    )
    try:
        completed = second.get(created.run_id)
        events = second.events(created.run_id)
        pending_work = second.runs.pending_work_item(created.run_id)
    finally:
        second.close()

    assert completed.status == RunStatus.INCOMPLETE.value
    assert completed.stop_reason == "budget_exhausted"
    assert completed.error == "active-time budget exhausted"
    assert pending_work is None
    assert events[-1].kind == "investigation.finished"


def test_restart_conservatively_charges_an_in_flight_model_request(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "project"
    workspace.mkdir(parents=True)
    state_dir = tmp_path / "state"

    class SimulatedProcessCrash(BaseException):
        pass

    class CrashingClient:
        last_response_metadata = None
        calls = 0

        def complete_structured_json(self, **_kwargs: object) -> dict[str, object]:
            self.calls += 1
            raise SimulatedProcessCrash

    first = AgentService(
        workspace_root=workspace_root,
        state_dir=state_dir,
        approval_secret=SECRET,
        investigation_planner_factory=lambda _record: ModelInvestigationPlanner(
            client=CrashingClient()
        ),
    )
    try:
        first.trust_workspace(
            workspace,
            trusted_by="tester",
            scope=AttestationScope.SOURCE_READ,
        )
        created = first.create_run(
            goal="Inspect the project",
            workspace=workspace,
            domain=Domain.GENERIC,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        first._stop_worker.set()
        first._wake_worker.set()
        first._worker.join(timeout=5)
        first.start(created.run_id, wait=False)
        item = first.runs.claim_next_work()
        assert item is not None
        with pytest.raises(SimulatedProcessCrash):
            first._execute_work(item)
        started = [
            event
            for event in first.events(created.run_id)
            if event.kind == "investigation.model_request_started"
        ][-1]
        time.sleep(0.02)
    finally:
        first.close()

    class FailingClient:
        last_response_metadata = None
        calls = 0

        def complete_structured_json(self, **_kwargs: object) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                raise PlannerTransportError("simulated transport failure")
            return {}

    second = AgentService(
        workspace_root=workspace_root,
        state_dir=state_dir,
        approval_secret=SECRET,
        investigation_planner_factory=lambda _record: ModelInvestigationPlanner(
            client=FailingClient()
        ),
    )
    try:
        deadline = time.monotonic() + 5
        completed = second.get(created.run_id)
        while completed.status != RunStatus.FAILED.value and time.monotonic() < deadline:
            time.sleep(0.01)
            completed = second.get(created.run_id)
        events = second.events(created.run_id)
    finally:
        second.close()

    abandoned = next(
        event for event in events if event.kind == "investigation.model_request_abandoned"
    )
    result = next(event for event in events if event.kind == "investigation.result")
    assert completed.status == RunStatus.FAILED.value
    assert completed.usage is not None
    assert completed.usage["physical_requests_used"] == 3
    assert completed.usage["transport_retries"] == 1
    assert completed.usage["schema_retries"] == 0
    assert (
        completed.usage["completion_tokens_charged"] >= started.payload["completion_tokens_charged"]
    )
    assert abandoned.payload["active_seconds"] > started.payload["active_seconds"]
    assert [call["request_index"] for call in result.payload["model_calls"]] == [1, 2, 3]
    assert [call["outcome"] for call in result.payload["model_calls"]] == [
        "process_interrupted",
        "transport_error",
        "schema_error",
    ]


def test_durable_compaction_state_rehydrates_without_orphaning_catalog(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "project"
    workspace.mkdir(parents=True)
    state_dir = tmp_path / "state"
    service = AgentService(
        workspace_root=workspace_root,
        state_dir=state_dir,
        approval_secret=SECRET,
        investigation_planner_factory=lambda _record: ScriptedInvestigationPlanner(),
    )
    try:
        service.trust_workspace(
            workspace,
            trusted_by="tester",
            scope=AttestationScope.SOURCE_READ,
        )
        created = service.create_run(
            goal="Inspect durable history",
            workspace=workspace,
            domain=Domain.GENERIC,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        service._stop_worker.set()
        service._wake_worker.set()
        service._worker.join(timeout=5)
        service.start(created.run_id, wait=False)
        item = service.runs.claim_next_work()
        assert item is not None
        service.runs.set_running(created.run_id, expected_status=RunStatus.STARTING)
        service.runs.mark_work_started(item.work_id)
        observation = ToolObservation(
            observation_id="obs_durable_history",
            tool="read_file",
            path="app.py",
            content_hash="a" * 64,
            text="value = 42",
            lines=("1: value = 42",),
        )
        usage: dict[str, int | float] = {
            "decisions_used": 0,
            "tool_calls_used": 0,
            "physical_requests_used": 1,
            "command_calls_used": 0,
            "completion_tokens_used": 12,
            "completion_tokens_charged": 12,
            "completion_tokens_requested": 1024,
            "observation_bytes_used": len(observation.text),
            "active_seconds": 0.1,
            "transport_retries": 0,
            "schema_retries": 0,
        }
        call = ModelCallRecord(
            request_index=1,
            logical_decision=1,
            requested_completion_tokens=1024,
            charged_completion_tokens=12,
            reported_prompt_tokens=500,
            reported_completion_tokens=12,
            reported_model="local-model",
            latency_seconds=0.1,
            outcome="success",
            request_kind="compaction",
        )
        service.runs.update_investigation_progress(
            created.run_id,
            usage=usage,
            event_kind="investigation.observation",
            event_payload={"observation": asdict(observation), **usage, "model_calls": []},
        )
        service.runs.update_investigation_progress(
            created.run_id,
            usage=usage,
            event_kind="investigation.compaction",
            event_payload={
                **usage,
                "pinned_notes": "Non-authoritative durable notes.",
                "compacted_observation_ids": [observation.observation_id],
                "model_calls": [asdict(call)],
            },
        )

        recovery = service._investigation_recovery(created.run_id)
    finally:
        service.close()

    assert recovery.catalog == (observation,)
    assert recovery.compaction_notes == "Non-authoritative durable notes."
    assert recovery.compacted_observation_ids == (observation.observation_id,)
    assert recovery.model_calls == (call,)
    assert recovery.resume_request_kind == "decision"


def test_interrupted_compaction_is_precharged_and_resumes_as_compaction(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "project"
    workspace.mkdir(parents=True)
    service = AgentService(
        workspace_root=workspace_root,
        state_dir=tmp_path / "state",
        approval_secret=SECRET,
        investigation_planner_factory=lambda _record: ScriptedInvestigationPlanner(),
    )

    class SimulatedProcessCrash(BaseException):
        pass

    class CrashingCompactionClient:
        last_response_metadata = None

        def complete_structured_json(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["schema_name"] == "investigation_compaction"
            raise SimulatedProcessCrash

    try:
        service.trust_workspace(
            workspace,
            trusted_by="tester",
            scope=AttestationScope.SOURCE_READ,
        )
        created = service.create_run(
            goal="Compact durable history",
            workspace=workspace,
            domain=Domain.GENERIC,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        service._stop_worker.set()
        service._wake_worker.set()
        service._worker.join(timeout=5)
        service.start(created.run_id, wait=False)
        item = service.runs.claim_next_work()
        assert item is not None
        service.runs.set_running(created.run_id, expected_status=RunStatus.STARTING)
        service.runs.mark_work_started(item.work_id)
        catalog = tuple(
            ToolObservation(
                observation_id=f"obs_precompact_{index}",
                tool="read_file",
                path=f"history_{index}.py",
                content_hash=str(index) * 64,
                text="source",
                lines=tuple(f"{line}: " + "x" * 80 for line in range(1, 101)),
            )
            for index in range(3)
        )
        baseline: dict[str, int | float] = {
            "decisions_used": 0,
            "tool_calls_used": 0,
            "physical_requests_used": 0,
            "command_calls_used": 0,
            "completion_tokens_used": 0,
            "completion_tokens_charged": 0,
            "completion_tokens_requested": 0,
            "observation_bytes_used": sum(len(item.text) for item in catalog),
            "active_seconds": 0.0,
            "transport_retries": 0,
            "schema_retries": 0,
        }
        for observation in catalog:
            service.runs.update_investigation_progress(
                created.run_id,
                usage=baseline,
                event_kind="investigation.observation",
                event_payload={
                    "observation": asdict(observation),
                    **baseline,
                    "model_calls": [],
                },
            )

        def persist(kind: str, payload: dict[str, object]) -> None:
            usage = dict(baseline)
            for name, value in payload.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    usage[name] = value
            service.runs.update_investigation_progress(
                created.run_id,
                usage=usage,
                event_kind=kind,
                event_payload=payload,
            )

        with pytest.raises(SimulatedProcessCrash):
            InvestigationLoop(
                planner=ModelInvestigationPlanner(
                    client=CrashingCompactionClient(),
                    context_tokens=24_576,
                ),
                trust=service.trust,
                event_sink=persist,
            ).run(
                run_id=created.run_id,
                goal=created.goal,
                workspace=workspace,
                initial_catalog=catalog,
                prior_usage=baseline,
            )

        interrupted = service._investigation_recovery(created.run_id)
        assert interrupted.in_flight_request is not None
        assert interrupted.in_flight_request.request_kind == "compaction"
        resumed = service._record_interrupted_model_request(
            service.runs.require(created.run_id),
            interrupted,
        )
    finally:
        service.close()

    assert resumed.in_flight_request is None
    assert resumed.resume_request_kind == "compaction"
    assert resumed.resume_physical_attempts_used == 1
    assert resumed.model_calls[-1].request_kind == "compaction"
    assert resumed.model_calls[-1].outcome == "process_interrupted"
