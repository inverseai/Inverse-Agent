"""Application service connecting durable workflows, approvals, and persisted runs."""

from __future__ import annotations

import builtins
import hashlib
import hmac
import json
import math
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from secrets import token_hex
from typing import Any, BinaryIO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.approvals import (
    ApprovalAuthority,
    SqliteApprovalReplayStore,
    action_digest,
)
from inverse_agent.attestations import (
    SCHEMA_VERSION as ATTESTATION_SCHEMA_VERSION,
)
from inverse_agent.attestations import (
    AttestationScope,
    ScopedTrustStore,
    migrate_attestation_database,
)
from inverse_agent.eval import load_trace
from inverse_agent.fs_tools import FsToolError, PolicyViolationError
from inverse_agent.investigation import (
    AgentAnswer,
    AgentBudget,
    CommandExecution,
    Decision,
    InvestigationLoop,
    InvestigationPlanner,
    InvestigationReport,
    InvestigationVerdict,
    ModelCallRecord,
    SourceCitation,
    StopReason,
    ToolCall,
    ToolObservation,
)
from inverse_agent.migrations import (
    AUXILIARY_SCHEMA_VERSION,
    LEGACY_SCOPE_GENERATIONS,
    RUNS_SCHEMA_VERSION,
    MigrationSpec,
    StateMigrationCoordinator,
    migrate_auxiliary_database,
    migrate_runs_database,
)
from inverse_agent.models import AutonomyLevel, Domain, RunKind, RunSpec, RunStatus
from inverse_agent.planner import Planner
from inverse_agent.policies import default_policy
from inverse_agent.redaction import neutralize_source_instructions, redact_text
from inverse_agent.run_state import is_terminal, require_transition
from inverse_agent.runner import (
    ApprovalChallenge,
    ApprovalNotRequired,
    CommandRequest,
    CommandResult,
    LocalRunner,
)
from inverse_agent.workflow import DurableAgentWorkflow, WorkflowResult

TRACE_PREVIEW_MAX_ACTIONS = 32
TRACE_PREVIEW_MAX_CHARS = 200_000
TRACE_PREVIEW_FIELD_CHARS = 4_096
TRACE_PREVIEW_STREAM_CHARS = 16_384
APPROVAL_GRANT_TTL_SECONDS = 300
COMMAND_OBSERVATION_MAX_BYTES = 12_000
COMMAND_OBSERVATION_MAX_LINES = 160


def _safe_preview_text(value: Any) -> str:
    if value is None:
        return ""
    return redact_text(str(value)).text


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    goal: str
    workspace: str
    domain: str
    autonomy_level: int
    status: str
    pending_approval: dict[str, Any] | None
    trace_path: str | None
    error: str | None
    created_at: float
    updated_at: float
    planner_fingerprint: str
    plan: tuple[str, ...] = ()
    plan_rationale: str = ""
    completed_actions: int = 0
    kind: str = RunKind.VERIFICATION.value
    stop_reason: str | None = None
    budget: dict[str, int | float] | None = None
    usage: dict[str, int | float] | None = None
    answer: dict[str, Any] | None = None
    scope_generations: dict[str, int] | None = None
    endpoint_fingerprint: str = ""
    cancel_requested_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    attempt: int = 0


@dataclass(frozen=True)
class RunEvent:
    sequence: int
    run_id: str
    kind: str
    payload: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class WorkItem:
    work_id: int
    run_id: str
    kind: str
    payload: dict[str, Any]
    attempts: int
    action_ordinal: int | None = None
    actor: str | None = None
    action_digest: str | None = None
    challenge_id: str | None = None
    approved_at: float | None = None
    grant_expires_at: float | None = None
    execution_started_at: float | None = None


@dataclass(frozen=True)
class _InvestigationApprovalRequired(Exception):
    challenge: ApprovalChallenge


@dataclass(frozen=True)
class _InvestigationApprovalExpired(Exception):
    challenge: ApprovalChallenge


@dataclass
class _InvestigationCommandExecutor:
    workspace: Path
    domain: Domain
    commands: dict[str, tuple[str, ...]]
    approval_authority: ApprovalAuthority
    approval_item: WorkItem | None
    event_sink: Callable[[str, dict[str, Any]], None]
    trace_sink: Callable[[str, ApprovalChallenge, CommandResult], None]
    dispatch_guard: Callable[[], AbstractContextManager[bool]]
    scope_lease_check: Callable[[], None]
    evidence_identity_key: bytes

    def execute(
        self,
        call: ToolCall,
        *,
        run_id: str,
        active_deadline: float,
    ) -> CommandExecution:
        command_name = call.command or ""
        argv = self.commands.get(command_name)
        if argv is None:
            raise PolicyViolationError("model selected an unavailable command tool")
        runner = LocalRunner(default_policy(self.workspace), self.approval_authority)
        request = CommandRequest(argv=argv, cwd=self.workspace, domain=self.domain)
        try:
            challenge = runner.approval_challenge(request)
        except ApprovalNotRequired as exc:
            raise PolicyViolationError("investigation commands require a fresh approval") from exc
        item = self.approval_item
        if item is None:
            raise _InvestigationApprovalRequired(challenge)
        if (
            item.actor is None
            or item.action_digest != challenge.action_digest
            or item.challenge_id is None
            or item.grant_expires_at is None
        ):
            raise PolicyViolationError(
                "approved investigation command does not match its challenge"
            )
        expires_at = int(item.grant_expires_at)
        rule = next(
            (
                candidate
                for candidate in default_policy(self.workspace).rules_for(self.domain)
                if candidate.name == challenge.rule
            ),
            None,
        )
        if rule is None:
            raise PolicyViolationError("investigation approval references an unknown rule")
        with self.dispatch_guard():
            self.scope_lease_check()
            now = int(time.time())
            if expires_at <= now:
                raise _InvestigationApprovalExpired(challenge)
            remaining = active_deadline - time.monotonic()
            if remaining <= 0:
                raise FsToolError("active-time budget expired before command execution")
            token, claims = self.approval_authority.issue(
                workspace=self.workspace,
                domain=self.domain,
                rule=rule,
                argv=challenge.argv,
                approved_by=item.actor,
                challenge_id=item.challenge_id,
                now=now,
                expires_at=expires_at,
            )
            self.event_sink(
                "approval.dequeued",
                {
                    "action_ordinal": item.action_ordinal,
                    "approval_id": claims.approval_id,
                    "grant_expires_at": expires_at,
                },
            )
            result = runner.run(
                CommandRequest(
                    argv=argv,
                    cwd=self.workspace,
                    domain=self.domain,
                    approval_token=token,
                    approval_challenge_id=item.challenge_id,
                    timeout_seconds=min(remaining, 3600.0),
                )
            )
        self.approval_item = None
        if result.status == RunStatus.REFUSED:
            raise PolicyViolationError(
                result.reason or "approved investigation command was refused"
            )
        self.trace_sink(command_name, challenge, result)
        observation = _command_observation(
            run_id=run_id,
            command_name=command_name,
            challenge=challenge,
            challenge_id=item.challenge_id,
            approval_id=claims.approval_id,
            result=result,
            identity_key=self.evidence_identity_key,
        )
        return CommandExecution(observation=observation)


def _command_observation(
    *,
    run_id: str,
    command_name: str,
    challenge: ApprovalChallenge,
    challenge_id: str,
    approval_id: str,
    result: CommandResult,
    identity_key: bytes,
) -> ToolObservation:
    returncode = result.returncode if isinstance(result.returncode, int) else "unavailable"
    status_summary = (
        "The command exceeded its compute budget."
        if result.status == RunStatus.FAILED and "compute budget" in result.reason.casefold()
        else "The command reported a failure."
        if result.status == RunStatus.FAILED
        else "The command completed."
    )
    header = (
        f"Command {command_name} {result.status.value} with exit code {returncode}. "
        f"{status_summary}"
    )
    stdout_lines = [line for line in str(result.stdout).splitlines() if line.strip()]
    stderr_lines = [line for line in str(result.stderr).splitlines() if line.strip()]
    raw_lines = [header]
    raw_lines.extend(f"stdout: {line}" for line in stdout_lines)
    raw_lines.extend(f"stderr: {line}" for line in stderr_lines)
    full_redaction = redact_text("\n".join(raw_lines))
    neutralized = neutralize_source_instructions(full_redaction.text)
    full_text = neutralized.text
    full_lines = full_text.splitlines()
    secret_redacted = result.stdout_redacted or result.stderr_redacted or full_redaction.blocked
    source_truncated = (
        result.stdout_truncated
        or result.stderr_truncated
        or "[OUTPUT_TRUNCATED]" in result.stdout
        or "[OUTPUT_TRUNCATED]" in result.stderr
    )
    encoded = full_text.encode("utf-8")
    incomplete = (
        source_truncated
        or secret_redacted
        or neutralized.redacted
        or neutralized.incomplete
        or len(encoded) > COMMAND_OBSERVATION_MAX_BYTES
        or len(full_lines) > COMMAND_OBSERVATION_MAX_LINES
    )
    selected = full_lines
    if incomplete or len(full_lines) > COMMAND_OBSERVATION_MAX_LINES:
        error_terms = ("error", "failed", "fatal", "exception", "denied", "traceback")
        classified = [
            line for line in full_lines[1:] if any(term in line.casefold() for term in error_terms)
        ][:40]
        selected = [
            full_lines[0] if full_lines else "[COMMAND_STATUS_REDACTED]",
            *full_lines[1:51],
            *classified,
            *full_lines[-50:],
            "[DISTILLED_OUTPUT_OMITTED]",
        ]
        selected = list(dict.fromkeys(selected))[:COMMAND_OBSERVATION_MAX_LINES]
    distilled = "\n".join(selected)
    raw = distilled.encode("utf-8")
    if len(raw) > COMMAND_OBSERVATION_MAX_BYTES:
        distilled = raw[:COMMAND_OBSERVATION_MAX_BYTES].decode("utf-8", errors="ignore")
        incomplete = True
    display_lines = tuple(
        f"{index}: {line}" for index, line in enumerate(distilled.splitlines(), start=1)
    )
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    evidence_identity = hmac.digest(
        identity_key,
        f"{run_id}\0{command_name}\0{challenge.action_digest}\0{content_hash}".encode(),
        "sha256",
    ).hex()
    return ToolObservation(
        observation_id=f"obs_command_{token_hex(16)}",
        tool="run_command",
        path=f"command/{command_name}",
        content_hash=content_hash,
        text=distilled,
        lines=display_lines,
        truncated=incomplete,
        incomplete=incomplete,
        redacted=secret_redacted or neutralized.redacted,
        metadata={
            "citable_command": not incomplete and bool(display_lines),
            "instruction_neutralized": neutralized.redacted,
            "output_redacted": secret_redacted,
            "output_truncated": source_truncated,
            "command_name": command_name,
            "status": result.status.value,
            "returncode": result.returncode,
            "rule": challenge.rule,
            "approval_id": approval_id,
            "action_digest": challenge.action_digest,
            "challenge_id": challenge_id,
            "evidence_identity": evidence_identity,
        },
    )


class RunStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        migrate_runs_database(self.path)

    def _connect(self, *, autocommit: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None if autocommit else "DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create(self, spec: RunSpec) -> RunRecord:
        now = time.time()
        record = RunRecord(
            run_id=spec.run_id,
            goal=spec.goal,
            workspace=str(spec.workspace.resolve()),
            domain=spec.domain.value,
            autonomy_level=spec.autonomy_level.value,
            status=RunStatus.PLANNED.value,
            pending_approval=None,
            trace_path=None,
            error=None,
            created_at=now,
            updated_at=now,
            planner_fingerprint=spec.planner_fingerprint,
            kind=spec.kind.value,
            budget=dict(spec.budget),
            usage={},
            scope_generations={},
            endpoint_fingerprint=spec.planner_fingerprint,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, goal, workspace, domain, autonomy_level, status,
                    pending_approval, trace_path, error, created_at, updated_at,
                    planner_fingerprint, plan, plan_rationale, completed_actions,
                    kind, stop_reason, budget, usage, answer, scope_generations,
                    endpoint_fingerprint, cancel_requested_at, started_at,
                    finished_at, attempt
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                self._to_row(record),
            )
            self._append_event(connection, record.run_id, "run.created", {"kind": record.kind})
        return record

    def update_from_result(
        self,
        run_id: str,
        result: WorkflowResult,
        *,
        expected_status: str,
    ) -> RunRecord:
        current = self.require(run_id)
        if current.status != expected_status:
            return current
        trace_path = next(
            (str(artifact.path) for artifact in result.trace.artifacts if artifact.path),
            current.trace_path,
        )
        error_action = next(
            (
                action["metadata"].get("reason")
                for action in reversed(result.trace.actions)
                if action["name"] == "workflow.error"
            ),
            None,
        )
        pending_approval = self._normalize_pending_approval(
            result.pending_approval,
            current,
        )
        updated = RunRecord(
            **{
                **asdict(current),
                "status": result.trace.status.value,
                "pending_approval": pending_approval,
                "trace_path": trace_path,
                "error": error_action,
                "updated_at": time.time(),
                "plan": tuple(result.trace.plan),
                "plan_rationale": result.trace.plan_rationale,
                "completed_actions": sum(
                    action["name"] != "workflow.error" for action in result.trace.actions
                ),
                "finished_at": (
                    time.time() if is_terminal(result.trace.status) else current.finished_at
                ),
            }
        )
        if current.status != updated.status:
            require_transition(RunStatus(current.status), RunStatus(updated.status))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status=?, pending_approval=?, trace_path=?, error=?, updated_at=?,
                    plan=?, plan_rationale=?, completed_actions=?, finished_at=?
                WHERE run_id=? AND status=?
                """,
                (
                    updated.status,
                    json.dumps(updated.pending_approval) if updated.pending_approval else None,
                    updated.trace_path,
                    updated.error,
                    updated.updated_at,
                    json.dumps(updated.plan),
                    updated.plan_rationale,
                    updated.completed_actions,
                    updated.finished_at,
                    run_id,
                    expected_status,
                ),
            )
            if cursor.rowcount == 1:
                self._append_event(
                    connection,
                    run_id,
                    "run.status",
                    {
                        "status": updated.status,
                        "pending_approval": bool(updated.pending_approval),
                    },
                )
        return self.require(run_id)

    @staticmethod
    def _normalize_pending_approval(
        pending: dict[str, Any] | None,
        current: RunRecord,
    ) -> dict[str, Any] | None:
        if pending is None:
            return None
        normalized = dict(pending)
        challenge_id = normalized.get("challenge_id")
        valid_challenge = (
            isinstance(challenge_id, str)
            and len(challenge_id) == 32
            and all(character in "0123456789abcdef" for character in challenge_id)
        )
        if not valid_challenge:
            previous = current.pending_approval or {}
            previous_id = previous.get("challenge_id")
            same_action = previous.get("action_digest") == normalized.get("action_digest")
            valid_previous = (
                isinstance(previous_id, str)
                and len(previous_id) == 32
                and all(character in "0123456789abcdef" for character in previous_id)
            )
            normalized["challenge_id"] = (
                previous_id if same_action and valid_previous else token_hex(16)
            )
        normalized.setdefault("action_ordinal", current.completed_actions)
        return normalized

    def enqueue_start(
        self,
        run_id: str,
        *,
        scope_generations: dict[str, int],
        endpoint_fingerprint: str,
    ) -> RunRecord:
        now = time.time()
        with self._connect(autocommit=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            record = self._require_in(connection, run_id)
            if record.status != RunStatus.PLANNED.value:
                connection.rollback()
                return record
            require_transition(RunStatus.PLANNED, RunStatus.QUEUED)
            connection.execute(
                """
                UPDATE runs SET status=?, scope_generations=?, endpoint_fingerprint=?,
                    updated_at=? WHERE run_id=? AND status=?
                """,
                (
                    RunStatus.QUEUED.value,
                    json.dumps(scope_generations, sort_keys=True),
                    endpoint_fingerprint,
                    now,
                    run_id,
                    RunStatus.PLANNED.value,
                ),
            )
            connection.execute(
                """
                INSERT INTO run_work_items(run_id, kind, payload, state, created_at)
                VALUES (?, 'start', '{}', 'pending', ?)
                """,
                (run_id, now),
            )
            self._append_event(
                connection,
                run_id,
                "run.queued",
                {"work_kind": "start"},
                created_at=now,
            )
            connection.commit()
        return self.require(run_id)

    def bind_legacy_scope_generations(
        self,
        run_id: str,
        *,
        scope_generations: dict[str, int],
        endpoint_fingerprint: str,
    ) -> RunRecord:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs SET scope_generations=?, endpoint_fingerprint=?, updated_at=?
                WHERE run_id=? AND scope_generations=?
                """,
                (
                    json.dumps(scope_generations, sort_keys=True),
                    endpoint_fingerprint,
                    time.time(),
                    run_id,
                    LEGACY_SCOPE_GENERATIONS,
                ),
            )
        return self.require(run_id)

    def enqueue_resume(
        self,
        run_id: str,
        *,
        expected_action_digest: str,
        expected_challenge_id: str,
        approved_by: str,
        approved_at: float,
        grant_expires_at: float,
    ) -> RunRecord:
        now = time.time()
        with self._connect(autocommit=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            record = self._require_in(connection, run_id)
            if record.status != RunStatus.WAITING_FOR_APPROVAL.value:
                connection.rollback()
                raise ValueError("run is not waiting for approval")
            challenge = record.pending_approval or {}
            if (
                challenge.get("action_digest") != expected_action_digest
                or challenge.get("challenge_id") != expected_challenge_id
            ):
                connection.rollback()
                raise ValueError("approval challenge is stale or does not match")
            require_transition(RunStatus.WAITING_FOR_APPROVAL, RunStatus.QUEUED)
            cursor = connection.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=? AND status=?",
                (
                    RunStatus.QUEUED.value,
                    now,
                    run_id,
                    RunStatus.WAITING_FOR_APPROVAL.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("run approval is already being processed")
            connection.execute(
                """
                INSERT INTO run_work_items(
                    run_id, kind, payload, state, created_at, action_ordinal,
                    actor, action_digest, challenge_id, approved_at, grant_expires_at
                ) VALUES (?, 'resume', '{}', 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now,
                    int(challenge.get("action_ordinal", record.completed_actions)),
                    approved_by,
                    expected_action_digest,
                    expected_challenge_id,
                    approved_at,
                    grant_expires_at,
                ),
            )
            self._append_event(
                connection,
                run_id,
                "run.queued",
                {
                    "work_kind": "resume",
                    "action_ordinal": int(
                        challenge.get("action_ordinal", record.completed_actions)
                    ),
                    "grant_expires_at": grant_expires_at,
                },
                created_at=now,
            )
            connection.commit()
        return self.require(run_id)

    def claim_next_work(self) -> WorkItem | None:
        now = time.time()
        with self._connect(autocommit=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT work_id, run_id, kind, payload, attempts, action_ordinal,
                    actor, action_digest, challenge_id, approved_at,
                    grant_expires_at, execution_started_at
                FROM run_work_items WHERE state='pending'
                ORDER BY work_id LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            run_id = str(row["run_id"])
            record = self._require_in(connection, run_id)
            if record.status != RunStatus.QUEUED.value:
                connection.execute(
                    """
                    UPDATE run_work_items SET state='discarded', completed_at=?,
                        last_error='run is no longer queued' WHERE work_id=?
                    """,
                    (now, row["work_id"]),
                )
                connection.commit()
                return None
            target = RunStatus.STARTING if row["kind"] == "start" else RunStatus.APPROVING
            require_transition(RunStatus.QUEUED, target)
            connection.execute(
                """
                UPDATE run_work_items SET state='claimed', claimed_at=?,
                    attempts=attempts+1 WHERE work_id=? AND state='pending'
                """,
                (now, row["work_id"]),
            )
            connection.execute(
                """
                UPDATE runs SET status=?, updated_at=?, started_at=COALESCE(started_at, ?),
                    attempt=attempt+1 WHERE run_id=? AND status=?
                """,
                (target.value, now, now, run_id, RunStatus.QUEUED.value),
            )
            self._append_event(
                connection,
                run_id,
                "run.status",
                {"status": target.value},
                created_at=now,
            )
            connection.commit()
        return WorkItem(
            work_id=int(row["work_id"]),
            run_id=run_id,
            kind=str(row["kind"]),
            payload=json.loads(row["payload"]),
            attempts=int(row["attempts"]) + 1,
            action_ordinal=(
                int(row["action_ordinal"]) if row["action_ordinal"] is not None else None
            ),
            actor=str(row["actor"]) if row["actor"] is not None else None,
            action_digest=(str(row["action_digest"]) if row["action_digest"] is not None else None),
            challenge_id=(str(row["challenge_id"]) if row["challenge_id"] is not None else None),
            approved_at=(float(row["approved_at"]) if row["approved_at"] is not None else None),
            grant_expires_at=(
                float(row["grant_expires_at"]) if row["grant_expires_at"] is not None else None
            ),
            execution_started_at=(
                float(row["execution_started_at"])
                if row["execution_started_at"] is not None
                else None
            ),
        )

    def mark_work_started(self, work_id: int) -> None:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE run_work_items SET execution_started_at=COALESCE(
                    execution_started_at, ?
                ) WHERE work_id=? AND state='claimed'
                """,
                (now, work_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("claimed work item is no longer executable")

    def finish_work(self, work_id: int, *, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE run_work_items SET state='completed', completed_at=?, last_error=?
                WHERE work_id=? AND state='claimed'
                """,
                (time.time(), error, work_id),
            )

    def append_event(self, run_id: str, kind: str, payload: dict[str, Any]) -> RunEvent:
        with self._connect() as connection:
            sequence = self._append_event(connection, run_id, kind, payload)
        return self.events(run_id, after=sequence - 1, limit=1)[0]

    def events(self, run_id: str, *, after: int = 0, limit: int = 200) -> builtins.list[RunEvent]:
        if after < 0:
            raise ValueError("event cursor must be non-negative")
        if not 1 <= limit <= 500:
            raise ValueError("event limit must be between 1 and 500")
        with self._connect() as connection:
            self._require_in(connection, run_id)
            rows = connection.execute(
                """
                SELECT sequence, run_id, kind, payload, created_at FROM run_events
                WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?
                """,
                (run_id, after, limit),
            ).fetchall()
        return [
            RunEvent(
                sequence=int(row["sequence"]),
                run_id=str(row["run_id"]),
                kind=str(row["kind"]),
                payload=json.loads(row["payload"]),
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]

    def set_running(self, run_id: str, *, expected_status: RunStatus) -> RunRecord:
        current = self.require(run_id)
        return self.transition(
            run_id,
            expected=expected_status,
            target=RunStatus.RUNNING,
            pending_approval=current.pending_approval,
        )

    def transition(
        self,
        run_id: str,
        *,
        expected: RunStatus,
        target: RunStatus,
        error: str | None = None,
        stop_reason: str | None = None,
        pending_approval: dict[str, Any] | None = None,
    ) -> RunRecord:
        require_transition(expected, target)
        now = time.time()
        finished_at = now if is_terminal(target) else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status=?, pending_approval=?, error=?, stop_reason=?,
                    updated_at=?, finished_at=COALESCE(?, finished_at)
                WHERE run_id=? AND status=?
                """,
                (
                    target.value,
                    json.dumps(pending_approval) if pending_approval else None,
                    error,
                    stop_reason,
                    now,
                    finished_at,
                    run_id,
                    expected.value,
                ),
            )
            if cursor.rowcount == 1:
                self._append_event(
                    connection,
                    run_id,
                    "run.status",
                    {"status": target.value, "stop_reason": stop_reason},
                    created_at=now,
                )
        return self.require(run_id)

    def cancel(self, run_id: str) -> RunRecord:
        now = time.time()
        with self._connect(autocommit=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            record = self._require_in(connection, run_id)
            current = RunStatus(record.status)
            if is_terminal(current):
                connection.rollback()
                return record
            target = (
                RunStatus.CANCELLED
                if current in {RunStatus.PLANNED, RunStatus.QUEUED, RunStatus.WAITING_FOR_APPROVAL}
                else RunStatus.CANCEL_REQUESTED
            )
            if current == RunStatus.CANCEL_REQUESTED:
                connection.rollback()
                return record
            require_transition(current, target)
            connection.execute(
                """
                UPDATE runs SET status=?, pending_approval=NULL, cancel_requested_at=?,
                    updated_at=?, finished_at=? WHERE run_id=? AND status=?
                """,
                (
                    target.value,
                    now,
                    now,
                    now if target == RunStatus.CANCELLED else None,
                    run_id,
                    current.value,
                ),
            )
            if target == RunStatus.CANCELLED:
                connection.execute(
                    """
                    UPDATE run_work_items SET state='discarded', completed_at=?,
                        last_error='cancelled' WHERE run_id=? AND state='pending'
                    """,
                    (now, run_id),
                )
            self._append_event(
                connection,
                run_id,
                "run.status",
                {"status": target.value},
                created_at=now,
            )
            connection.commit()
        return self.require(run_id)

    def decline(
        self,
        run_id: str,
        *,
        expected_action_digest: str,
        expected_challenge_id: str,
        reason: str,
    ) -> RunRecord:
        now = time.time()
        with self._connect(autocommit=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            record = self._require_in(connection, run_id)
            if record.status != RunStatus.WAITING_FOR_APPROVAL.value:
                connection.rollback()
                raise ValueError("run is not waiting for approval")
            challenge = record.pending_approval or {}
            if (
                challenge.get("action_digest") != expected_action_digest
                or challenge.get("challenge_id") != expected_challenge_id
            ):
                connection.rollback()
                raise ValueError("approval challenge is stale or does not match")
            require_transition(RunStatus.WAITING_FOR_APPROVAL, RunStatus.REFUSED)
            connection.execute(
                """
                UPDATE runs SET status=?, pending_approval=NULL, error=?, stop_reason=?,
                    updated_at=?, finished_at=? WHERE run_id=? AND status=?
                """,
                (
                    RunStatus.REFUSED.value,
                    reason,
                    "approval_declined",
                    now,
                    now,
                    run_id,
                    RunStatus.WAITING_FOR_APPROVAL.value,
                ),
            )
            self._append_event(
                connection,
                run_id,
                "run.status",
                {"status": RunStatus.REFUSED.value, "stop_reason": "approval_declined"},
                created_at=now,
            )
            connection.commit()
        return self.require(run_id)

    def refresh_approval(
        self,
        run_id: str,
        *,
        expected: RunStatus,
        reason: str,
    ) -> RunRecord:
        current = self.require(run_id)
        pending = dict(current.pending_approval or {})
        if not pending.get("action_digest"):
            raise RuntimeError("approval challenge cannot be refreshed")
        pending["challenge_id"] = token_hex(16)
        refreshed = self.transition(
            run_id,
            expected=expected,
            target=RunStatus.WAITING_FOR_APPROVAL,
            pending_approval=pending,
        )
        if refreshed.status == RunStatus.WAITING_FOR_APPROVAL.value:
            self.append_event(
                run_id,
                "approval.refreshed",
                {
                    "reason": reason,
                    "action_ordinal": pending.get("action_ordinal"),
                },
            )
        return refreshed

    def is_cancel_requested(self, run_id: str) -> bool:
        return self.require(run_id).status == RunStatus.CANCEL_REQUESTED.value

    def update_investigation_progress(
        self,
        run_id: str,
        *,
        usage: dict[str, int | float],
        event_kind: str,
        event_payload: dict[str, Any],
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET usage=?, updated_at=? WHERE run_id=? AND status=?",
                (
                    json.dumps(usage, sort_keys=True),
                    now,
                    run_id,
                    RunStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount == 1:
                self._append_event(
                    connection,
                    run_id,
                    event_kind,
                    event_payload,
                    created_at=now,
                )

    def attach_trace(self, run_id: str, trace_path: Path) -> RunRecord:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET trace_path=?, updated_at=? WHERE run_id=?",
                (str(trace_path.resolve()), now, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown run: {run_id}")
        return self.require(run_id)

    def complete_investigation(
        self,
        run_id: str,
        report: InvestigationReport,
        *,
        expected_status: RunStatus,
        usage: dict[str, int | float],
    ) -> RunRecord:
        target = {
            InvestigationVerdict.PASS: RunStatus.SUCCEEDED,
            InvestigationVerdict.INCOMPLETE: RunStatus.INCOMPLETE,
            InvestigationVerdict.FAILED: RunStatus.FAILED,
        }[report.verdict]
        require_transition(expected_status, target)
        now = time.time()
        answer = asdict(report.answer) if report.answer is not None else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status=?, stop_reason=?, error=?, answer=?, usage=?,
                    pending_approval=NULL, updated_at=?, finished_at=?
                WHERE run_id=? AND status=?
                """,
                (
                    target.value,
                    report.stop_reason.value,
                    report.error or None,
                    json.dumps(answer, sort_keys=True) if answer is not None else None,
                    json.dumps(usage, sort_keys=True),
                    now,
                    now,
                    run_id,
                    expected_status.value,
                ),
            )
            if cursor.rowcount == 1:
                self._append_event(
                    connection,
                    run_id,
                    "investigation.finished",
                    {
                        "status": target.value,
                        "stop_reason": report.stop_reason.value,
                        "answer": answer,
                        "usage": usage,
                    },
                    created_at=now,
                )
        return self.require(run_id)

    def recover_work_items(self) -> None:
        """Make crash-interrupted queue claims visible to startup reconciliation."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE run_work_items SET state='pending', claimed_at=NULL,
                    last_error='service restarted before completion'
                WHERE state='claimed'
                """
            )

    def pending_work_item(self, run_id: str) -> WorkItem | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT work_id, run_id, kind, payload, attempts, action_ordinal,
                    actor, action_digest, challenge_id, approved_at,
                    grant_expires_at, execution_started_at
                FROM run_work_items
                WHERE run_id=? AND state='pending' ORDER BY work_id LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkItem(
            work_id=int(row["work_id"]),
            run_id=str(row["run_id"]),
            kind=str(row["kind"]),
            payload=json.loads(row["payload"]),
            attempts=int(row["attempts"]),
            action_ordinal=(
                int(row["action_ordinal"]) if row["action_ordinal"] is not None else None
            ),
            actor=str(row["actor"]) if row["actor"] is not None else None,
            action_digest=(str(row["action_digest"]) if row["action_digest"] is not None else None),
            challenge_id=(str(row["challenge_id"]) if row["challenge_id"] is not None else None),
            approved_at=(float(row["approved_at"]) if row["approved_at"] is not None else None),
            grant_expires_at=(
                float(row["grant_expires_at"]) if row["grant_expires_at"] is not None else None
            ),
            execution_started_at=(
                float(row["execution_started_at"])
                if row["execution_started_at"] is not None
                else None
            ),
        )

    def pending_work_kind(self, run_id: str) -> str | None:
        item = self.pending_work_item(run_id)
        return item.kind if item is not None else None

    def discard_pending_work(self, run_id: str, *, reason: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE run_work_items SET state='discarded', completed_at=?, last_error=?
                WHERE run_id=? AND state='pending'
                """,
                (time.time(), reason, run_id),
            )

    def recover_to_queued(self, run_id: str, *, expected: RunStatus) -> RunRecord:
        require_transition(expected, RunStatus.QUEUED)
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=? AND status=?",
                (RunStatus.QUEUED.value, now, run_id, expected.value),
            )
            if cursor.rowcount == 1:
                self._append_event(
                    connection,
                    run_id,
                    "run.recovered",
                    {"from": expected.value, "status": RunStatus.QUEUED.value},
                    created_at=now,
                )
        return self.require(run_id)

    def claim_start(self, run_id: str) -> RunRecord | None:
        with self._connect(autocommit=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown run: {run_id}")
            record = self._from_row(row)
            if record.status != RunStatus.PLANNED.value:
                connection.rollback()
                return None
            cursor = connection.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=? AND status=?",
                (
                    RunStatus.STARTING.value,
                    time.time(),
                    run_id,
                    RunStatus.PLANNED.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        return self.require(run_id)

    def claim_pending(
        self,
        run_id: str,
        expected_action_digest: str,
        expected_challenge_id: str,
    ) -> RunRecord:
        with self._connect(autocommit=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown run: {run_id}")
            record = self._from_row(row)
            if record.status != RunStatus.WAITING_FOR_APPROVAL.value:
                connection.rollback()
                raise ValueError("run is not waiting for approval")
            actual_digest = (record.pending_approval or {}).get("action_digest")
            if actual_digest != expected_action_digest:
                connection.rollback()
                raise ValueError("approval challenge is stale or does not match")
            actual_challenge_id = (record.pending_approval or {}).get("challenge_id")
            if actual_challenge_id != expected_challenge_id:
                connection.rollback()
                raise ValueError("approval challenge is stale or does not match")
            cursor = connection.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=? AND status=?",
                (
                    RunStatus.APPROVING.value,
                    time.time(),
                    run_id,
                    RunStatus.WAITING_FOR_APPROVAL.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("run approval is already being processed")
            connection.commit()
        return record

    def mark_failed(self, run_id: str, error: str, *, expected_status: str) -> RunRecord:
        return self.transition(
            run_id,
            expected=RunStatus(expected_status),
            target=RunStatus.FAILED,
            error=error,
            stop_reason="recovery_failed",
        )

    def mark_refused(self, run_id: str, error: str, *, expected_status: str) -> RunRecord:
        return self.transition(
            run_id,
            expected=RunStatus(expected_status),
            target=RunStatus.REFUSED,
            error=error,
            stop_reason="recovery_refused",
        )

    def get(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._from_row(row) if row else None

    def require(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise KeyError(f"unknown run: {run_id}")
        return record

    def list(self, *, limit: int = 100, offset: int = 0) -> builtins.list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_by_status(self, statuses: tuple[str, ...]) -> builtins.list[RunRecord]:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _status in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs WHERE status IN ({placeholders}) ORDER BY created_at DESC",
                statuses,
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_active_for_workspace(self, workspace: Path) -> builtins.list[RunRecord]:
        terminal = tuple(status.value for status in RunStatus if is_terminal(status))
        placeholders = ", ".join("?" for _status in terminal)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs WHERE workspace=? AND status NOT IN ({placeholders}) "
                "ORDER BY created_at",
                (str(workspace.resolve()), *terminal),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _to_row(record: RunRecord) -> tuple[Any, ...]:
        return (
            record.run_id,
            record.goal,
            record.workspace,
            record.domain,
            record.autonomy_level,
            record.status,
            json.dumps(record.pending_approval) if record.pending_approval else None,
            record.trace_path,
            record.error,
            record.created_at,
            record.updated_at,
            record.planner_fingerprint,
            json.dumps(record.plan),
            record.plan_rationale,
            record.completed_actions,
            record.kind,
            record.stop_reason,
            json.dumps(record.budget or {}, sort_keys=True),
            json.dumps(record.usage or {}, sort_keys=True),
            json.dumps(record.answer, sort_keys=True) if record.answer is not None else None,
            json.dumps(record.scope_generations or {}, sort_keys=True),
            record.endpoint_fingerprint,
            record.cancel_requested_at,
            record.started_at,
            record.finished_at,
            record.attempt,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=str(row["run_id"]),
            goal=str(row["goal"]),
            workspace=str(row["workspace"]),
            domain=str(row["domain"]),
            autonomy_level=int(row["autonomy_level"]),
            status=str(row["status"]),
            pending_approval=(
                json.loads(row["pending_approval"]) if row["pending_approval"] else None
            ),
            trace_path=str(row["trace_path"]) if row["trace_path"] else None,
            error=str(row["error"]) if row["error"] else None,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            planner_fingerprint=str(row["planner_fingerprint"]),
            plan=tuple(json.loads(row["plan"])) if row["plan"] else (),
            plan_rationale=str(row["plan_rationale"]),
            completed_actions=int(row["completed_actions"]),
            kind=str(row["kind"]),
            stop_reason=str(row["stop_reason"]) if row["stop_reason"] else None,
            budget=json.loads(row["budget"]) if row["budget"] else {},
            usage=json.loads(row["usage"]) if row["usage"] else {},
            answer=json.loads(row["answer"]) if row["answer"] else None,
            scope_generations=(
                json.loads(row["scope_generations"]) if row["scope_generations"] else {}
            ),
            endpoint_fingerprint=str(row["endpoint_fingerprint"]),
            cancel_requested_at=(
                float(row["cancel_requested_at"])
                if row["cancel_requested_at"] is not None
                else None
            ),
            started_at=float(row["started_at"]) if row["started_at"] is not None else None,
            finished_at=(float(row["finished_at"]) if row["finished_at"] is not None else None),
            attempt=int(row["attempt"]),
        )

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        created_at: float | None = None,
    ) -> int:
        cursor = connection.execute(
            "INSERT INTO run_events(run_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (
                run_id,
                kind,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                time.time() if created_at is None else created_at,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("run event sequence was not assigned")
        return int(cursor.lastrowid)

    @classmethod
    def _require_in(cls, connection: sqlite3.Connection, run_id: str) -> RunRecord:
        row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return cls._from_row(row)


class WorkspaceTrustStore(ScopedTrustStore):
    """Compatibility facade whose authority is only ``code_execution``.

    New production code uses the scoped methods directly.  Keeping this narrow
    facade avoids making older local callers silently gain source-read consent.
    """

    def trust(self, workspace: Path, *, trusted_by: str) -> dict[str, Any]:
        granted = self.grant(
            workspace,
            AttestationScope.CODE_EXECUTION,
            granted_by=trusted_by,
        )
        return {
            "workspace": granted["workspace"],
            "trusted_by": granted["granted_by"],
            "trusted_at": granted["granted_at"],
            "generation": granted["generation"],
        }

    def is_trusted(self, workspace: Path) -> bool:
        return self.has_scope(workspace, AttestationScope.CODE_EXECUTION)

    def status(self, workspace: Path) -> dict[str, Any] | None:
        for item in self.status_for(workspace):
            if item["scope"] == AttestationScope.CODE_EXECUTION.value:
                return {
                    "workspace": str(workspace.resolve()),
                    "trusted_by": item["granted_by"],
                    "trusted_at": item["granted_at"],
                    "generation": item["generation"],
                }
        return None


class StateDirectoryLease:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: BinaryIO = path.open("a+b")
        self._locked = False
        try:
            self._handle.seek(0, 2)
            if self._handle.tell() == 0:
                self._handle.write(b"\0")
                self._handle.flush()
            self._handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(
                    self._handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            self._locked = True
        except OSError as exc:
            self._handle.close()
            raise RuntimeError("state directory is already in use") from exc

    def close(self) -> None:
        if not self._locked:
            return
        self._handle.seek(0)
        try:
            if sys.platform == "win32":
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._locked = False
            self._handle.close()


InvestigationPlannerFactory = Callable[[RunRecord], InvestigationPlanner]


@dataclass(frozen=True)
class _GenerationBoundTrust:
    store: ScopedTrustStore
    generations: dict[str, int]

    def has_scope(self, workspace: Path, scope: AttestationScope) -> bool:
        generation = self.generations.get(scope.value)
        return generation is not None and self.store.has_generation(
            workspace,
            scope,
            generation,
        )


@dataclass(frozen=True)
class _InFlightModelRequest:
    request_index: int
    logical_decision: int
    requested_completion_tokens: int
    charged_completion_tokens: int
    started_at: float
    retry_kind: str | None
    transport_retries_used: int
    schema_retries_used: int
    physical_attempts_used: int
    request_kind: str


@dataclass(frozen=True)
class _InvestigationRecovery:
    catalog: tuple[ToolObservation, ...]
    evidence_identities: dict[str, str]
    prior_tool_calls: tuple[ToolCall, ...]
    resume_decision: Decision | None
    model_calls: tuple[ModelCallRecord, ...]
    in_flight_request: _InFlightModelRequest | None
    resume_transport_retries_used: int
    resume_schema_retries_used: int
    resume_physical_attempts_used: int
    resume_request_kind: str
    compaction_notes: str
    compacted_observation_ids: tuple[str, ...]
    result: InvestigationReport | None


class AgentService:
    def __init__(
        self,
        *,
        workspace_root: Path,
        state_dir: Path,
        approval_secret: bytes,
        planner: Planner | None = None,
        planner_fingerprint: str = "deterministic",
        investigation_planner_factory: InvestigationPlannerFactory | None = None,
        investigation_budget: AgentBudget | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.state_dir = state_dir.resolve()
        self.planner_fingerprint = planner_fingerprint
        if self.state_dir.is_relative_to(self.workspace_root):
            raise ValueError("state directory must be outside the workspace root")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._scope_dispatch_lock = threading.RLock()
        self._trace_write_lock = threading.Lock()
        state_lease = StateDirectoryLease(self.state_dir / "service.lock")
        workflow: DurableAgentWorkflow | None = None
        try:
            runs_path = self.state_dir / "runs.sqlite"
            trust_path = self.state_dir / "workspace-trust.sqlite"
            replay_path = self.state_dir / "approval-replay.sqlite"
            checkpoint_path = self.state_dir / "checkpoints.sqlite"
            StateMigrationCoordinator(
                self.state_dir,
                (
                    MigrationSpec(
                        "runs",
                        runs_path,
                        RUNS_SCHEMA_VERSION,
                        migrate_runs_database,
                    ),
                    MigrationSpec(
                        "attestations",
                        trust_path,
                        ATTESTATION_SCHEMA_VERSION,
                        lambda path: migrate_attestation_database(
                            path,
                            legacy_trust_path=path,
                        ),
                    ),
                    MigrationSpec(
                        "approval_replay",
                        replay_path,
                        AUXILIARY_SCHEMA_VERSION,
                        migrate_auxiliary_database,
                    ),
                    MigrationSpec(
                        "checkpoints",
                        checkpoint_path,
                        AUXILIARY_SCHEMA_VERSION,
                        migrate_auxiliary_database,
                    ),
                ),
            ).run()
            replay_store = SqliteApprovalReplayStore(replay_path)
            self.approval_authority = ApprovalAuthority(approval_secret, replay_store)
            self._evidence_identity_root = hmac.digest(
                approval_secret,
                b"inverse-agent/evidence-identity/v1",
                "sha256",
            )
            self.runs = RunStore(runs_path)
            self.trust = ScopedTrustStore(trust_path, legacy_trust_path=trust_path)
            self._bind_legacy_scope_generations()
            workflow = DurableAgentWorkflow(
                checkpoint_path=checkpoint_path,
                trace_dir=self.state_dir / "traces",
                approval_authority=self.approval_authority,
                planner=planner,
            )
            self.workflow = workflow
            self.investigation_planner_factory = investigation_planner_factory
            self.investigation_budget = investigation_budget or AgentBudget()
            self.investigation_budget.validate()
            self._wake_worker = threading.Event()
            self._stop_worker = threading.Event()
            self.runs.recover_work_items()
            self._reconcile_incomplete_runs()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="inverse-agent-run-worker",
                daemon=True,
            )
            self._worker.start()
            self._wake_worker.set()
        except Exception:
            if workflow is not None:
                workflow.close()
            state_lease.close()
            raise
        self._state_lease = state_lease

    def close(self) -> None:
        self._stop_worker.set()
        self._wake_worker.set()
        if threading.current_thread() is not self._worker:
            self._worker.join()
        try:
            self.workflow.close()
        finally:
            self._state_lease.close()

    def create_run(
        self,
        *,
        goal: str,
        workspace: Path,
        domain: Domain,
        kind: RunKind = RunKind.VERIFICATION,
        autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED,
        budget: dict[str, int | float] | None = None,
    ) -> RunRecord:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        if not resolved.is_dir():
            raise ValueError("workspace directory does not exist")
        selected_budget = dict(budget or {})
        if kind == RunKind.INVESTIGATION:
            defaults = asdict(self.investigation_budget)
            defaults.update(selected_budget)
            try:
                parsed_budget = AgentBudget(**defaults)
            except TypeError as exc:
                raise ValueError("investigation budget contains unknown fields") from exc
            parsed_budget.validate()
            selected_budget = asdict(parsed_budget)
        spec = RunSpec(
            goal=redact_text(goal).text,
            workspace=resolved,
            domain=domain,
            kind=kind,
            autonomy_level=autonomy_level,
            budget=selected_budget,
            planner_fingerprint=self.planner_fingerprint,
        )
        return self.runs.create(spec)

    def start(self, run_id: str, *, wait: bool = True) -> RunRecord:
        record = self.runs.require(run_id)
        if record.status == RunStatus.PLANNED.value:
            if record.planner_fingerprint != self.planner_fingerprint:
                raise ValueError("planner configuration changed after run creation")
            if (
                record.kind == RunKind.INVESTIGATION.value
                and self.investigation_planner_factory is None
            ):
                raise ValueError("investigation runs require a configured model planner")
            scopes = self._required_scopes(record)
            try:
                generations = self.trust.capture_generations(Path(record.workspace), scopes)
            except ValueError as exc:
                if AttestationScope.CODE_EXECUTION in scopes:
                    raise ValueError("workspace is not trusted for code execution") from exc
                raise
            record = self.runs.enqueue_start(
                run_id,
                scope_generations=generations,
                endpoint_fingerprint=self.planner_fingerprint,
            )
            self._wake_worker.set()
        if wait and not is_terminal(record.status):
            return self._wait_for_pause_or_terminal(run_id)
        return self.runs.require(run_id)

    def approve_and_resume(
        self,
        run_id: str,
        *,
        approved_by: str,
        expected_action_digest: str,
        expected_challenge_id: str,
        wait: bool = True,
    ) -> RunRecord:
        identity = redact_text(approved_by.strip()).text
        if not identity:
            raise ValueError("approved_by is required")
        record = self.runs.require(run_id)
        if record.status != RunStatus.WAITING_FOR_APPROVAL.value:
            raise ValueError("run is not waiting for approval")
        if not record.pending_approval:
            raise ValueError("pending approval record is missing")
        self._require_scope_lease(record)
        challenge = record.pending_approval
        workspace = Path(challenge["workspace"])
        domain = Domain(challenge["domain"])
        policy = default_policy(workspace)
        rule = next(
            (item for item in policy.rules_for(domain) if item.name == challenge["rule"]),
            None,
        )
        if rule is None:
            raise ValueError("approval challenge references an unknown rule")
        argv = tuple(challenge["argv"])
        expected = action_digest(workspace=workspace, domain=domain, rule=rule, argv=argv)
        if expected != challenge["action_digest"]:
            raise ValueError("approval challenge digest is invalid")
        if (
            expected != expected_action_digest
            or challenge.get("challenge_id") != expected_challenge_id
        ):
            raise ValueError("approval challenge is stale or does not match")
        approved_at = time.time()
        grant_expires_at = approved_at + APPROVAL_GRANT_TTL_SECONDS
        record = self.runs.enqueue_resume(
            run_id,
            expected_action_digest=expected_action_digest,
            expected_challenge_id=expected_challenge_id,
            approved_by=identity,
            approved_at=approved_at,
            grant_expires_at=grant_expires_at,
        )
        self._wake_worker.set()
        if wait:
            return self._wait_for_pause_or_terminal(run_id)
        return record

    def decline(
        self,
        run_id: str,
        *,
        declined_by: str,
        expected_action_digest: str,
        expected_challenge_id: str,
    ) -> RunRecord:
        identity = redact_text(declined_by.strip()).text
        if not identity:
            raise ValueError("declined_by is required")
        return self.runs.decline(
            run_id,
            expected_action_digest=expected_action_digest,
            expected_challenge_id=expected_challenge_id,
            reason=f"declined by {identity}",
        )

    def get(self, run_id: str) -> RunRecord:
        return self.runs.require(run_id)

    def list(self, *, limit: int = 100, offset: int = 0) -> builtins.list[RunRecord]:
        return self.runs.list(limit=limit, offset=offset)

    def trust_workspace(
        self,
        workspace: Path,
        *,
        trusted_by: str,
        scope: AttestationScope = AttestationScope.CODE_EXECUTION,
    ) -> dict[str, Any]:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        if not resolved.is_dir():
            raise ValueError("workspace directory does not exist")
        if not trusted_by.strip():
            raise ValueError("trusted_by is required")
        with self._scope_dispatch_lock:
            previous_generation = self.trust.generation(resolved, scope)
            granted = self.trust.grant(resolved, scope, granted_by=trusted_by.strip())
            if previous_generation is not None:
                self._cancel_scope_leases(resolved, scope)
        self._wake_worker.set()
        if scope == AttestationScope.CODE_EXECUTION:
            return {
                **granted,
                "trusted_by": granted["granted_by"],
                "trusted_at": granted["granted_at"],
            }
        return granted

    def revoke_workspace(self, workspace: Path, *, scope: AttestationScope) -> bool:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        with self._scope_dispatch_lock:
            revoked = self.trust.revoke(resolved, scope)
            if not revoked:
                return False
            self._cancel_scope_leases(resolved, scope)
        self._wake_worker.set()
        return True

    def _cancel_scope_leases(self, workspace: Path, scope: AttestationScope) -> None:
        for record in self.runs.list_active_for_workspace(workspace):
            if scope.value in (record.scope_generations or {}):
                self.runs.cancel(record.run_id)

    def workspace_trust_status(self, workspace: Path) -> dict[str, Any]:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        if not resolved.is_dir():
            raise ValueError("workspace directory does not exist")
        scopes = self.trust.status_for(resolved)
        code_execution = next(
            (item for item in scopes if item["scope"] == AttestationScope.CODE_EXECUTION.value),
            None,
        )
        return {
            "workspace": str(resolved),
            "scopes": {item["scope"]: item for item in scopes},
            # Compatibility projection for v0.1 callers: "trusted" still means
            # code execution only, never source disclosure.
            "trusted": code_execution is not None,
            "trusted_by": code_execution["granted_by"] if code_execution else None,
            "trusted_at": code_execution["granted_at"] if code_execution else None,
        }

    def cancel(self, run_id: str) -> RunRecord:
        record = self.runs.cancel(run_id)
        self._wake_worker.set()
        return record

    def events(self, run_id: str, *, after: int = 0, limit: int = 200) -> builtins.list[RunEvent]:
        return self.runs.events(run_id, after=after, limit=limit)

    def plan_view(self, run_id: str) -> dict[str, Any]:
        record = self.runs.require(run_id)
        return {
            "run_id": record.run_id,
            "status": record.status,
            "plan": list(record.plan),
            "rationale": record.plan_rationale,
            "completed_actions": record.completed_actions,
        }

    def _append_investigation_command_trace(
        self,
        run_id: str,
        command_name: str,
        challenge: ApprovalChallenge,
        result: CommandResult,
    ) -> None:
        record = self.runs.require(run_id)
        trace_path = (self.state_dir / "traces" / f"{run_id}.trace.json").resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_write_lock:
            if trace_path.is_file():
                trace = load_trace(trace_path)
            else:
                trace = {
                    "task_input": record.goal,
                    "domain": record.domain,
                    "baseline": "durable-investigation-workflow-v02",
                    "run_id": run_id,
                    "actions": [],
                    "approvals": [],
                    "artifacts": [],
                    "cost": {},
                    "planner_fingerprint": record.planner_fingerprint,
                    "plan": list(record.plan),
                    "plan_rationale": record.plan_rationale,
                }
            actions = trace.get("actions")
            if not isinstance(actions, list):
                raise ValueError("investigation trace actions are unavailable")
            actions.append(
                {
                    "name": command_name,
                    "metadata": {
                        "status": result.status.value,
                        "rule": challenge.rule,
                        "reason": result.reason,
                        "returncode": result.returncode,
                        "approval_id": result.approval_id,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "stdout_redacted": result.stdout_redacted,
                        "stderr_redacted": result.stderr_redacted,
                        "stdout_truncated": result.stdout_truncated,
                        "stderr_truncated": result.stderr_truncated,
                    },
                    "at": time.time(),
                }
            )
            trace["status"] = record.status
            trace["duration_seconds"] = max(
                0.0,
                time.time() - (record.started_at or record.created_at),
            )
            temporary = trace_path.with_name(f".{trace_path.name}.{token_hex(8)}.tmp")
            try:
                temporary.write_text(
                    json.dumps(trace, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
                os.replace(temporary, trace_path)
            finally:
                temporary.unlink(missing_ok=True)
        self.runs.attach_trace(run_id, trace_path)

    def trace_preview(self, run_id: str) -> dict[str, Any]:
        record = self.runs.require(run_id)
        trace_path = (self.state_dir / "traces" / f"{record.run_id}.trace.json").resolve()
        if not trace_path.is_file():
            raise FileNotFoundError("run trace is not available")
        try:
            trace = load_trace(trace_path)
        except (OSError, ValueError) as exc:
            raise ValueError("run trace is unavailable") from exc

        raw_actions = trace.get("actions", [])
        if not isinstance(raw_actions, list):
            raise ValueError("run trace actions are unavailable")
        budget = TRACE_PREVIEW_MAX_CHARS
        actions: builtins.list[dict[str, Any]] = []
        for raw_action in raw_actions[:TRACE_PREVIEW_MAX_ACTIONS]:
            if not isinstance(raw_action, dict):
                continue
            metadata = raw_action.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            action: dict[str, Any] = {
                "name": _safe_preview_text(raw_action.get("name", ""))[:TRACE_PREVIEW_FIELD_CHARS],
                "status": _safe_preview_text(metadata.get("status", ""))[
                    :TRACE_PREVIEW_FIELD_CHARS
                ],
                "rule": _safe_preview_text(metadata.get("rule", ""))[:TRACE_PREVIEW_FIELD_CHARS],
                "reason": _safe_preview_text(metadata.get("reason", ""))[
                    :TRACE_PREVIEW_FIELD_CHARS
                ],
                "returncode": (
                    metadata.get("returncode")
                    if isinstance(metadata.get("returncode"), int)
                    and not isinstance(metadata.get("returncode"), bool)
                    else None
                ),
            }
            for field_name in ("stdout", "stderr"):
                value = _safe_preview_text(metadata.get(field_name, ""))
                clip_length = min(budget, TRACE_PREVIEW_STREAM_CHARS)
                clipped = value[:clip_length]
                action[field_name] = clipped
                action[f"{field_name}_truncated"] = len(value) > len(clipped)
                budget -= len(clipped)
            actions.append(action)
            if budget <= 0:
                break
        return {
            "run_id": record.run_id,
            "status": record.status,
            "duration_seconds": float(trace.get("duration_seconds", 0.0)),
            "actions": actions,
            "actions_truncated": len(raw_actions) > len(actions),
            "output_truncated": budget <= 0,
        }

    def _worker_loop(self) -> None:
        while not self._stop_worker.is_set():
            item = self.runs.claim_next_work()
            if item is None:
                self._wake_worker.wait(0.25)
                self._wake_worker.clear()
                continue
            error: str | None = None
            try:
                self._execute_work(item)
            except Exception as exc:  # keep the sole worker alive after one bad run
                error = redact_text(str(exc)).text
                current = self.runs.require(item.run_id)
                if current.status == RunStatus.CANCEL_REQUESTED.value:
                    self.runs.transition(
                        item.run_id,
                        expected=RunStatus.CANCEL_REQUESTED,
                        target=RunStatus.CANCELLED,
                        stop_reason="cancelled",
                    )
                elif current.status in {
                    RunStatus.STARTING.value,
                    RunStatus.APPROVING.value,
                    RunStatus.RUNNING.value,
                }:
                    self.runs.transition(
                        item.run_id,
                        expected=RunStatus(current.status),
                        target=RunStatus.FAILED,
                        error=f"worker execution failed: {error}",
                        stop_reason="worker_error",
                    )
            finally:
                self.runs.finish_work(item.work_id, error=error)

    def _execute_work(self, item: WorkItem) -> None:
        record = self.runs.require(item.run_id)
        source_bearing = record.kind == RunKind.INVESTIGATION.value
        if source_bearing and record.endpoint_fingerprint != self.planner_fingerprint:
            raise ValueError("planner configuration changed while the run was queued")
        self._require_scope_lease(record)
        if record.kind == RunKind.INVESTIGATION.value:
            if item.kind not in {"start", "resume"}:
                raise ValueError("investigation work item kind is invalid")
            self._execute_investigation(record, item)
            return
        self._execute_verification(record, item)

    def _execute_verification(self, record: RunRecord, item: WorkItem) -> None:
        expected = RunStatus.STARTING if item.kind == "start" else RunStatus.APPROVING
        if RunStatus(record.status) != expected:
            return
        grant_expires_at = item.grant_expires_at
        if item.kind == "resume":
            if (
                item.actor is None
                or item.action_digest is None
                or item.challenge_id is None
                or grant_expires_at is None
            ):
                raise ValueError("approved resume work item is incomplete")
            if time.time() >= grant_expires_at:
                self.runs.refresh_approval(
                    record.run_id,
                    expected=RunStatus.APPROVING,
                    reason="approval grant expired before dequeue",
                )
                return
        running = self.runs.set_running(record.run_id, expected_status=expected)
        if running.status != RunStatus.RUNNING.value:
            return
        self.runs.mark_work_started(item.work_id)
        with self._scope_dispatch_lock:
            current = self.runs.require(record.run_id)
            if current.status != RunStatus.RUNNING.value:
                return
            self._require_scope_lease(current)
            result = self._invoke_verification_workflow(
                current,
                item,
                grant_expires_at=grant_expires_at,
            )
        if result is None:
            return
        current = self.runs.require(running.run_id)
        if current.status == RunStatus.CANCEL_REQUESTED.value:
            self.runs.transition(
                running.run_id,
                expected=RunStatus.CANCEL_REQUESTED,
                target=RunStatus.CANCELLED,
                stop_reason="cancelled",
            )
            return
        self.runs.update_from_result(
            running.run_id,
            result,
            expected_status=RunStatus.RUNNING.value,
        )

    def _invoke_verification_workflow(
        self,
        running: RunRecord,
        item: WorkItem,
        *,
        grant_expires_at: float | None,
    ) -> WorkflowResult | None:
        if item.kind == "start":
            return self.workflow.start(self._spec(running))
        actor = item.actor
        approved_digest = item.action_digest
        approved_challenge_id = item.challenge_id
        if (
            actor is None
            or approved_digest is None
            or approved_challenge_id is None
            or grant_expires_at is None
        ):
            raise ValueError("approved resume work item is incomplete")
        challenge = running.pending_approval or {}
        if (
            challenge.get("action_digest") != approved_digest
            or challenge.get("challenge_id") != approved_challenge_id
        ):
            raise ValueError("approved resume work item does not match the run challenge")
        now = int(time.time())
        expires_at = int(grant_expires_at)
        if expires_at <= now:
            self.runs.refresh_approval(
                running.run_id,
                expected=RunStatus.RUNNING,
                reason="approval grant expired at execution boundary",
            )
            return None
        workspace = Path(str(challenge["workspace"]))
        domain = Domain(str(challenge["domain"]))
        rule = next(
            (
                candidate
                for candidate in default_policy(workspace).rules_for(domain)
                if candidate.name == challenge["rule"]
            ),
            None,
        )
        if rule is None:
            raise ValueError("approval challenge references an unknown rule")
        approval_token, claims = self.approval_authority.issue(
            workspace=workspace,
            domain=domain,
            rule=rule,
            argv=tuple(challenge["argv"]),
            approved_by=actor,
            challenge_id=approved_challenge_id,
            now=now,
            expires_at=expires_at,
        )
        self.runs.append_event(
            running.run_id,
            "approval.dequeued",
            {
                "action_ordinal": item.action_ordinal,
                "approval_id": claims.approval_id,
                "grant_expires_at": expires_at,
            },
        )
        return self.workflow.resume(
            running.run_id,
            approval_token,
            approved_digest,
            approved_challenge_id,
        )

    @staticmethod
    def _event_string(payload: dict[str, Any], name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str):
            raise ValueError(f"durable investigation event {name} is invalid")
        return value

    @staticmethod
    def _event_int(payload: dict[str, Any], name: str) -> int:
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"durable investigation event {name} is invalid")
        return value

    @classmethod
    def _observation_from_event(cls, raw: object) -> ToolObservation:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("durable investigation observation is invalid")
        lines = raw.get("lines")
        metadata = raw.get("metadata")
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise ValueError("durable investigation observation lines are invalid")
        if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
            raise ValueError("durable investigation observation metadata is invalid")
        flags: dict[str, bool] = {}
        for name in ("truncated", "incomplete", "redacted"):
            value = raw.get(name)
            if not isinstance(value, bool):
                raise ValueError(f"durable investigation observation {name} is invalid")
            flags[name] = value
        start_line = cls._event_int(raw, "start_line")
        if start_line < 1:
            raise ValueError("durable investigation observation start_line is invalid")
        return ToolObservation(
            observation_id=cls._event_string(raw, "observation_id"),
            tool=cls._event_string(raw, "tool"),
            path=cls._event_string(raw, "path"),
            content_hash=cls._event_string(raw, "content_hash"),
            text=cls._event_string(raw, "text"),
            lines=tuple(lines),
            start_line=start_line,
            truncated=flags["truncated"],
            incomplete=flags["incomplete"],
            redacted=flags["redacted"],
            metadata=dict(metadata),
        )

    @classmethod
    def _tool_call_from_event(cls, raw: object) -> ToolCall:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("durable investigation tool decision is invalid")
        optional_strings: dict[str, str | None] = {}
        for name in ("query", "glob", "command", "based_on_observation_id"):
            value = raw.get(name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"durable investigation tool decision {name} is invalid")
            optional_strings[name] = value
        start_line = cls._event_int(raw, "start_line")
        max_lines = cls._event_int(raw, "max_lines")
        if start_line < 1 or max_lines < 1:
            raise ValueError("durable investigation tool line window is invalid")
        return ToolCall(
            tool=cls._event_string(raw, "tool"),
            path=cls._event_string(raw, "path"),
            query=optional_strings["query"],
            glob=optional_strings["glob"],
            start_line=start_line,
            max_lines=max_lines,
            command=optional_strings["command"],
            based_on_observation_id=optional_strings["based_on_observation_id"],
        )

    @classmethod
    def _answer_from_event(cls, raw: object) -> AgentAnswer:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("durable investigation answer is invalid")
        findings = raw.get("findings")
        next_actions = raw.get("next_actions")
        raw_citations = raw.get("citations")
        if not isinstance(findings, list) or not all(
            isinstance(finding, str) for finding in findings
        ):
            raise ValueError("durable investigation findings are invalid")
        if not isinstance(next_actions, list) or not all(
            isinstance(action, str) for action in next_actions
        ):
            raise ValueError("durable investigation next actions are invalid")
        if not isinstance(raw_citations, list):
            raise ValueError("durable investigation citations are invalid")
        citations: builtins.list[SourceCitation] = []
        for raw_citation in raw_citations:
            if not isinstance(raw_citation, dict) or not all(
                isinstance(key, str) for key in raw_citation
            ):
                raise ValueError("durable investigation citation is invalid")
            start_line = cls._event_int(raw_citation, "start_line")
            end_line = cls._event_int(raw_citation, "end_line")
            citations.append(
                SourceCitation(
                    observation_id=cls._event_string(raw_citation, "observation_id"),
                    path=cls._event_string(raw_citation, "path"),
                    start_line=start_line,
                    end_line=end_line,
                    note=cls._event_string(raw_citation, "note"),
                )
            )
        complete = raw.get("complete")
        issue_present = raw.get("issue_present")
        if not isinstance(complete, bool) or not isinstance(issue_present, bool):
            raise ValueError("durable investigation answer flags are invalid")
        return AgentAnswer(
            summary=cls._event_string(raw, "summary"),
            findings=tuple(findings),
            next_actions=tuple(next_actions),
            citations=tuple(citations),
            complete=complete,
            issue_present=issue_present,
        )

    @classmethod
    def _decision_from_event(cls, payload: dict[str, Any]) -> Decision | None:
        raw = payload.get("decision")
        if raw is None:
            return None
        kind = payload.get("decision_kind")
        if kind == "tool":
            return cls._tool_call_from_event(raw)
        if kind == "answer":
            return cls._answer_from_event(raw)
        raise ValueError("durable investigation decision kind is invalid")

    @classmethod
    def _model_calls_from_event(cls, raw: object) -> tuple[ModelCallRecord, ...]:
        if not isinstance(raw, list):
            raise ValueError("durable investigation model-call ledger is invalid")
        calls: builtins.list[ModelCallRecord] = []
        for item in raw:
            if not isinstance(item, dict) or not all(isinstance(key, str) for key in item):
                raise ValueError("durable investigation model-call record is invalid")
            request_index = cls._event_int(item, "request_index")
            logical_decision = cls._event_int(item, "logical_decision")
            if request_index < 1 or logical_decision < 1:
                raise ValueError("durable investigation model-call index is invalid")
            optional_ints: dict[str, int | None] = {}
            for name in ("reported_prompt_tokens", "reported_completion_tokens"):
                value = item.get(name)
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                ):
                    raise ValueError(f"durable investigation model-call {name} is invalid")
                optional_ints[name] = value
            reported_model = item.get("reported_model")
            if reported_model is not None and not isinstance(reported_model, str):
                raise ValueError("durable investigation model-call reported_model is invalid")
            latency = item.get("latency_seconds")
            if (
                not isinstance(latency, int | float)
                or isinstance(latency, bool)
                or not math.isfinite(float(latency))
                or latency < 0
            ):
                raise ValueError("durable investigation model-call latency is invalid")
            outcome = cls._event_string(item, "outcome")
            if not outcome:
                raise ValueError("durable investigation model-call outcome is invalid")
            request_kind = item.get("request_kind", "decision")
            if request_kind not in {"decision", "compaction"}:
                raise ValueError("durable investigation model-call request kind is invalid")
            calls.append(
                ModelCallRecord(
                    request_index=request_index,
                    logical_decision=logical_decision,
                    requested_completion_tokens=cls._event_int(item, "requested_completion_tokens"),
                    charged_completion_tokens=cls._event_int(item, "charged_completion_tokens"),
                    reported_prompt_tokens=optional_ints["reported_prompt_tokens"],
                    reported_completion_tokens=optional_ints["reported_completion_tokens"],
                    reported_model=reported_model,
                    latency_seconds=float(latency),
                    outcome=outcome,
                    request_kind=request_kind,
                )
            )
        return tuple(calls)

    @classmethod
    def _in_flight_request_from_event(cls, raw: object) -> _InFlightModelRequest:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("durable in-flight model request is invalid")
        request_index = cls._event_int(raw, "request_index")
        logical_decision = cls._event_int(raw, "logical_decision")
        if request_index < 1 or logical_decision < 1:
            raise ValueError("durable in-flight model request index is invalid")
        started_at = raw.get("started_at")
        if (
            not isinstance(started_at, int | float)
            or isinstance(started_at, bool)
            or not math.isfinite(float(started_at))
            or started_at < 0
        ):
            raise ValueError("durable in-flight model request timestamp is invalid")
        retry_kind = raw.get("retry_kind")
        if retry_kind not in {None, "transport", "schema"}:
            raise ValueError("durable in-flight model retry kind is invalid")
        transport_retries_used = cls._event_int(raw, "transport_retries_used")
        schema_retries_used = cls._event_int(raw, "schema_retries_used")
        physical_attempts_used = cls._event_int(raw, "physical_attempts_used")
        if transport_retries_used > 1 or schema_retries_used > 1:
            raise ValueError("durable in-flight logical retry count is invalid")
        if not 1 <= physical_attempts_used <= 3:
            raise ValueError("durable in-flight physical-attempt count is invalid")
        if physical_attempts_used < 1 + transport_retries_used + schema_retries_used:
            raise ValueError("durable in-flight physical-attempt state is inconsistent")
        if retry_kind == "transport" and transport_retries_used != 1:
            raise ValueError("durable transport retry state is inconsistent")
        if retry_kind == "schema" and schema_retries_used != 1:
            raise ValueError("durable schema retry state is inconsistent")
        request_kind = raw.get("request_kind", "decision")
        if request_kind not in {"decision", "compaction"}:
            raise ValueError("durable in-flight model request kind is invalid")
        return _InFlightModelRequest(
            request_index=request_index,
            logical_decision=logical_decision,
            requested_completion_tokens=cls._event_int(raw, "requested_completion_tokens"),
            charged_completion_tokens=cls._event_int(raw, "charged_completion_tokens"),
            started_at=float(started_at),
            retry_kind=retry_kind,
            transport_retries_used=transport_retries_used,
            schema_retries_used=schema_retries_used,
            physical_attempts_used=physical_attempts_used,
            request_kind=request_kind,
        )

    @classmethod
    def _report_from_event(
        cls,
        run_id: str,
        payload: dict[str, Any],
        catalog: tuple[ToolObservation, ...],
    ) -> InvestigationReport:
        try:
            verdict = InvestigationVerdict(cls._event_string(payload, "verdict"))
            stop_reason = StopReason(cls._event_string(payload, "stop_reason"))
        except ValueError as exc:
            raise ValueError("durable investigation result enum is invalid") from exc
        raw_answer = payload.get("answer")
        answer = None if raw_answer is None else cls._answer_from_event(raw_answer)
        active_seconds = payload.get("active_seconds")
        if (
            not isinstance(active_seconds, int | float)
            or isinstance(active_seconds, bool)
            or not math.isfinite(float(active_seconds))
            or active_seconds < 0
        ):
            raise ValueError("durable investigation result active_seconds is invalid")
        return InvestigationReport(
            run_id=run_id,
            verdict=verdict,
            stop_reason=stop_reason,
            answer=answer,
            catalog=catalog,
            decisions_used=cls._event_int(payload, "decisions_used"),
            tool_calls_used=cls._event_int(payload, "tool_calls_used"),
            physical_requests_used=cls._event_int(payload, "physical_requests_used"),
            command_calls_used=cls._event_int(payload, "command_calls_used"),
            completion_tokens_used=cls._event_int(payload, "completion_tokens_used"),
            completion_tokens_charged=cls._event_int(payload, "completion_tokens_charged"),
            completion_tokens_requested=cls._event_int(payload, "completion_tokens_requested"),
            observation_bytes_used=cls._event_int(payload, "observation_bytes_used"),
            active_seconds=float(active_seconds),
            transport_retries=cls._event_int(payload, "transport_retries"),
            schema_retries=cls._event_int(payload, "schema_retries"),
            model_calls=cls._model_calls_from_event(payload.get("model_calls")),
            error=cls._event_string(payload, "error"),
        )

    def _investigation_recovery(self, run_id: str) -> _InvestigationRecovery:
        catalog: builtins.list[ToolObservation] = []
        identities: dict[str, str] = {}
        prior_calls: builtins.list[ToolCall] = []
        pending_decision: Decision | None = None
        model_calls: tuple[ModelCallRecord, ...] = ()
        in_flight_request: _InFlightModelRequest | None = None
        resume_transport_retries_used = 0
        resume_schema_retries_used = 0
        resume_physical_attempts_used = 0
        resume_request_kind = "decision"
        compaction_notes = ""
        compacted_observation_ids: tuple[str, ...] = ()
        result: InvestigationReport | None = None
        cursor = 0
        while True:
            events = self.runs.events(run_id, after=cursor, limit=500)
            if not events:
                break
            for event in events:
                cursor = event.sequence
                if "model_calls" in event.payload:
                    model_calls = self._model_calls_from_event(event.payload.get("model_calls"))
                    if in_flight_request is not None and any(
                        call.request_index == in_flight_request.request_index
                        for call in model_calls
                    ):
                        in_flight_request = None
                if event.kind == "investigation.decision":
                    if result is not None or pending_decision is not None:
                        raise ValueError("durable investigation decision ordering is invalid")
                    pending_decision = self._decision_from_event(event.payload)
                    resume_transport_retries_used = 0
                    resume_schema_retries_used = 0
                    resume_physical_attempts_used = 0
                    resume_request_kind = "decision"
                elif event.kind == "investigation.compaction":
                    if result is not None or pending_decision is not None:
                        raise ValueError("durable investigation compaction ordering is invalid")
                    notes = event.payload.get("pinned_notes")
                    raw_ids = event.payload.get("compacted_observation_ids")
                    if not isinstance(notes, str) or not notes.strip() or len(notes) > 4096:
                        raise ValueError("durable investigation compaction notes are invalid")
                    if not isinstance(raw_ids, list) or any(
                        not isinstance(item, str) or not item for item in raw_ids
                    ):
                        raise ValueError("durable compacted observation IDs are invalid")
                    compacted = tuple(raw_ids)
                    if len(set(compacted)) != len(compacted) or not set(compacted).issubset(
                        {observation.observation_id for observation in catalog}
                    ):
                        raise ValueError("durable compaction references unknown observations")
                    compaction_notes = notes
                    compacted_observation_ids = compacted
                    resume_transport_retries_used = 0
                    resume_schema_retries_used = 0
                    resume_physical_attempts_used = 0
                    resume_request_kind = "decision"
                elif event.kind == "investigation.observation":
                    if result is not None or isinstance(pending_decision, AgentAnswer):
                        raise ValueError("durable investigation observation ordering is invalid")
                    observation = self._observation_from_event(event.payload.get("observation"))
                    catalog.append(observation)
                    identity = event.payload.get("evidence_identity")
                    if identity is not None:
                        if not isinstance(identity, str) or not identity:
                            raise ValueError("durable investigation evidence identity is invalid")
                        identities[observation.observation_id] = identity
                    if isinstance(pending_decision, ToolCall):
                        prior_calls.append(pending_decision)
                        pending_decision = None
                elif event.kind == "investigation.result":
                    if result is not None:
                        raise ValueError("durable investigation contains duplicate results")
                    result = self._report_from_event(
                        run_id,
                        event.payload,
                        tuple(catalog),
                    )
                    pending_decision = None
                elif event.kind == "investigation.model_request_started":
                    in_flight_request = self._in_flight_request_from_event(
                        event.payload.get("request")
                    )
                    resume_transport_retries_used = in_flight_request.transport_retries_used
                    resume_schema_retries_used = in_flight_request.schema_retries_used
                    resume_physical_attempts_used = in_flight_request.physical_attempts_used
                    resume_request_kind = in_flight_request.request_kind
                elif event.kind == "investigation.model_request_abandoned":
                    in_flight_request = None
            if len(events) < 500:
                break
        return _InvestigationRecovery(
            catalog=tuple(catalog),
            evidence_identities=identities,
            prior_tool_calls=tuple(prior_calls),
            resume_decision=pending_decision,
            model_calls=model_calls,
            in_flight_request=in_flight_request,
            resume_transport_retries_used=resume_transport_retries_used,
            resume_schema_retries_used=resume_schema_retries_used,
            resume_physical_attempts_used=resume_physical_attempts_used,
            resume_request_kind=resume_request_kind,
            compaction_notes=compaction_notes,
            compacted_observation_ids=compacted_observation_ids,
            result=result,
        )

    def _record_interrupted_model_request(
        self,
        record: RunRecord,
        recovery: _InvestigationRecovery,
    ) -> _InvestigationRecovery:
        request = recovery.in_flight_request
        if request is None:
            return recovery
        if record.status != RunStatus.RUNNING.value:
            raise ValueError("an in-flight model request is attached to a non-running run")
        usage = dict(record.usage or {})
        active_seconds = usage.get("active_seconds", 0.0)
        if (
            not isinstance(active_seconds, int | float)
            or isinstance(active_seconds, bool)
            or not math.isfinite(float(active_seconds))
            or active_seconds < 0
        ):
            raise ValueError("persisted in-flight active-time accounting is invalid")
        physical = usage.get("physical_requests_used", 0)
        requested = usage.get("completion_tokens_requested", 0)
        charged = usage.get("completion_tokens_charged", 0)
        if (
            not isinstance(physical, int)
            or isinstance(physical, bool)
            or physical < request.request_index
            or not isinstance(requested, int)
            or isinstance(requested, bool)
            or requested < request.requested_completion_tokens
            or not isinstance(charged, int)
            or isinstance(charged, bool)
            or charged < request.charged_completion_tokens
        ):
            raise ValueError("persisted in-flight model precharge is invalid")
        interrupted_seconds = max(0.0, time.time() - request.started_at)
        usage["active_seconds"] = float(active_seconds) + interrupted_seconds
        model_calls = recovery.model_calls + (
            ModelCallRecord(
                request_index=request.request_index,
                logical_decision=request.logical_decision,
                requested_completion_tokens=request.requested_completion_tokens,
                charged_completion_tokens=request.charged_completion_tokens,
                reported_prompt_tokens=None,
                reported_completion_tokens=None,
                reported_model=None,
                latency_seconds=interrupted_seconds,
                outcome="process_interrupted",
                request_kind=request.request_kind,
            ),
        )
        self.runs.update_investigation_progress(
            record.run_id,
            usage=usage,
            event_kind="investigation.model_request_abandoned",
            event_payload={
                **usage,
                "abandoned_request_index": request.request_index,
                "resume_transport_retries_used": request.transport_retries_used,
                "resume_schema_retries_used": request.schema_retries_used,
                "resume_physical_attempts_used": request.physical_attempts_used,
                "resume_request_kind": request.request_kind,
                "model_calls": [asdict(call) for call in model_calls],
            },
        )
        updated = self._investigation_recovery(record.run_id)
        if updated.in_flight_request is not None:
            raise RuntimeError("interrupted model request was not durably reconciled")
        return updated

    def _execute_investigation(self, record: RunRecord, item: WorkItem) -> None:
        if self.investigation_planner_factory is None:
            raise ValueError("investigation planner is not configured")
        expected = RunStatus.STARTING if item.kind == "start" else RunStatus.APPROVING
        if RunStatus(record.status) != expected:
            return
        if item.kind == "resume":
            if (
                item.actor is None
                or item.action_digest is None
                or item.challenge_id is None
                or item.grant_expires_at is None
            ):
                raise ValueError("approved investigation resume work item is incomplete")
            if time.time() >= item.grant_expires_at:
                self.runs.refresh_approval(
                    record.run_id,
                    expected=RunStatus.APPROVING,
                    reason="approval grant expired before investigation command dequeue",
                )
                return
        running = self.runs.set_running(record.run_id, expected_status=expected)
        if running.status != RunStatus.RUNNING.value:
            return
        self.runs.mark_work_started(item.work_id)
        budget = AgentBudget(**(running.budget or asdict(self.investigation_budget)))
        budget.validate()
        planner = self.investigation_planner_factory(running)
        baseline = dict(running.usage or {})
        recovery = self._investigation_recovery(running.run_id)
        if recovery.result is not None:
            raise RuntimeError("durable investigation result was not reconciled at startup")
        if recovery.in_flight_request is not None:
            raise RuntimeError("in-flight model request was not reconciled at startup")
        progress = dict(baseline)
        evidence_identity_key = hmac.digest(
            self._evidence_identity_root,
            running.run_id.encode("utf-8"),
            "sha256",
        )

        def event_sink(kind: str, payload: dict[str, object]) -> None:
            for name in (
                "decisions_used",
                "tool_calls_used",
                "physical_requests_used",
                "command_calls_used",
                "observation_bytes_used",
                "completion_tokens_used",
                "completion_tokens_charged",
                "completion_tokens_requested",
                "active_seconds",
                "transport_retries",
                "schema_retries",
            ):
                value = payload.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    progress[name] = max(progress.get(name, 0), value)
            self.runs.update_investigation_progress(
                running.run_id,
                usage=progress,
                event_kind=kind,
                event_payload=dict(payload),
            )

        profile = detect_workspace(Path(running.workspace))
        command_prefix = f"{running.domain}."
        commands = {
            name: tuple(argv)
            for name, argv in profile.commands.items()
            if name.startswith(command_prefix)
        }

        def command_event_sink(kind: str, payload: dict[str, Any]) -> None:
            self.runs.append_event(running.run_id, kind, payload)

        def command_trace_sink(
            command_name: str,
            challenge: ApprovalChallenge,
            result: CommandResult,
        ) -> None:
            self._append_investigation_command_trace(
                running.run_id,
                command_name,
                challenge,
                result,
            )

        def command_scope_lease_check() -> None:
            current = self.runs.require(running.run_id)
            if current.status != RunStatus.RUNNING.value:
                raise ValueError("investigation command is no longer dispatchable")
            self._require_scope_lease(current)

        command_executor = (
            _InvestigationCommandExecutor(
                workspace=Path(running.workspace),
                domain=Domain(running.domain),
                commands=commands,
                approval_authority=self.approval_authority,
                approval_item=(
                    item
                    if item.kind == "resume" and isinstance(recovery.resume_decision, ToolCall)
                    else None
                ),
                event_sink=command_event_sink,
                trace_sink=command_trace_sink,
                dispatch_guard=lambda: self._scope_dispatch_lock,
                scope_lease_check=command_scope_lease_check,
                evidence_identity_key=evidence_identity_key,
            )
            if running.autonomy_level != AutonomyLevel.ADVISORY.value
            else None
        )

        loop = InvestigationLoop(
            planner=planner,
            trust=_GenerationBoundTrust(
                self.trust,
                dict(running.scope_generations or {}),
            ),
            budget=budget,
            command_executor=command_executor,
            event_sink=event_sink,
            cancel_requested=lambda: self.runs.is_cancel_requested(running.run_id),
            evidence_identity_key=evidence_identity_key,
        )
        try:
            report = loop.run(
                run_id=running.run_id,
                goal=running.goal,
                workspace=Path(running.workspace),
                initial_catalog=recovery.catalog,
                prior_usage=baseline,
                initial_evidence_identities=recovery.evidence_identities,
                prior_tool_calls=recovery.prior_tool_calls,
                resume_decision=recovery.resume_decision,
                initial_model_calls=recovery.model_calls,
                initial_transport_retries_used=recovery.resume_transport_retries_used,
                initial_schema_retries_used=recovery.resume_schema_retries_used,
                initial_physical_attempts_used=recovery.resume_physical_attempts_used,
                initial_resume_request_kind=recovery.resume_request_kind,
                initial_compaction_notes=recovery.compaction_notes,
                initial_compacted_observation_ids=recovery.compacted_observation_ids,
            )
        except _InvestigationApprovalExpired:
            self.runs.refresh_approval(
                running.run_id,
                expected=RunStatus.RUNNING,
                reason="approval grant expired at investigation command boundary",
            )
            return
        except _InvestigationApprovalRequired as pause:
            challenge_id = token_hex(16)
            self.runs.transition(
                running.run_id,
                expected=RunStatus.RUNNING,
                target=RunStatus.WAITING_FOR_APPROVAL,
                pending_approval={
                    "kind": "command_approval",
                    "challenge_id": challenge_id,
                    "action_ordinal": int(progress.get("tool_calls_used", 0)),
                    **asdict(pause.challenge),
                },
            )
            return
        current = self.runs.require(running.run_id)
        if (
            report.verdict == InvestigationVerdict.CANCELLED
            or current.status == RunStatus.CANCEL_REQUESTED.value
        ):
            if current.status == RunStatus.RUNNING.value:
                self.runs.transition(
                    running.run_id,
                    expected=RunStatus.RUNNING,
                    target=RunStatus.CANCEL_REQUESTED,
                )
            self.runs.transition(
                running.run_id,
                expected=RunStatus.CANCEL_REQUESTED,
                target=RunStatus.CANCELLED,
                stop_reason="cancelled",
            )
            return
        usage = self._report_usage(baseline, report)
        self.runs.complete_investigation(
            running.run_id,
            report,
            expected_status=RunStatus.RUNNING,
            usage=usage,
        )

    @staticmethod
    def _report_usage(
        baseline: dict[str, int | float], report: InvestigationReport
    ) -> dict[str, int | float]:
        cumulative = {
            "decisions_used": report.decisions_used,
            "tool_calls_used": report.tool_calls_used,
            "physical_requests_used": report.physical_requests_used,
            "command_calls_used": report.command_calls_used,
            "completion_tokens_used": report.completion_tokens_used,
            "completion_tokens_charged": report.completion_tokens_charged,
            "completion_tokens_requested": report.completion_tokens_requested,
            "observation_bytes_used": report.observation_bytes_used,
            "active_seconds": report.active_seconds,
            "transport_retries": report.transport_retries,
            "schema_retries": report.schema_retries,
        }
        usage = dict(baseline)
        usage.update(cumulative)
        return usage

    def _wait_for_pause_or_terminal(self, run_id: str) -> RunRecord:
        while True:
            record = self.runs.require(run_id)
            if record.status == RunStatus.WAITING_FOR_APPROVAL.value or is_terminal(record.status):
                return record
            if not self._worker.is_alive():
                raise RuntimeError("run worker stopped unexpectedly")
            time.sleep(0.01)

    @staticmethod
    def _required_scopes(record: RunRecord) -> tuple[AttestationScope, ...]:
        scopes: builtins.list[AttestationScope] = []
        if record.kind == RunKind.INVESTIGATION.value:
            scopes.append(AttestationScope.SOURCE_READ)
        if record.autonomy_level != AutonomyLevel.ADVISORY.value:
            scopes.append(AttestationScope.CODE_EXECUTION)
        return tuple(scopes)

    def _require_scope_lease(self, record: RunRecord) -> None:
        generations = record.scope_generations or {}
        workspace = Path(record.workspace)
        for scope in self._required_scopes(record):
            generation = generations.get(scope.value)
            if generation is None or not self.trust.has_generation(
                workspace,
                scope,
                generation,
            ):
                raise ValueError(f"workspace attestation for {scope.value} was revoked or changed")

    def _bind_legacy_scope_generations(self) -> None:
        active_statuses = tuple(
            status.value
            for status in RunStatus
            if status != RunStatus.PLANNED and not is_terminal(status)
        )
        for record in self.runs.list_by_status(active_statuses):
            if record.scope_generations != {"__legacy_v01__": 1}:
                continue
            if record.kind != RunKind.VERIFICATION.value:
                continue
            try:
                generations = self.trust.capture_generations(
                    Path(record.workspace),
                    self._required_scopes(record),
                )
            except ValueError:
                continue
            self.runs.bind_legacy_scope_generations(
                record.run_id,
                scope_generations=generations,
                endpoint_fingerprint=record.endpoint_fingerprint or self.planner_fingerprint,
            )

    def _reconcile_incomplete_runs(self) -> None:
        for record in self.runs.list_by_status((RunStatus.CANCEL_REQUESTED.value,)):
            self.runs.discard_pending_work(record.run_id, reason="cancelled during restart")
            self.runs.transition(
                record.run_id,
                expected=RunStatus.CANCEL_REQUESTED,
                target=RunStatus.CANCELLED,
                stop_reason="cancelled",
            )

        active_statuses = (
            RunStatus.STARTING.value,
            RunStatus.APPROVING.value,
            RunStatus.WAITING_FOR_APPROVAL.value,
            RunStatus.RUNNING.value,
        )
        for record in self.runs.list_by_status(active_statuses):
            status = RunStatus(record.status)
            pending_work = self.runs.pending_work_item(record.run_id)
            if record.kind == RunKind.INVESTIGATION.value:
                recovery = self._investigation_recovery(record.run_id)
                if recovery.in_flight_request is not None:
                    recovery = self._record_interrupted_model_request(record, recovery)
                    record = self.runs.require(record.run_id)
                if recovery.result is not None:
                    self.runs.discard_pending_work(
                        record.run_id,
                        reason="durable investigation result already exists",
                    )
                    if recovery.result.verdict == InvestigationVerdict.CANCELLED:
                        self.runs.transition(
                            record.run_id,
                            expected=status,
                            target=RunStatus.CANCEL_REQUESTED,
                        )
                        self.runs.transition(
                            record.run_id,
                            expected=RunStatus.CANCEL_REQUESTED,
                            target=RunStatus.CANCELLED,
                            stop_reason="cancelled",
                        )
                    else:
                        self.runs.complete_investigation(
                            record.run_id,
                            recovery.result,
                            expected_status=status,
                            usage=self._report_usage(
                                dict(record.usage or {}),
                                recovery.result,
                            ),
                        )
                elif (
                    status == RunStatus.WAITING_FOR_APPROVAL
                    and isinstance(recovery.resume_decision, ToolCall)
                    and record.pending_approval is not None
                    and pending_work is None
                ):
                    continue
                elif pending_work is not None and pending_work.kind in {"start", "resume"}:
                    command_outcome_is_unknown = (
                        pending_work.kind == "resume"
                        and pending_work.execution_started_at is not None
                        and isinstance(recovery.resume_decision, ToolCall)
                    )
                    if command_outcome_is_unknown:
                        self.runs.discard_pending_work(
                            record.run_id,
                            reason="approved investigation command outcome may be unknown",
                        )
                        self.runs.mark_failed(
                            record.run_id,
                            "investigation was interrupted after an approved command dispatch; "
                            "command outcome may be unknown",
                            expected_status=record.status,
                        )
                    else:
                        self.runs.recover_to_queued(record.run_id, expected=status)
                else:
                    self.runs.discard_pending_work(
                        record.run_id,
                        reason="investigation work item is unavailable",
                    )
                    self.runs.transition(
                        record.run_id,
                        expected=status,
                        target=RunStatus.INCOMPLETE,
                        error="investigation restart has no durable start work item",
                        stop_reason="recovery_incomplete",
                    )
                continue
            try:
                result = self.workflow.current(record.run_id)
                if (
                    status in {RunStatus.APPROVING, RunStatus.RUNNING}
                    and pending_work is not None
                    and pending_work.kind == "resume"
                    and result.pending_approval is not None
                    and (result.pending_approval or {}).get("challenge_id")
                    == (record.pending_approval or {}).get("challenge_id")
                ):
                    if pending_work.execution_started_at is None:
                        self.runs.recover_to_queued(record.run_id, expected=status)
                    else:
                        self.runs.discard_pending_work(
                            record.run_id,
                            reason="approved command outcome may be unknown",
                        )
                        self.runs.mark_failed(
                            record.run_id,
                            "workflow was interrupted after an approved command dispatch; "
                            "command outcome may be unknown",
                            expected_status=record.status,
                        )
                    continue
                if result.pending_approval is None and result.trace.status not in {
                    RunStatus.SUCCEEDED,
                    RunStatus.INCOMPLETE,
                    RunStatus.CANCELLED,
                    RunStatus.FAILED,
                    RunStatus.REFUSED,
                }:
                    self.runs.discard_pending_work(
                        record.run_id,
                        reason="command outcome may be unknown",
                    )
                    self.runs.mark_failed(
                        record.run_id,
                        "workflow was interrupted before a durable pause; "
                        "command outcome may be unknown",
                        expected_status=record.status,
                    )
                    continue
                self.runs.discard_pending_work(
                    record.run_id,
                    reason="workflow checkpoint already contains the result",
                )
                self.runs.update_from_result(
                    record.run_id,
                    result,
                    expected_status=record.status,
                )
            except KeyError:
                if pending_work is not None and pending_work.kind == "start":
                    self.runs.recover_to_queued(record.run_id, expected=status)
                    continue
                self.runs.mark_failed(
                    record.run_id,
                    "workflow recovery failed: no durable checkpoint exists",
                    expected_status=record.status,
                )
            except Exception as exc:
                self.runs.discard_pending_work(
                    record.run_id,
                    reason="workflow recovery projection failed",
                )
                self.runs.mark_failed(
                    record.run_id,
                    f"workflow recovery failed: {exc}",
                    expected_status=record.status,
                )

        for record in self.runs.list_by_status((RunStatus.QUEUED.value,)):
            if self.runs.pending_work_item(record.run_id) is None:
                self.runs.transition(
                    record.run_id,
                    expected=RunStatus.QUEUED,
                    target=RunStatus.FAILED,
                    error="queued run has no durable work item",
                    stop_reason="queue_corruption",
                )

    @staticmethod
    def _spec(record: RunRecord) -> RunSpec:
        return RunSpec(
            goal=record.goal,
            workspace=Path(record.workspace),
            domain=Domain(record.domain),
            kind=RunKind(record.kind),
            autonomy_level=AutonomyLevel(record.autonomy_level),
            budget=dict(record.budget or {}),
            planner_fingerprint=record.planner_fingerprint,
            run_id=record.run_id,
        )
