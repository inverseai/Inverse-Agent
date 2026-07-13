"""Application service connecting durable workflows, approvals, and persisted runs."""

from __future__ import annotations

import builtins
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from secrets import token_hex
from typing import Any, BinaryIO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from inverse_agent.approvals import (
    ApprovalAuthority,
    SqliteApprovalReplayStore,
    action_digest,
)
from inverse_agent.eval import load_trace
from inverse_agent.models import AutonomyLevel, Domain, RunSpec, RunStatus
from inverse_agent.planner import Planner
from inverse_agent.policies import default_policy
from inverse_agent.redaction import redact_text
from inverse_agent.workflow import DurableAgentWorkflow, WorkflowResult

TRACE_PREVIEW_MAX_ACTIONS = 32
TRACE_PREVIEW_MAX_CHARS = 200_000
TRACE_PREVIEW_FIELD_CHARS = 4_096
TRACE_PREVIEW_STREAM_CHARS = 16_384


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


class RunStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    autonomy_level INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    pending_approval TEXT,
                    trace_path TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    planner_fingerprint TEXT NOT NULL DEFAULT 'deterministic',
                    plan TEXT NOT NULL DEFAULT '[]',
                    plan_rationale TEXT NOT NULL DEFAULT '',
                    completed_actions INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
            if "planner_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN planner_fingerprint "
                    "TEXT NOT NULL DEFAULT 'deterministic'"
                )
            if "plan" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN plan TEXT NOT NULL DEFAULT '[]'")
            if "plan_rationale" not in columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN plan_rationale TEXT NOT NULL DEFAULT ''"
                )
            if "completed_actions" not in columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN completed_actions INTEGER NOT NULL DEFAULT 0"
                )

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
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._to_row(record),
            )
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
            }
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                UPDATE runs SET status=?, pending_approval=?, trace_path=?, error=?, updated_at=?,
                    plan=?, plan_rationale=?, completed_actions=?
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
                    run_id,
                    expected_status,
                ),
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

    def claim_start(self, run_id: str) -> RunRecord | None:
        with sqlite3.connect(self.path, timeout=30, isolation_level=None) as connection:
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
        with sqlite3.connect(self.path, timeout=30, isolation_level=None) as connection:
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
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE runs SET status=?, pending_approval=NULL, error=?, updated_at=? "
                "WHERE run_id=? AND status=?",
                (RunStatus.FAILED.value, error, time.time(), run_id, expected_status),
            )
        return self.require(run_id)

    def mark_refused(self, run_id: str, error: str, *, expected_status: str) -> RunRecord:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE runs SET status=?, pending_approval=NULL, error=?, updated_at=? "
                "WHERE run_id=? AND status=?",
                (RunStatus.REFUSED.value, error, time.time(), run_id, expected_status),
            )
        return self.require(run_id)

    def get(self, run_id: str) -> RunRecord | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._from_row(row) if row else None

    def require(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise KeyError(f"unknown run: {run_id}")
        return record

    def list(self, *, limit: int = 100, offset: int = 0) -> list[RunRecord]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_by_status(self, statuses: tuple[str, ...]) -> builtins.list[RunRecord]:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _status in statuses)
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                f"SELECT * FROM runs WHERE status IN ({placeholders}) ORDER BY created_at DESC",
                statuses,
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
        )

    @staticmethod
    def _from_row(row: tuple[Any, ...]) -> RunRecord:
        return RunRecord(
            run_id=row[0],
            goal=row[1],
            workspace=row[2],
            domain=row[3],
            autonomy_level=row[4],
            status=row[5],
            pending_approval=json.loads(row[6]) if row[6] else None,
            trace_path=row[7],
            error=row[8],
            created_at=row[9],
            updated_at=row[10],
            planner_fingerprint=row[11],
            plan=tuple(json.loads(row[12])) if row[12] else (),
            plan_rationale=row[13],
            completed_actions=row[14],
        )


class WorkspaceTrustStore:
    """Explicit local attestation required before any workspace code executes."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trusted_workspaces (
                    workspace TEXT PRIMARY KEY,
                    trusted_by TEXT NOT NULL,
                    trusted_at REAL NOT NULL
                )
                """
            )

    def trust(self, workspace: Path, *, trusted_by: str) -> dict[str, Any]:
        resolved = str(workspace.resolve())
        trusted_at = time.time()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO trusted_workspaces(workspace, trusted_by, trusted_at)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace) DO UPDATE SET
                    trusted_by=excluded.trusted_by, trusted_at=excluded.trusted_at
                """,
                (resolved, trusted_by, trusted_at),
            )
        return {"workspace": resolved, "trusted_by": trusted_by, "trusted_at": trusted_at}

    def is_trusted(self, workspace: Path) -> bool:
        return self.status(workspace) is not None

    def status(self, workspace: Path) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT workspace, trusted_by, trusted_at FROM trusted_workspaces WHERE workspace=?",
                (str(workspace.resolve()),),
            ).fetchone()
        if row is None:
            return None
        return {"workspace": row[0], "trusted_by": row[1], "trusted_at": row[2]}


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


class AgentService:
    def __init__(
        self,
        *,
        workspace_root: Path,
        state_dir: Path,
        approval_secret: bytes,
        planner: Planner | None = None,
        planner_fingerprint: str = "deterministic",
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.state_dir = state_dir.resolve()
        self.planner_fingerprint = planner_fingerprint
        if self.state_dir.is_relative_to(self.workspace_root):
            raise ValueError("state directory must be outside the workspace root")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state_lease = StateDirectoryLease(self.state_dir / "service.lock")
        workflow: DurableAgentWorkflow | None = None
        try:
            replay_store = SqliteApprovalReplayStore(self.state_dir / "approval-replay.sqlite")
            self.approval_authority = ApprovalAuthority(approval_secret, replay_store)
            self.runs = RunStore(self.state_dir / "runs.sqlite")
            self.trust = WorkspaceTrustStore(self.state_dir / "workspace-trust.sqlite")
            workflow = DurableAgentWorkflow(
                checkpoint_path=self.state_dir / "checkpoints.sqlite",
                trace_dir=self.state_dir / "traces",
                approval_authority=self.approval_authority,
                planner=planner,
            )
            self.workflow = workflow
            self._reconcile_incomplete_runs()
        except Exception:
            if workflow is not None:
                workflow.close()
            state_lease.close()
            raise
        self._state_lease = state_lease

    def close(self) -> None:
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
        autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED,
    ) -> RunRecord:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        if not resolved.is_dir():
            raise ValueError("workspace directory does not exist")
        spec = RunSpec(
            goal=redact_text(goal).text,
            workspace=resolved,
            domain=domain,
            autonomy_level=autonomy_level,
            planner_fingerprint=self.planner_fingerprint,
        )
        return self.runs.create(spec)

    def start(self, run_id: str) -> RunRecord:
        record = self.runs.require(run_id)
        if record.status != RunStatus.PLANNED.value:
            return record
        if record.planner_fingerprint != self.planner_fingerprint:
            raise ValueError("planner configuration changed after run creation")
        if record.autonomy_level != AutonomyLevel.ADVISORY.value and not self.trust.is_trusted(
            Path(record.workspace)
        ):
            raise ValueError("workspace is not trusted for code execution")
        claimed = self.runs.claim_start(run_id)
        if claimed is None:
            return self.runs.require(run_id)
        try:
            return self.runs.update_from_result(
                run_id,
                self.workflow.start(self._spec(claimed)),
                expected_status=RunStatus.STARTING.value,
            )
        except Exception as exc:
            error = redact_text(str(exc)).text
            self.runs.mark_failed(
                run_id,
                f"workflow start failed: {error}",
                expected_status=RunStatus.STARTING.value,
            )
            raise

    def approve_and_resume(
        self,
        run_id: str,
        *,
        approved_by: str,
        expected_action_digest: str,
        expected_challenge_id: str,
    ) -> RunRecord:
        record = self.runs.claim_pending(
            run_id,
            expected_action_digest,
            expected_challenge_id,
        )
        try:
            if not record.pending_approval:
                raise ValueError("pending approval record is missing")
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
            token, _claims = self.approval_authority.issue(
                workspace=workspace,
                domain=domain,
                rule=rule,
                argv=argv,
                approved_by=approved_by,
                challenge_id=str(challenge["challenge_id"]),
            )
            return self.runs.update_from_result(
                run_id,
                self.workflow.resume(
                    run_id,
                    token,
                    expected_action_digest,
                    expected_challenge_id,
                ),
                expected_status=RunStatus.APPROVING.value,
            )
        except Exception as exc:
            error = redact_text(str(exc)).text
            self.runs.mark_failed(
                run_id,
                f"approval resume failed: {error}",
                expected_status=RunStatus.APPROVING.value,
            )
            raise

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
        self.runs.claim_pending(
            run_id,
            expected_action_digest,
            expected_challenge_id,
        )
        return self.runs.mark_refused(
            run_id,
            f"declined by {identity}",
            expected_status=RunStatus.APPROVING.value,
        )

    def get(self, run_id: str) -> RunRecord:
        return self.runs.require(run_id)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[RunRecord]:
        return self.runs.list(limit=limit, offset=offset)

    def trust_workspace(self, workspace: Path, *, trusted_by: str) -> dict[str, Any]:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        if not resolved.is_dir():
            raise ValueError("workspace directory does not exist")
        if not trusted_by.strip():
            raise ValueError("trusted_by is required")
        return self.trust.trust(resolved, trusted_by=trusted_by.strip())

    def workspace_trust_status(self, workspace: Path) -> dict[str, Any]:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        if not resolved.is_dir():
            raise ValueError("workspace directory does not exist")
        status = self.trust.status(resolved)
        return {
            "workspace": str(resolved),
            "trusted": status is not None,
            "trusted_by": status["trusted_by"] if status else None,
            "trusted_at": status["trusted_at"] if status else None,
        }

    def plan_view(self, run_id: str) -> dict[str, Any]:
        record = self.runs.require(run_id)
        return {
            "run_id": record.run_id,
            "status": record.status,
            "plan": list(record.plan),
            "rationale": record.plan_rationale,
            "completed_actions": record.completed_actions,
        }

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
        actions: list[dict[str, Any]] = []
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

    def _reconcile_incomplete_runs(self) -> None:
        incomplete_statuses = (
            RunStatus.STARTING.value,
            RunStatus.APPROVING.value,
            RunStatus.WAITING_FOR_APPROVAL.value,
            RunStatus.RUNNING.value,
        )
        for record in self.runs.list_by_status(incomplete_statuses):
            try:
                result = self.workflow.current(record.run_id)
                if result.pending_approval is None and result.trace.status not in {
                    RunStatus.SUCCEEDED,
                    RunStatus.FAILED,
                    RunStatus.REFUSED,
                }:
                    self.runs.mark_failed(
                        record.run_id,
                        "workflow was interrupted before a durable pause; "
                        "command outcome may be unknown",
                        expected_status=record.status,
                    )
                    continue
                self.runs.update_from_result(
                    record.run_id,
                    result,
                    expected_status=record.status,
                )
            except Exception as exc:
                self.runs.mark_failed(
                    record.run_id,
                    f"workflow recovery failed: {exc}",
                    expected_status=record.status,
                )
                continue

    @staticmethod
    def _spec(record: RunRecord) -> RunSpec:
        return RunSpec(
            goal=record.goal,
            workspace=Path(record.workspace),
            domain=Domain(record.domain),
            autonomy_level=AutonomyLevel(record.autonomy_level),
            planner_fingerprint=record.planner_fingerprint,
            run_id=record.run_id,
        )
