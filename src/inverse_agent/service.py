"""Application service connecting durable workflows, approvals, and persisted runs."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from inverse_agent.approvals import (
    ApprovalAuthority,
    SqliteApprovalReplayStore,
    action_digest,
)
from inverse_agent.models import AutonomyLevel, Domain, RunSpec, RunStatus
from inverse_agent.planner import Planner
from inverse_agent.policies import default_policy
from inverse_agent.workflow import DurableAgentWorkflow, WorkflowResult


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
                    updated_at REAL NOT NULL
                )
                """
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
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._to_row(record),
            )
        return record

    def update_from_result(self, run_id: str, result: WorkflowResult) -> RunRecord:
        current = self.require(run_id)
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
        updated = RunRecord(
            **{
                **asdict(current),
                "status": result.trace.status.value,
                "pending_approval": result.pending_approval,
                "trace_path": trace_path,
                "error": error_action,
                "updated_at": time.time(),
            }
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                UPDATE runs SET status=?, pending_approval=?, trace_path=?, error=?, updated_at=?
                WHERE run_id=?
                """,
                (
                    updated.status,
                    json.dumps(updated.pending_approval) if updated.pending_approval else None,
                    updated.trace_path,
                    updated.error,
                    updated.updated_at,
                    run_id,
                ),
            )
        return updated

    def claim_pending(self, run_id: str, expected_action_digest: str) -> RunRecord:
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

    def mark_failed(self, run_id: str, error: str) -> RunRecord:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE runs SET status=?, pending_approval=NULL, error=?, updated_at=? WHERE run_id=?",
                (RunStatus.FAILED.value, error, time.time(), run_id),
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

    def list(self) -> list[RunRecord]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
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
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT 1 FROM trusted_workspaces WHERE workspace=?",
                (str(workspace.resolve()),),
            ).fetchone()
        return row is not None


class AgentService:
    def __init__(
        self,
        *,
        workspace_root: Path,
        state_dir: Path,
        approval_secret: bytes,
        planner: Planner | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.state_dir = state_dir.resolve()
        if self.state_dir.is_relative_to(self.workspace_root):
            raise ValueError("state directory must be outside the workspace root")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        replay_store = SqliteApprovalReplayStore(self.state_dir / "approval-replay.sqlite")
        self.approval_authority = ApprovalAuthority(approval_secret, replay_store)
        self.runs = RunStore(self.state_dir / "runs.sqlite")
        self.trust = WorkspaceTrustStore(self.state_dir / "workspace-trust.sqlite")
        self.workflow = DurableAgentWorkflow(
            checkpoint_path=self.state_dir / "checkpoints.sqlite",
            trace_dir=self.state_dir / "traces",
            approval_authority=self.approval_authority,
            planner=planner,
        )
        self._reconcile_incomplete_runs()

    def close(self) -> None:
        self.workflow.close()

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
            goal=goal,
            workspace=resolved,
            domain=domain,
            autonomy_level=autonomy_level,
        )
        return self.runs.create(spec)

    def start(self, run_id: str) -> RunRecord:
        record = self.runs.require(run_id)
        if record.status != RunStatus.PLANNED.value:
            raise ValueError(f"run cannot start from status {record.status}")
        if (
            record.autonomy_level != AutonomyLevel.ADVISORY.value
            and not self.trust.is_trusted(Path(record.workspace))
        ):
            raise ValueError("workspace is not trusted for code execution")
        spec = self._spec(record)
        return self.runs.update_from_result(run_id, self.workflow.start(spec))

    def approve_and_resume(
        self,
        run_id: str,
        *,
        approved_by: str,
        expected_action_digest: str,
    ) -> RunRecord:
        record = self.runs.claim_pending(run_id, expected_action_digest)
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
            )
            return self.runs.update_from_result(run_id, self.workflow.resume(run_id, token))
        except Exception as exc:
            self.runs.mark_failed(run_id, f"approval resume failed: {exc}")
            raise

    def get(self, run_id: str) -> RunRecord:
        return self.runs.require(run_id)

    def list(self) -> list[RunRecord]:
        return self.runs.list()

    def trust_workspace(self, workspace: Path, *, trusted_by: str) -> dict[str, Any]:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("workspace is outside configured workspace root")
        if not resolved.is_dir():
            raise ValueError("workspace directory does not exist")
        if not trusted_by.strip():
            raise ValueError("trusted_by is required")
        return self.trust.trust(resolved, trusted_by=trusted_by.strip())

    def _reconcile_incomplete_runs(self) -> None:
        for record in self.runs.list():
            if record.status not in {
                RunStatus.APPROVING.value,
                RunStatus.WAITING_FOR_APPROVAL.value,
                RunStatus.RUNNING.value,
            }:
                continue
            try:
                result = self.workflow.current(record.run_id)
            except (KeyError, sqlite3.Error, ValueError):
                continue
            self.runs.update_from_result(record.run_id, result)

    @staticmethod
    def _spec(record: RunRecord) -> RunSpec:
        return RunSpec(
            goal=record.goal,
            workspace=Path(record.workspace),
            domain=Domain(record.domain),
            autonomy_level=AutonomyLevel(record.autonomy_level),
            run_id=record.run_id,
        )
