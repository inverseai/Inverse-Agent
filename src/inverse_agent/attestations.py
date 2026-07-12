"""Scoped workspace attestations.

v0.1 recorded a single boolean trust per workspace whose only meaning was code
execution. Disclosing source to a model is a distinct risk, so v0.2 splits trust
into typed scopes stored in a *new* ``workspace_attestations`` table. The legacy
``trusted_workspaces`` table is intentionally left absent here: a scope-unaware
v0.1 binary that finds no legacy rows fails closed instead of treating a
``source_read`` grant as execution authority.

Existing v0.1 records migrate to ``code_execution`` only (the conservative
reading of the old consent text). The scope enum reserves ``model_egress`` for a
future milestone so per-workspace cloud consent can land on this same storage.
"""

from __future__ import annotations

import sqlite3
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class AttestationScope(StrEnum):
    SOURCE_READ = "source_read"
    CODE_EXECUTION = "code_execution"
    # Reserved for v0.3 per-workspace cloud consent; not yet grantable.
    MODEL_EGRESS = "model_egress"


_GRANTABLE_SCOPES = frozenset({AttestationScope.SOURCE_READ, AttestationScope.CODE_EXECUTION})


class ScopedTrustStore:
    """Durable, scope-aware workspace attestations with revocation."""

    def __init__(self, path: Path, *, legacy_trust_path: Path | None = None) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            # Fail closed on an unexpected version BEFORE any DDL mutates the file.
            version = self._user_version(connection)
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError(
                    "attestation store has an unrecognized schema version; refusing to open"
                )
            self._create_schema(connection)
            if version == 0:
                self._migrate_from_legacy(connection, legacy_trust_path)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_attestations (
                workspace TEXT NOT NULL,
                scope TEXT NOT NULL,
                granted_by TEXT NOT NULL,
                granted_at REAL NOT NULL,
                PRIMARY KEY (workspace, scope)
            )
            """
        )

    @staticmethod
    def _user_version(connection: sqlite3.Connection) -> int:
        row = connection.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _migrate_from_legacy(
        connection: sqlite3.Connection, legacy_trust_path: Path | None
    ) -> None:
        if legacy_trust_path is None or not legacy_trust_path.exists():
            return
        with sqlite3.connect(legacy_trust_path) as legacy:
            has_table = legacy.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trusted_workspaces'"
            ).fetchone()
            if not has_table:
                return
            rows = legacy.execute(
                "SELECT workspace, trusted_by, trusted_at FROM trusted_workspaces"
            ).fetchall()
        # Durably commit the destination rows FIRST, so a crash cannot leave both
        # stores empty. Only after the new scope rows are committed do we clear
        # the legacy table (so a downgraded v0.1 binary fails closed).
        for workspace, trusted_by, trusted_at in rows:
            connection.execute(
                """
                INSERT OR IGNORE INTO workspace_attestations
                    (workspace, scope, granted_by, granted_at)
                VALUES (?, ?, ?, ?)
                """,
                (workspace, AttestationScope.CODE_EXECUTION.value, trusted_by, trusted_at),
            )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
        with sqlite3.connect(legacy_trust_path) as legacy:
            legacy.execute("DELETE FROM trusted_workspaces")
            legacy.commit()

    def grant(
        self, workspace: Path, scope: AttestationScope, *, granted_by: str
    ) -> dict[str, Any]:
        if scope not in _GRANTABLE_SCOPES:
            raise ValueError(f"scope {scope.value!r} cannot be granted in this milestone")
        identity = granted_by.strip()
        if not identity:
            raise ValueError("granted_by is required")
        resolved = str(workspace.resolve())
        granted_at = time.time()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO workspace_attestations (workspace, scope, granted_by, granted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(workspace, scope) DO UPDATE SET
                    granted_by=excluded.granted_by, granted_at=excluded.granted_at
                """,
                (resolved, scope.value, identity, granted_at),
            )
        return {
            "workspace": resolved,
            "scope": scope.value,
            "granted_by": identity,
            "granted_at": granted_at,
        }

    def revoke(self, workspace: Path, scope: AttestationScope) -> bool:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                "DELETE FROM workspace_attestations WHERE workspace=? AND scope=?",
                (resolved, scope.value),
            )
            return cursor.rowcount > 0

    def has_scope(self, workspace: Path, scope: AttestationScope) -> bool:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT 1 FROM workspace_attestations WHERE workspace=? AND scope=?",
                (resolved, scope.value),
            ).fetchone()
        return row is not None

    def scopes_for(self, workspace: Path) -> tuple[str, ...]:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT scope FROM workspace_attestations WHERE workspace=? ORDER BY scope",
                (resolved,),
            ).fetchall()
        return tuple(row[0] for row in rows)
