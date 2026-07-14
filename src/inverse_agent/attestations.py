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

SCHEMA_VERSION = 2


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
        migrate_attestation_database(self.path, legacy_trust_path=legacy_trust_path)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        _create_v1_schema(connection)

    @staticmethod
    def _user_version(connection: sqlite3.Connection) -> int:
        row = connection.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _migrate_from_legacy(
        connection: sqlite3.Connection, legacy_trust_path: Path | None
    ) -> None:
        _migrate_from_legacy(connection, legacy_trust_path)

    def grant(self, workspace: Path, scope: AttestationScope, *, granted_by: str) -> dict[str, Any]:
        if scope not in _GRANTABLE_SCOPES:
            raise ValueError(f"scope {scope.value!r} cannot be granted in this milestone")
        identity = granted_by.strip()
        if not identity:
            raise ValueError("granted_by is required")
        resolved = str(workspace.resolve())
        granted_at = time.time()
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            generation = self._next_generation(connection, resolved, scope)
            connection.execute(
                """
                INSERT INTO workspace_scope_generations(workspace, scope, generation)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace, scope) DO UPDATE SET generation=excluded.generation
                """,
                (resolved, scope.value, generation),
            )
            connection.execute(
                """
                INSERT INTO workspace_attestations
                    (workspace, scope, granted_by, granted_at, generation)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace, scope) DO UPDATE SET
                    granted_by=excluded.granted_by,
                    granted_at=excluded.granted_at,
                    generation=excluded.generation
                """,
                (resolved, scope.value, identity, granted_at, generation),
            )
            connection.commit()
        return {
            "workspace": resolved,
            "scope": scope.value,
            "granted_by": identity,
            "granted_at": granted_at,
            "generation": generation,
        }

    def revoke(self, workspace: Path, scope: AttestationScope) -> bool:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT 1 FROM workspace_attestations WHERE workspace=? AND scope=?",
                (resolved, scope.value),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            generation = self._next_generation(connection, resolved, scope)
            connection.execute(
                """
                INSERT INTO workspace_scope_generations(workspace, scope, generation)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace, scope) DO UPDATE SET generation=excluded.generation
                """,
                (resolved, scope.value, generation),
            )
            connection.execute(
                "DELETE FROM workspace_attestations WHERE workspace=? AND scope=?",
                (resolved, scope.value),
            )
            connection.commit()
        return True

    def has_scope(self, workspace: Path, scope: AttestationScope) -> bool:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT 1 FROM workspace_attestations WHERE workspace=? AND scope=?",
                (resolved, scope.value),
            ).fetchone()
        return row is not None

    def generation(self, workspace: Path, scope: AttestationScope) -> int | None:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT generation FROM workspace_attestations
                WHERE workspace=? AND scope=?
                """,
                (resolved, scope.value),
            ).fetchone()
        return int(row[0]) if row else None

    def has_generation(self, workspace: Path, scope: AttestationScope, generation: int) -> bool:
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            return False
        return self.generation(workspace, scope) == generation

    def capture_generations(
        self, workspace: Path, scopes: tuple[AttestationScope, ...]
    ) -> dict[str, int]:
        captured: dict[str, int] = {}
        for scope in scopes:
            generation = self.generation(workspace, scope)
            if generation is None:
                raise ValueError(f"workspace is not attested for {scope.value}")
            captured[scope.value] = generation
        return captured

    def scopes_for(self, workspace: Path) -> tuple[str, ...]:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT scope FROM workspace_attestations WHERE workspace=? ORDER BY scope",
                (resolved,),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def status_for(self, workspace: Path) -> tuple[dict[str, Any], ...]:
        resolved = str(workspace.resolve())
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """
                SELECT scope, granted_by, granted_at, generation
                FROM workspace_attestations WHERE workspace=? ORDER BY scope
                """,
                (resolved,),
            ).fetchall()
        return tuple(
            {
                "scope": row[0],
                "granted_by": row[1],
                "granted_at": row[2],
                "generation": row[3],
            }
            for row in rows
        )

    @staticmethod
    def _next_generation(
        connection: sqlite3.Connection, workspace: str, scope: AttestationScope
    ) -> int:
        row = connection.execute(
            """
            SELECT generation FROM workspace_scope_generations
            WHERE workspace=? AND scope=?
            """,
            (workspace, scope.value),
        ).fetchone()
        return (int(row[0]) if row else 0) + 1


def migrate_attestation_database(path: Path, *, legacy_trust_path: Path | None = None) -> None:
    """Run forward-only attestation migrations, refusing unknown versions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30, isolation_level=None) as connection:
        version_row = connection.execute("PRAGMA user_version").fetchone()
        version = int(version_row[0]) if version_row else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                "attestation store has an unrecognized schema version; refusing to open"
            )
        if version == 0:
            connection.execute("BEGIN IMMEDIATE")
            _create_v1_schema(connection)
            _migrate_from_legacy(connection, legacy_trust_path)
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
            version = 1
        if version == 1:
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(workspace_attestations)"
                ).fetchall()
            }
            if "generation" not in columns:
                connection.execute(
                    "ALTER TABLE workspace_attestations "
                    "ADD COLUMN generation INTEGER NOT NULL DEFAULT 1"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_scope_generations (
                    workspace TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    PRIMARY KEY (workspace, scope)
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO workspace_scope_generations
                    (workspace, scope, generation)
                SELECT workspace, scope, generation FROM workspace_attestations
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()


def _create_v1_schema(connection: sqlite3.Connection) -> None:
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


def _migrate_from_legacy(connection: sqlite3.Connection, legacy_trust_path: Path | None) -> None:
    if legacy_trust_path is None or not legacy_trust_path.exists():
        return
    destination_row = connection.execute("PRAGMA database_list").fetchone()
    destination = Path(str(destination_row[2])).resolve() if destination_row else None
    if destination == legacy_trust_path.resolve():
        has_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trusted_workspaces'"
        ).fetchone()
        if not has_table:
            return
        rows = connection.execute(
            "SELECT workspace, trusted_by, trusted_at FROM trusted_workspaces"
        ).fetchall()
        for workspace, trusted_by, trusted_at in rows:
            connection.execute(
                """
                INSERT OR IGNORE INTO workspace_attestations
                    (workspace, scope, granted_by, granted_at)
                VALUES (?, ?, ?, ?)
                """,
                (workspace, AttestationScope.CODE_EXECUTION.value, trusted_by, trusted_at),
            )
        connection.execute("DELETE FROM trusted_workspaces")
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
    for workspace, trusted_by, trusted_at in rows:
        connection.execute(
            """
            INSERT OR IGNORE INTO workspace_attestations
                (workspace, scope, granted_by, granted_at)
            VALUES (?, ?, ?, ?)
            """,
            (workspace, AttestationScope.CODE_EXECUTION.value, trusted_by, trusted_at),
        )
    # Commit the destination rows before clearing the legacy table.  A crash can
    # leave duplicate authority temporarily, but never leave both stores empty.
    connection.commit()
    with sqlite3.connect(legacy_trust_path) as legacy:
        legacy.execute("DELETE FROM trusted_workspaces")
        legacy.commit()
