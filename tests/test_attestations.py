"""Scoped attestation store, migration, and downgrade-safety tests."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from inverse_agent.attestations import AttestationScope, ScopedTrustStore


def test_grant_and_check_scope(tmp_path: Path) -> None:
    store = ScopedTrustStore(tmp_path / "att.sqlite")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store.grant(workspace, AttestationScope.SOURCE_READ, granted_by="alice")
    assert store.has_scope(workspace, AttestationScope.SOURCE_READ)
    assert not store.has_scope(workspace, AttestationScope.CODE_EXECUTION)
    assert store.scopes_for(workspace) == ("source_read",)


def test_revoke_scope(tmp_path: Path) -> None:
    store = ScopedTrustStore(tmp_path / "att.sqlite")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store.grant(workspace, AttestationScope.SOURCE_READ, granted_by="alice")
    assert store.revoke(workspace, AttestationScope.SOURCE_READ)
    assert not store.has_scope(workspace, AttestationScope.SOURCE_READ)
    assert not store.revoke(workspace, AttestationScope.SOURCE_READ)


def test_model_egress_scope_not_grantable(tmp_path: Path) -> None:
    store = ScopedTrustStore(tmp_path / "att.sqlite")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ValueError, match="cannot be granted"):
        store.grant(workspace, AttestationScope.MODEL_EGRESS, granted_by="alice")


def test_migration_maps_legacy_to_code_execution_only(tmp_path: Path) -> None:
    legacy = tmp_path / "workspace-trust.sqlite"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with sqlite3.connect(legacy) as connection:
        connection.execute(
            "CREATE TABLE trusted_workspaces "
            "(workspace TEXT PRIMARY KEY, trusted_by TEXT NOT NULL, trusted_at REAL NOT NULL)"
        )
        connection.execute(
            "INSERT INTO trusted_workspaces VALUES (?, ?, ?)",
            (str(workspace.resolve()), "legacy-user", time.time()),
        )
    store = ScopedTrustStore(tmp_path / "att.sqlite", legacy_trust_path=legacy)
    assert store.has_scope(workspace, AttestationScope.CODE_EXECUTION)
    # Migration must never confer source_read.
    assert not store.has_scope(workspace, AttestationScope.SOURCE_READ)


def test_migration_empties_legacy_table_for_downgrade_safety(tmp_path: Path) -> None:
    legacy = tmp_path / "workspace-trust.sqlite"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with sqlite3.connect(legacy) as connection:
        connection.execute(
            "CREATE TABLE trusted_workspaces "
            "(workspace TEXT PRIMARY KEY, trusted_by TEXT NOT NULL, trusted_at REAL NOT NULL)"
        )
        connection.execute(
            "INSERT INTO trusted_workspaces VALUES (?, ?, ?)",
            (str(workspace.resolve()), "legacy-user", time.time()),
        )
    ScopedTrustStore(tmp_path / "att.sqlite", legacy_trust_path=legacy)
    # A downgraded v0.1 binary reading the legacy file must now find no rows.
    with sqlite3.connect(legacy) as connection:
        remaining = connection.execute("SELECT COUNT(*) FROM trusted_workspaces").fetchone()[0]
    assert remaining == 0


def test_v01_query_against_migrated_store_authorizes_nothing(tmp_path: Path) -> None:
    """A scope-unaware v0.1 binary must find no legacy rows and fail closed."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = ScopedTrustStore(tmp_path / "att.sqlite")
    store.grant(workspace, AttestationScope.SOURCE_READ, granted_by="alice")
    # Emulate v0.1's exact query shape against the new store's file.
    with sqlite3.connect(store.path) as connection:
        legacy_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trusted_workspaces'"
        ).fetchone()
    assert legacy_table is None  # v0.1 SELECT would raise/return nothing -> not trusted


def test_newer_schema_refused(tmp_path: Path) -> None:
    path = tmp_path / "att.sqlite"
    ScopedTrustStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")
    with pytest.raises(RuntimeError, match="unrecognized schema version"):
        ScopedTrustStore(path)


def test_unexpected_schema_version_refused_before_ddl(tmp_path: Path) -> None:
    # An out-of-range version must be refused before any DDL mutates the file.
    path = tmp_path / "att.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 7")
    with pytest.raises(RuntimeError, match="unrecognized schema version"):
        ScopedTrustStore(path)
