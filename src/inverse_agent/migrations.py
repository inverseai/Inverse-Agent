"""Forward-only, crash-recoverable migrations for application-owned state."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex
from tempfile import NamedTemporaryFile
from typing import Any

RUNS_SCHEMA_VERSION = 3
AUXILIARY_SCHEMA_VERSION = 1
LEGACY_SCOPE_GENERATIONS = '{"__legacy_v01__":1}'


@dataclass(frozen=True)
class MigrationSpec:
    name: str
    path: Path
    target_version: int
    migrate: Callable[[Path], None]


def sqlite_user_version(path: Path) -> int:
    if not path.exists():
        return 0
    with sqlite3.connect(path) as connection:
        row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


class StateMigrationCoordinator:
    """Coordinates a version vector with WAL-consistent SQLite backups.

    Each database migration is transactional on its own SQLite file.  The
    coordinator makes a consistent backup of every application-owned file
    before the first migration and records a completion marker only after all
    files reach their target versions.  An interrupted vector is restored in
    full on the next startup before migrations are attempted again.
    """

    def __init__(self, state_dir: Path, specs: Sequence[MigrationSpec]) -> None:
        self.state_dir = state_dir.resolve()
        self.specs = tuple(specs)
        self.active_dir = self.state_dir / ".migration-active"
        self.manifest_path = self.active_dir / "manifest.json"
        self.marker_path = self.state_dir / ".migration-complete.json"

    def run(self) -> None:
        self._recover_interrupted_vector()
        versions = self._versions()
        for spec in self.specs:
            current = versions[spec.name]
            if current > spec.target_version:
                raise RuntimeError(
                    f"{spec.name} state schema version {current} is newer than this binary "
                    f"supports ({spec.target_version}); refusing to open"
                )
        at_target = {spec.name for spec in self.specs if versions[spec.name] == spec.target_version}
        behind = {spec.name for spec in self.specs if versions[spec.name] < spec.target_version}
        marker = self._read_marker()
        known_completed_vector = marker is not None and marker.get("versions") == versions
        if at_target and behind and not known_completed_vector:
            raise RuntimeError(
                "state databases have a mixed schema-version vector without a recoverable "
                "migration backup; restore the complete state directory from one backup"
            )
        if all(versions[item.name] == item.target_version for item in self.specs):
            return

        migration_id = token_hex(16)
        self._prepare_backups(versions, migration_id=migration_id)
        try:
            for spec in self.specs:
                if sqlite_user_version(spec.path) < spec.target_version:
                    spec.migrate(spec.path)
            final = self._versions()
            expected = self._target_vector()
            if final != expected:
                raise RuntimeError(
                    f"state migration produced an unexpected version vector: {final!r}"
                )
            self._atomic_json(
                self.marker_path,
                {"migration_id": migration_id, "versions": final},
            )
        except Exception:
            # Leave the ready manifest and backups in place.  The next startup
            # restores the complete vector before retrying.
            raise
        self._remove_active_dir()

    def _versions(self) -> dict[str, int]:
        return {spec.name: sqlite_user_version(spec.path) for spec in self.specs}

    def _target_vector(self) -> dict[str, int]:
        return {spec.name: spec.target_version for spec in self.specs}

    def _prepare_backups(self, versions: dict[str, int], *, migration_id: str) -> None:
        if self.active_dir.exists():
            raise RuntimeError(
                "an unresolved state migration directory exists; restore the state directory "
                "from backup before retrying"
            )
        self.active_dir.mkdir(parents=False)
        manifest: dict[str, Any] = {
            "migration_id": migration_id,
            "phase": "backing_up",
            "versions": versions,
            "targets": self._target_vector(),
            "databases": {},
        }
        self._atomic_json(self.manifest_path, manifest)
        try:
            for spec in self.specs:
                existed = spec.path.exists()
                backup_name = f"{spec.name}.sqlite"
                manifest["databases"][spec.name] = {
                    "path": str(spec.path),
                    "backup": backup_name,
                    "existed": existed,
                }
                if existed:
                    self._sqlite_backup(spec.path, self.active_dir / backup_name)
            manifest["phase"] = "ready"
            self._atomic_json(self.manifest_path, manifest)
        except Exception:
            # No migration has started while the manifest is in backing_up.
            self._remove_active_dir()
            raise

    def _recover_interrupted_vector(self) -> None:
        if not self.active_dir.exists():
            return
        if not self.manifest_path.is_file():
            raise RuntimeError(
                "state migration metadata is missing; restore the state directory manually"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("phase") not in {
            "backing_up",
            "ready",
        }:
            raise RuntimeError(
                "state migration metadata is invalid; restore the state directory manually"
            )
        if manifest["phase"] == "backing_up":
            # Migration cannot begin until phase=ready is committed.
            self._remove_active_dir()
            return
        marker = self._read_marker()
        targets = manifest.get("targets")
        migration_id = manifest.get("migration_id")
        if (
            isinstance(migration_id, str)
            and marker is not None
            and marker.get("migration_id") == migration_id
            and marker.get("versions") == targets
        ):
            self._remove_active_dir()
            return
        databases = manifest.get("databases")
        if not isinstance(databases, dict):
            raise RuntimeError(
                "state migration database map is invalid; restore the state directory manually"
            )
        for spec in self.specs:
            entry = databases.get(spec.name)
            if not isinstance(entry, dict) or entry.get("path") != str(spec.path):
                raise RuntimeError(
                    "state migration database map does not match this installation; "
                    "restore the state directory manually"
                )
            existed = entry.get("existed") is True
            backup_name = entry.get("backup")
            backup = self.active_dir / str(backup_name)
            if existed:
                if not backup.is_file():
                    raise RuntimeError(
                        f"state migration backup for {spec.name} is missing; restore manually"
                    )
                self._sqlite_backup(backup, spec.path)
            elif spec.path.exists():
                spec.path.unlink()
        self._remove_active_dir()

    def _read_marker(self) -> dict[str, Any] | None:
        if not self.marker_path.is_file():
            return None
        value = json.loads(self.marker_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    @staticmethod
    def _sqlite_backup(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_connection = sqlite3.connect(source)
        destination_connection = sqlite3.connect(destination)
        try:
            source_connection.backup(destination_connection)
        finally:
            destination_connection.close()
            source_connection.close()

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def _remove_active_dir(self) -> None:
        if self.active_dir.exists():
            shutil.rmtree(self.active_dir)


def migrate_runs_database(path: Path) -> None:
    """Migrate the application-owned run/event/work-item database."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30, isolation_level=None) as connection:
        version_row = connection.execute("PRAGMA user_version").fetchone()
        version = int(version_row[0]) if version_row else 0
        if version > RUNS_SCHEMA_VERSION:
            raise RuntimeError("run database schema is newer than this binary supports")
        if version == 0:
            connection.execute("BEGIN IMMEDIATE")
            _create_or_upgrade_v1_runs(connection)
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
            version = 1
        if version == 1:
            connection.execute("BEGIN IMMEDIATE")
            _upgrade_runs_v2(connection)
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
            version = 2
        if version == 2:
            connection.execute("BEGIN IMMEDIATE")
            _upgrade_runs_v3(connection)
            connection.execute(f"PRAGMA user_version = {RUNS_SCHEMA_VERSION}")
            connection.commit()


def migrate_auxiliary_database(path: Path) -> None:
    """Version an otherwise library-owned SQLite store for vector backup/refusal."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30, isolation_level=None) as connection:
        version_row = connection.execute("PRAGMA user_version").fetchone()
        version = int(version_row[0]) if version_row else 0
        if version > AUXILIARY_SCHEMA_VERSION:
            raise RuntimeError("auxiliary state schema is newer than this binary supports")
        if version == 0:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(f"PRAGMA user_version = {AUXILIARY_SCHEMA_VERSION}")
            connection.commit()


def _create_or_upgrade_v1_runs(connection: sqlite3.Connection) -> None:
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
    columns = _columns(connection, "runs")
    additions = {
        "planner_fingerprint": "TEXT NOT NULL DEFAULT 'deterministic'",
        "plan": "TEXT NOT NULL DEFAULT '[]'",
        "plan_rationale": "TEXT NOT NULL DEFAULT ''",
        "completed_actions": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, declaration in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {declaration}")


def _upgrade_runs_v2(connection: sqlite3.Connection) -> None:
    columns = _columns(connection, "runs")
    legacy_scope_rows = "scope_generations" not in columns
    additions = {
        "kind": "TEXT NOT NULL DEFAULT 'verification'",
        "stop_reason": "TEXT",
        "budget": "TEXT NOT NULL DEFAULT '{}'",
        "usage": "TEXT NOT NULL DEFAULT '{}'",
        "answer": "TEXT",
        "scope_generations": "TEXT NOT NULL DEFAULT '{}'",
        "endpoint_fingerprint": "TEXT NOT NULL DEFAULT ''",
        "cancel_requested_at": "REAL",
        "started_at": "REAL",
        "finished_at": "REAL",
        "attempt": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, declaration in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {declaration}")
    if legacy_scope_rows:
        connection.execute(
            "UPDATE runs SET scope_generations=?",
            (LEGACY_SCOPE_GENERATIONS,),
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS run_work_items (
            work_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK(kind IN ('start', 'resume')),
            payload TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'pending'
                CHECK(state IN ('pending', 'claimed', 'completed', 'discarded')),
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            claimed_at REAL,
            completed_at REAL,
            last_error TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS one_live_work_item_per_run
        ON run_work_items(run_id) WHERE state IN ('pending', 'claimed')
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS run_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS run_events_cursor ON run_events(run_id, sequence)"
    )


def _upgrade_runs_v3(connection: sqlite3.Connection) -> None:
    columns = _columns(connection, "run_work_items")
    additions = {
        "action_ordinal": "INTEGER",
        "actor": "TEXT",
        "action_digest": "TEXT",
        "challenge_id": "TEXT",
        "approved_at": "REAL",
        "grant_expires_at": "REAL",
        "execution_started_at": "REAL",
    }
    for name, declaration in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE run_work_items ADD COLUMN {name} {declaration}")


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
