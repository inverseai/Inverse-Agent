"""Crash recovery and fail-closed tests for coordinated state migrations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from inverse_agent.migrations import (
    MigrationSpec,
    StateMigrationCoordinator,
    migrate_auxiliary_database,
)


def _create_database(path: Path, *, version: int, value: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE state(value TEXT NOT NULL)")
        connection.execute("INSERT INTO state VALUES (?)", (value,))
        connection.execute(f"PRAGMA user_version = {version}")


def _migrate_to_v2(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE state SET value=value || '-migrated'")
        connection.execute("PRAGMA user_version = 2")


def _value(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT value FROM state").fetchone()
    assert row is not None
    return str(row[0])


def test_coordinator_restores_whole_vector_after_interrupted_migration(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    _create_database(first, version=1, value="first")
    _create_database(second, version=1, value="second")

    def fail_second(_path: Path) -> None:
        raise RuntimeError("simulated migration crash")

    interrupted = StateMigrationCoordinator(
        tmp_path,
        (
            MigrationSpec("first", first, 2, _migrate_to_v2),
            MigrationSpec("second", second, 2, fail_second),
        ),
    )
    with pytest.raises(RuntimeError, match="simulated migration crash"):
        interrupted.run()

    assert _value(first) == "first-migrated"
    assert _value(second) == "second"
    assert (tmp_path / ".migration-active" / "manifest.json").is_file()

    recovered = StateMigrationCoordinator(
        tmp_path,
        (
            MigrationSpec("first", first, 2, _migrate_to_v2),
            MigrationSpec("second", second, 2, _migrate_to_v2),
        ),
    )
    recovered.run()

    assert _value(first) == "first-migrated"
    assert _value(second) == "second-migrated"
    assert not (tmp_path / ".migration-active").exists()


def test_mixed_version_vector_without_backup_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    _create_database(first, version=2, value="first")
    _create_database(second, version=1, value="second")

    coordinator = StateMigrationCoordinator(
        tmp_path,
        (
            MigrationSpec("first", first, 2, _migrate_to_v2),
            MigrationSpec("second", second, 2, _migrate_to_v2),
        ),
    )
    with pytest.raises(RuntimeError, match="mixed schema-version vector"):
        coordinator.run()

    assert _value(first) == "first"
    assert _value(second) == "second"
    assert not (tmp_path / ".migration-active").exists()


def test_completed_prior_release_vector_can_advance_one_database(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    _create_database(first, version=1, value="first")
    _create_database(second, version=2, value="second")
    (tmp_path / ".migration-complete.json").write_text(
        json.dumps({"versions": {"first": 1, "second": 2}}),
        encoding="utf-8",
    )

    StateMigrationCoordinator(
        tmp_path,
        (
            MigrationSpec("first", first, 2, _migrate_to_v2),
            MigrationSpec("second", second, 2, _migrate_to_v2),
        ),
    ).run()

    assert _value(first) == "first-migrated"
    assert _value(second) == "second"


def test_stale_completion_marker_cannot_discard_newer_interrupted_backup(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    _create_database(first, version=1, value="first")
    _create_database(second, version=1, value="second")
    (tmp_path / ".migration-complete.json").write_text(
        json.dumps(
            {
                "migration_id": "stale-migration",
                "versions": {"first": 2, "second": 2},
            }
        ),
        encoding="utf-8",
    )

    def fail_second(_path: Path) -> None:
        raise RuntimeError("simulated migration crash")

    with pytest.raises(RuntimeError, match="simulated migration crash"):
        StateMigrationCoordinator(
            tmp_path,
            (
                MigrationSpec("first", first, 2, _migrate_to_v2),
                MigrationSpec("second", second, 2, fail_second),
            ),
        ).run()

    StateMigrationCoordinator(
        tmp_path,
        (
            MigrationSpec("first", first, 2, _migrate_to_v2),
            MigrationSpec("second", second, 2, _migrate_to_v2),
        ),
    ).run()

    assert _value(first) == "first-migrated"
    assert _value(second) == "second-migrated"


def test_auxiliary_database_refuses_future_schema(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(RuntimeError, match="newer than this binary"):
        migrate_auxiliary_database(path)
