"""Tests for DatabaseTool."""

from __future__ import annotations

from pathlib import Path

from darwin.tools.database import DatabaseTool


def test_create_insert_and_select_round_trip(tmp_path: Path) -> None:
    db = DatabaseTool(tmp_path)
    create = db.execute({
        "action": "db_exec",
        "path": "test.sqlite3",
        "sql": "CREATE TABLE notes(id INTEGER PRIMARY KEY, text TEXT)",
    })
    assert create.success
    insert = db.execute({
        "action": "db_exec",
        "path": "test.sqlite3",
        "sql": "INSERT INTO notes(text) VALUES (?)",
        "params": ["hello"],
    })
    assert insert.success
    select = db.execute({
        "action": "db_query",
        "path": "test.sqlite3",
        "sql": "SELECT id, text FROM notes",
    })
    assert select.success
    assert "hello" in select.output
    assert select.metadata["rows"] == 1


def test_forbidden_tokens_rejected(tmp_path: Path) -> None:
    db = DatabaseTool(tmp_path)
    bad = db.execute({
        "action": "db_exec",
        "path": "test.sqlite3",
        "sql": "ATTACH DATABASE '/etc/passwd' AS evil",
    })
    assert not bad.success
    assert "forbidden token" in bad.error.lower()


def test_path_escape_rejected(tmp_path: Path) -> None:
    db = DatabaseTool(tmp_path)
    result = db.execute({
        "action": "db_query",
        "path": "../../escape.sqlite3",
        "sql": "SELECT 1",
    })
    assert not result.success
    assert "sandboxescape" in result.error.lower() or "outside" in result.error.lower()


def test_row_cap_truncates_large_result(tmp_path: Path) -> None:
    db = DatabaseTool(tmp_path, max_rows=3)
    db.execute({
        "action": "db_exec",
        "path": "test.sqlite3",
        "sql": "CREATE TABLE n(v INTEGER)",
    })
    for i in range(10):
        db.execute({
            "action": "db_exec",
            "path": "test.sqlite3",
            "sql": "INSERT INTO n VALUES (?)",
            "params": [i],
        })
    result = db.execute({
        "action": "db_query",
        "path": "test.sqlite3",
        "sql": "SELECT v FROM n",
    })
    assert result.success
    assert result.metadata["rows"] == 3


def test_empty_sql_rejected(tmp_path: Path) -> None:
    db = DatabaseTool(tmp_path)
    result = db.execute({"action": "db_query", "sql": ""})
    assert not result.success
    assert "empty" in result.error.lower()
