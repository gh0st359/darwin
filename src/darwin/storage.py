from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from darwin.types import Transition


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def loads(value: str) -> Any:
    return json.loads(value)


class PersistentStore:
    """SQLite-backed durable memory for Darwin's experience stream."""

    def __init__(self, path: str | Path = "darwin_memory.sqlite3") -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._initialize()

    def record_transition(self, transition: Transition) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into transitions(t, action, before_state, after_state, reward, metadata)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    transition.t,
                    transition.action,
                    dumps(dict(transition.before)),
                    dumps(dict(transition.after)),
                    float(transition.reward),
                    dumps(dict(transition.metadata)),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def load_transitions(self, limit: int | None = None) -> list[Transition]:
        sql = """
            select t, action, before_state, after_state, reward, metadata
            from transitions
            order by id asc
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " limit ?"
            params = (limit,)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [
            Transition(
                before=loads(row["before_state"]),
                action=row["action"],
                after=loads(row["after_state"]),
                reward=float(row["reward"]),
                t=int(row["t"]),
                metadata=loads(row["metadata"]),
            )
            for row in rows
        ]

    def record_concept(self, concept: Mapping[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                insert into concepts(name, kind, level, support, reward_mean, payload, updated_at)
                values (?, ?, ?, ?, ?, ?, current_timestamp)
                on conflict(name) do update set
                    kind = excluded.kind,
                    level = excluded.level,
                    support = excluded.support,
                    reward_mean = excluded.reward_mean,
                    payload = excluded.payload,
                    updated_at = current_timestamp
                """,
                (
                    concept["name"],
                    concept["kind"],
                    int(concept.get("level", 0)),
                    int(concept.get("support", 0)),
                    float(concept.get("reward_mean", 0.0)),
                    dumps(dict(concept)),
                ),
            )
            connection.commit()

    def record_thought(self, kind: str, content: str, payload: Mapping[str, Any] | None = None) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into thoughts(kind, content, payload, created_at)
                values (?, ?, ?, current_timestamp)
                """,
                (kind, content, dumps(dict(payload or {}))),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_thoughts(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select kind, content, payload, created_at
                from thoughts
                order by id desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "kind": row["kind"],
                "content": row["content"],
                "payload": loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_chat(self, role: str, content: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into chat_messages(role, content, created_at)
                values (?, ?, current_timestamp)
                """,
                (role, content),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_chat(self, limit: int = 20) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select role, content, created_at
                from chat_messages
                order by id desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in rows
        ]

    def record_experiment(self, payload: Mapping[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into experiments(status, action, uncertainty, prediction, result, created_at)
                values (?, ?, ?, ?, ?, current_timestamp)
                """,
                (
                    payload.get("status", "proposed"),
                    payload.get("action", ""),
                    float(payload.get("uncertainty", 0.0)),
                    dumps(payload.get("prediction", {})),
                    dumps(payload.get("result", {})),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def record_plan(self, payload: Mapping[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into plans(goal, score, actions, payload, created_at)
                values (?, ?, ?, ?, current_timestamp)
                """,
                (
                    dumps(payload.get("goal", {})),
                    float(payload.get("score", 0.0)),
                    dumps(payload.get("actions", [])),
                    dumps(dict(payload)),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def record_semantic_frame(self, payload: Mapping[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into semantic_frames(
                    source, speech_act, topic, intent, confidence, uncertainty,
                    original_text, payload, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
                """,
                (
                    payload.get("source", "unknown"),
                    payload.get("speech_act", "statement"),
                    payload.get("topic", "general"),
                    payload.get("intent", "conversation"),
                    float(payload.get("confidence", 0.0)),
                    float(payload.get("uncertainty", 1.0)),
                    payload.get("original_text", ""),
                    dumps(dict(payload)),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def load_semantic_records(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
            select payload
            from semantic_frames
            order by id asc
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " limit ?"
            params = (limit,)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [loads(row["payload"]) for row in rows]

    def recent_semantic_records(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select payload
                from semantic_frames
                order by id desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [loads(row["payload"]) for row in rows]

    def record_self_modification(self, payload: Mapping[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into self_modifications(proposal_id, kind, target, status, payload, created_at)
                values (?, ?, ?, ?, ?, current_timestamp)
                """,
                (
                    str(payload.get("proposal_id", "")),
                    str(payload.get("kind", "")),
                    str(payload.get("target", "")),
                    str(payload.get("status", "proposed")),
                    dumps(dict(payload)),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_self_modifications(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select proposal_id, kind, target, status, payload, created_at
                from self_modifications
                order by id desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "proposal_id": row["proposal_id"],
                "kind": row["kind"],
                "target": row["target"],
                "status": row["status"],
                "payload": loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        tables = [
            "transitions",
            "concepts",
            "thoughts",
            "chat_messages",
            "experiments",
            "plans",
            "semantic_frames",
            "self_modifications",
            "quarantine",
            "snapshots",
            "gate_history",
        ]
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"select count(*) from {table}").fetchone()[0])
                for table in tables
            }

    def record_quarantine(self, payload: Mapping[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into quarantine(entry_id, proposal_id, kind, status, description,
                                       snapshot_id, payload, created_at)
                values (?, ?, ?, ?, ?, ?, ?, current_timestamp)
                """,
                (
                    str(payload.get("entry_id", "")),
                    str(payload.get("proposal_id", "")),
                    str(payload.get("kind", "")),
                    str(payload.get("status", "applied")),
                    str(payload.get("description", "")),
                    str(payload.get("snapshot_id", "")),
                    dumps(dict(payload)),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_quarantine(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select entry_id, proposal_id, kind, status, description, snapshot_id,
                       payload, created_at
                from quarantine
                order by id desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "entry_id": row["entry_id"],
                "proposal_id": row["proposal_id"],
                "kind": row["kind"],
                "status": row["status"],
                "description": row["description"],
                "snapshot_id": row["snapshot_id"],
                "payload": loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_snapshot(self, payload: Mapping[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into snapshots(snapshot_id, gate_identity, content_hash, payload, created_at)
                values (?, ?, ?, ?, current_timestamp)
                """,
                (
                    str(payload.get("snapshot_id", "")),
                    str(payload.get("gate_identity", "")),
                    str(payload.get("content_hash", "")),
                    dumps(dict(payload)),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def record_gate_history(self, payload: Mapping[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into gate_history(old_gate_id, new_gate_id, shadow_agreement,
                                        shadow_sample_size, notes, created_at)
                values (?, ?, ?, ?, ?, current_timestamp)
                """,
                (
                    str(payload.get("old_gate_id", "")),
                    str(payload.get("new_gate_id", "")),
                    float(payload.get("shadow_agreement", 0.0)),
                    int(payload.get("shadow_sample_size", 0)),
                    str(payload.get("notes", "")),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
            finally:
                connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                create table if not exists transitions (
                    id integer primary key autoincrement,
                    t integer not null,
                    action text not null,
                    before_state text not null,
                    after_state text not null,
                    reward real not null,
                    metadata text not null,
                    created_at text default current_timestamp
                );

                create table if not exists concepts (
                    name text primary key,
                    kind text not null,
                    level integer not null,
                    support integer not null,
                    reward_mean real not null,
                    payload text not null,
                    updated_at text not null
                );

                create table if not exists thoughts (
                    id integer primary key autoincrement,
                    kind text not null,
                    content text not null,
                    payload text not null,
                    created_at text not null
                );

                create table if not exists chat_messages (
                    id integer primary key autoincrement,
                    role text not null,
                    content text not null,
                    created_at text not null
                );

                create table if not exists experiments (
                    id integer primary key autoincrement,
                    status text not null,
                    action text not null,
                    uncertainty real not null,
                    prediction text not null,
                    result text not null,
                    created_at text not null
                );

                create table if not exists plans (
                    id integer primary key autoincrement,
                    goal text not null,
                    score real not null,
                    actions text not null,
                    payload text not null,
                    created_at text not null
                );

                create table if not exists semantic_frames (
                    id integer primary key autoincrement,
                    source text not null,
                    speech_act text not null,
                    topic text not null,
                    intent text not null,
                    confidence real not null,
                    uncertainty real not null,
                    original_text text not null,
                    payload text not null,
                    created_at text not null
                );

                create table if not exists self_modifications (
                    id integer primary key autoincrement,
                    proposal_id text not null,
                    kind text not null,
                    target text not null,
                    status text not null,
                    payload text not null,
                    created_at text not null
                );

                create table if not exists quarantine (
                    id integer primary key autoincrement,
                    entry_id text not null,
                    proposal_id text not null,
                    kind text not null,
                    status text not null,
                    description text not null,
                    snapshot_id text not null,
                    payload text not null,
                    created_at text not null
                );

                create table if not exists snapshots (
                    id integer primary key autoincrement,
                    snapshot_id text not null,
                    gate_identity text not null,
                    content_hash text not null,
                    payload text not null,
                    created_at text not null
                );

                create table if not exists gate_history (
                    id integer primary key autoincrement,
                    old_gate_id text not null,
                    new_gate_id text not null,
                    shadow_agreement real not null,
                    shadow_sample_size integer not null,
                    notes text not null,
                    created_at text not null
                );
                """
            )
            # Forward-compatible track column (used by v7 private simulation tracks).
            # Add only if missing so existing databases upgrade in place.
            for table in ("transitions", "thoughts", "self_modifications"):
                self._add_column_if_missing(connection, table, "track", "text default 'public'")
            connection.commit()

    def _add_column_if_missing(
        self, connection: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        rows = connection.execute(f"pragma table_info({table})").fetchall()
        existing = {row[1] for row in rows}
        if column not in existing:
            connection.execute(f"alter table {table} add column {column} {definition}")
