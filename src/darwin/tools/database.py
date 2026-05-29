"""DatabaseTool — bounded sqlite query against a sandbox file."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from darwin.tools.base import Tool, ToolResult, resolve_sandboxed
from darwin.types import Action


_FORBIDDEN_TOKENS = (
    "attach", "detach", "pragma", "vacuum",
    "create trigger", "drop trigger", "load_extension",
)


class DatabaseTool(Tool):
    """SQLite read/write against a single file inside the sandbox.

    The tool refuses ``ATTACH`` / ``DETACH`` / ``LOAD_EXTENSION`` /
    ``PRAGMA`` and the destructive ``VACUUM`` form. Statements are
    classified by the first keyword (SELECT vs everything else) and
    routed accordingly: SELECT through ``fetchall`` with a row cap,
    everything else through ``execute`` followed by ``commit``. The
    caller can specify the database file relative to the sandbox; any
    path that escapes the sandbox is rejected.
    """

    name = "db"
    description = "Run a SQL statement against a sandbox sqlite file."

    def __init__(
        self,
        sandbox_root: str | Path,
        *,
        max_rows: int = 256,
        max_output_chars: int = 16 * 1024,
        statement_timeout_seconds: float = 5.0,
    ) -> None:
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.max_rows = int(max_rows)
        self.max_output_chars = int(max_output_chars)
        self.statement_timeout_seconds = float(statement_timeout_seconds)

    def actions(self) -> list[Action]:
        return [
            Action("db_query", cost=0.0, description="run a SELECT and return rows"),
            Action("db_exec", cost=0.0, description="run a non-SELECT statement and commit"),
        ]

    def execute(self, input: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        action = str(input.get("action", "")).lower()
        path = input.get("path") or "db.sqlite3"
        sql = str(input.get("sql", "")).strip()
        params = input.get("params") or ()
        if not sql:
            return self._wrap(
                action or "db", started, False, "",
                error="empty sql",
                input_payload=input,
            )
        denied = self._denied(sql)
        if denied:
            return self._wrap(
                action or "db", started, False, "",
                error=f"sql rejected: contains forbidden token {denied!r}",
                input_payload=input,
            )
        try:
            target = resolve_sandboxed(self.sandbox_root, path)
        except Exception as exc:
            return self._wrap(
                action or "db", started, False, "",
                error=f"{type(exc).__name__}: {exc}",
                input_payload=input,
            )
        is_select = sql.lstrip().lower().startswith(("select", "with"))
        try:
            with closing(sqlite3.connect(str(target), timeout=self.statement_timeout_seconds)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if is_select:
                    rows = cursor.execute(sql, params).fetchmany(self.max_rows)
                    formatted = self._format_rows(rows)
                    output = formatted
                    if len(output) > self.max_output_chars:
                        output = output[: self.max_output_chars] + "\n... [truncated]"
                    return self._wrap(
                        "db_query", started, True, output,
                        input_payload=input,
                        metadata={"rows": len(rows), "columns": [d[0] for d in (cursor.description or [])]},
                    )
                else:
                    cursor.execute(sql, params)
                    conn.commit()
                    return self._wrap(
                        "db_exec", started, True,
                        f"executed; rowcount={cursor.rowcount}",
                        input_payload=input,
                        metadata={"rowcount": cursor.rowcount},
                    )
        except sqlite3.Error as exc:
            return self._wrap(
                action or "db", started, False, "",
                error=f"sqlite3.{type(exc).__name__}: {exc}",
                input_payload=input,
            )

    def _denied(self, sql: str) -> str:
        lowered = sql.lower()
        for token in _FORBIDDEN_TOKENS:
            if token in lowered:
                return token
        return ""

    def _format_rows(self, rows: list[sqlite3.Row]) -> str:
        if not rows:
            return "(no rows)"
        # Pretty-print as a table with column headers.
        cols = rows[0].keys()
        out_lines: list[str] = [" | ".join(cols)]
        out_lines.append("-+-".join(["-" * len(c) for c in cols]))
        for row in rows:
            out_lines.append(" | ".join(str(row[c]) for c in cols))
        return "\n".join(out_lines)
