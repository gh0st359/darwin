"""FilesystemTool — bounded read/write/list inside a sandbox root."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from darwin.tools.base import (
    SandboxEscape,
    Tool,
    ToolResult,
    resolve_sandboxed,
)
from darwin.types import Action


class FilesystemTool(Tool):
    """Read, write, list, and remove files inside a single sandbox root.

    No path traversal is allowed; symlinks resolve through the sandbox.
    File size is bounded on read (default 256 KiB) and on write (default
    1 MiB). Removal is single-file only; recursive directory deletion is
    not exposed at all.
    """

    name = "filesystem"
    description = "Read, write, list, and remove files inside the sandbox."

    def __init__(
        self,
        sandbox_root: str | Path,
        *,
        max_read_bytes: int = 256 * 1024,
        max_write_bytes: int = 1024 * 1024,
    ) -> None:
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    def actions(self) -> list[Action]:
        return [
            Action("fs_read", cost=0.0, description="read a file from the sandbox"),
            Action("fs_write", cost=0.0, description="write a file inside the sandbox"),
            Action("fs_list", cost=0.0, description="list directory entries inside the sandbox"),
            Action("fs_remove", cost=0.0, description="remove a single file inside the sandbox"),
            Action("fs_stat", cost=0.0, description="stat a file inside the sandbox"),
        ]

    def execute(self, input: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        action = str(input.get("action", "")).lower()
        path = input.get("path", "")
        try:
            if action in ("fs_read", "read"):
                return self._do_read(started, path, input)
            if action in ("fs_write", "write"):
                return self._do_write(started, path, input)
            if action in ("fs_list", "list"):
                return self._do_list(started, path, input)
            if action in ("fs_remove", "remove"):
                return self._do_remove(started, path, input)
            if action in ("fs_stat", "stat"):
                return self._do_stat(started, path, input)
            return self._wrap(
                action or "fs_unknown", started, False, "",
                error=f"unknown filesystem action {action!r}",
                input_payload=input,
            )
        except SandboxEscape as exc:
            return self._wrap(
                action or "fs_error", started, False, "",
                error=f"sandbox escape: {exc}",
                input_payload=input,
            )
        except OSError as exc:
            return self._wrap(
                action or "fs_error", started, False, "",
                error=f"{type(exc).__name__}: {exc}",
                input_payload=input,
            )

    # -- per-action ---------------------------------------------------------

    def _do_read(self, started: float, path: str, input: dict[str, Any]) -> ToolResult:
        target = resolve_sandboxed(self.sandbox_root, path)
        if not target.exists() or not target.is_file():
            return self._wrap(
                "fs_read", started, False, "",
                error=f"{path!r} does not exist or is not a file",
                input_payload=input,
            )
        size = target.stat().st_size
        if size > self.max_read_bytes:
            return self._wrap(
                "fs_read", started, False, "",
                error=f"file too large ({size} bytes > {self.max_read_bytes})",
                input_payload=input,
                metadata={"size": size},
            )
        with target.open("rb") as handle:
            data = handle.read(self.max_read_bytes)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        return self._wrap(
            "fs_read", started, True, text,
            input_payload=input,
            metadata={"size": size, "encoding": "utf-8"},
        )

    def _do_write(self, started: float, path: str, input: dict[str, Any]) -> ToolResult:
        target = resolve_sandboxed(self.sandbox_root, path)
        content = input.get("content", "")
        if isinstance(content, bytes):
            data = content
        else:
            data = str(content).encode("utf-8")
        if len(data) > self.max_write_bytes:
            return self._wrap(
                "fs_write", started, False, "",
                error=f"payload too large ({len(data)} > {self.max_write_bytes})",
                input_payload=input,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.write(data)
        return self._wrap(
            "fs_write", started, True, f"wrote {len(data)} bytes to {path}",
            input_payload=input,
            metadata={"bytes": len(data)},
        )

    def _do_list(self, started: float, path: str, input: dict[str, Any]) -> ToolResult:
        target = resolve_sandboxed(self.sandbox_root, path or ".")
        if not target.exists() or not target.is_dir():
            return self._wrap(
                "fs_list", started, False, "",
                error=f"{path!r} does not exist or is not a directory",
                input_payload=input,
            )
        entries: list[str] = []
        for entry in sorted(target.iterdir()):
            kind = "d" if entry.is_dir() else "f"
            entries.append(f"{kind} {entry.name}")
        return self._wrap(
            "fs_list", started, True, "\n".join(entries),
            input_payload=input,
            metadata={"count": len(entries)},
        )

    def _do_remove(self, started: float, path: str, input: dict[str, Any]) -> ToolResult:
        target = resolve_sandboxed(self.sandbox_root, path)
        if not target.exists():
            return self._wrap(
                "fs_remove", started, False, "",
                error=f"{path!r} does not exist",
                input_payload=input,
            )
        if target.is_dir():
            return self._wrap(
                "fs_remove", started, False, "",
                error=(
                    "recursive directory removal is intentionally not exposed; "
                    "remove files individually"
                ),
                input_payload=input,
            )
        target.unlink()
        return self._wrap(
            "fs_remove", started, True, f"removed {path}",
            input_payload=input,
        )

    def _do_stat(self, started: float, path: str, input: dict[str, Any]) -> ToolResult:
        target = resolve_sandboxed(self.sandbox_root, path)
        if not target.exists():
            return self._wrap(
                "fs_stat", started, False, "",
                error=f"{path!r} does not exist",
                input_payload=input,
            )
        st = target.stat()
        kind = "directory" if target.is_dir() else "file"
        return self._wrap(
            "fs_stat", started, True,
            f"{kind} size={st.st_size} mtime={st.st_mtime:.1f}",
            input_payload=input,
            metadata={"size": st.st_size, "mtime": st.st_mtime, "kind": kind},
        )
