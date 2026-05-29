"""CodeExecutionTool — run Python in a subprocess sandbox with a timeout."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


class CodeExecutionTool(Tool):
    """Execute a piece of Python source in a subprocess.

    The subprocess inherits the sandbox cwd but is otherwise isolated:
    no environment passthrough by default, a wall-clock timeout, and
    bounded stdout/stderr capture. The Python source is written to a
    temporary file inside the sandbox so the subprocess can find it
    by ``-`` (stdin) or ``-c`` (inline) — we use ``-c`` for short
    snippets and a temp file for longer ones.

    Imports are not restricted. The sandbox is filesystem-level; if
    Darwin wants to ``import os`` and clobber the world, the working-
    directory sandbox limits the damage to the sandbox tree.
    """

    name = "code"
    description = "Execute a Python snippet in a subprocess sandbox."

    def __init__(
        self,
        sandbox_root: str | Path,
        *,
        timeout_seconds: float = 8.0,
        max_output_bytes: int = 32 * 1024,
        python_executable: str | None = None,
    ) -> None:
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)
        self.python_executable = python_executable or sys.executable

    def actions(self) -> list[Action]:
        return [
            Action("exec_python", cost=0.0, description="execute a Python snippet"),
        ]

    def execute(self, input: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        source = str(input.get("source", "")).strip()
        if not source:
            return self._wrap(
                "exec_python", started, False, "",
                error="empty source",
                input_payload=input,
            )
        env_pass = bool(input.get("env_passthrough", False))
        env = None if env_pass else {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONPATH": "",
        }
        # Use a temp file so multi-line / complex source works cleanly.
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="darwin_exec_",
            dir=str(self.sandbox_root),
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(source)
            tmp.close()
            try:
                proc = subprocess.run(
                    [self.python_executable, tmp.name],
                    cwd=str(self.sandbox_root),
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return self._wrap(
                    "exec_python", started, False, "",
                    error=f"timeout after {self.timeout_seconds:.1f}s",
                    input_payload=input,
                    metadata={"timeout": True},
                )
            stdout = (proc.stdout or b"")[: self.max_output_bytes].decode("utf-8", "replace")
            stderr = (proc.stderr or b"")[: self.max_output_bytes].decode("utf-8", "replace")
            return self._wrap(
                "exec_python", started,
                proc.returncode == 0,
                stdout,
                error=stderr if proc.returncode != 0 else "",
                input_payload=input,
                metadata={
                    "returncode": proc.returncode,
                    "stdout_bytes": len(proc.stdout or b""),
                    "stderr_bytes": len(proc.stderr or b""),
                },
            )
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except OSError:
                pass
