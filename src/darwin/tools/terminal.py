"""TerminalTool — bounded shell execution inside a sandbox directory."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


# Default deny-list. The shell is intentionally narrow by default; Darwin
# can be given a wider command surface via the ``allowed_commands`` argument
# at construction time. Anything starting with these tokens is rejected.
_DEFAULT_DENY = (
    "rm -rf", "sudo", "su ", "chmod 777", "chown ", "mkfs", "dd if=",
    "shutdown", "reboot", "halt", ":(){", "| sh", "curl ", "wget ",
    ">>/etc", ">/etc",
)


class TerminalTool(Tool):
    """Run a shell command inside the sandbox cwd with a wall-clock timeout.

    The command runs through ``subprocess.run`` with ``shell=True`` so
    pipelines and redirections work, but a deny-list rejects clearly
    destructive patterns by default. Stdout and stderr are captured and
    truncated to bounded sizes. The process inherits the sandbox cwd as
    its working directory.
    """

    name = "terminal"
    description = "Run a shell command inside the sandbox with a timeout."

    def __init__(
        self,
        sandbox_root: str | Path,
        *,
        timeout_seconds: float = 10.0,
        max_output_bytes: int = 32 * 1024,
        deny_patterns: Sequence[str] | None = None,
        allowed_commands: Sequence[str] | None = None,
    ) -> None:
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)
        self.deny_patterns = tuple(deny_patterns) if deny_patterns is not None else _DEFAULT_DENY
        self.allowed_commands = (
            tuple(allowed_commands) if allowed_commands is not None else None
        )

    def actions(self) -> list[Action]:
        return [
            Action("shell", cost=0.0, description="run a shell command in the sandbox"),
        ]

    def execute(self, input: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        command = str(input.get("command", "")).strip()
        if not command:
            return self._wrap(
                "shell", started, False, "",
                error="empty command",
                input_payload=input,
            )
        denied = self._denied(command)
        if denied:
            return self._wrap(
                "shell", started, False, "",
                error=f"command rejected by deny-list pattern {denied!r}",
                input_payload=input,
            )
        if self.allowed_commands is not None and not self._allowed(command):
            return self._wrap(
                "shell", started, False, "",
                error=(
                    "command not in allowlist; instantiate TerminalTool with a "
                    "broader allowlist to permit this command"
                ),
                input_payload=input,
            )
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.sandbox_root),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return self._wrap(
                "shell", started, False, "",
                error=f"timeout after {self.timeout_seconds:.1f}s",
                input_payload=input,
                metadata={"timeout": True, "stdout": (exc.stdout or b"")[: self.max_output_bytes].decode("utf-8", "replace")},
            )
        except OSError as exc:
            return self._wrap(
                "shell", started, False, "",
                error=f"{type(exc).__name__}: {exc}",
                input_payload=input,
            )
        stdout = (proc.stdout or b"")[: self.max_output_bytes].decode("utf-8", "replace")
        stderr = (proc.stderr or b"")[: self.max_output_bytes].decode("utf-8", "replace")
        return self._wrap(
            "shell", started, proc.returncode == 0, stdout,
            error=stderr if proc.returncode != 0 else "",
            input_payload=input,
            metadata={
                "returncode": proc.returncode,
                "stdout_bytes": len(proc.stdout or b""),
                "stderr_bytes": len(proc.stderr or b""),
            },
        )

    # -- guards ------------------------------------------------------------

    def _denied(self, command: str) -> str:
        lowered = command.lower()
        for pat in self.deny_patterns:
            if pat.lower() in lowered:
                return pat
        return ""

    def _allowed(self, command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False
        first = tokens[0]
        return first in self.allowed_commands
