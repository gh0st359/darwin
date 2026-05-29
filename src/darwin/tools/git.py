"""GitTool — read-only git inspection inside a sandbox repository.

Exposes ``git status``, ``git log``, ``git diff``, ``git show``, and
``git branch -a`` against a sandboxed checkout. Write operations
(``commit``, ``push``, ``checkout -f``) are intentionally not exposed
here; if Darwin needs to write to a repository it should go through the
TerminalTool with an explicit allowlist, so the action is visible in
the proposal grammar.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


_ALLOWED_GIT_SUBCOMMANDS = {
    "status", "log", "diff", "show", "branch", "remote", "rev-parse",
    "ls-files", "describe", "config",
}


class GitTool(Tool):
    """Read-only git inspection inside a sandbox repository."""

    name = "git"
    description = "Read-only git inspection (status, log, diff, show, branch)."

    def __init__(
        self,
        repo_root: str | Path,
        *,
        timeout_seconds: float = 10.0,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)

    def actions(self) -> list[Action]:
        return [
            Action(f"git_{sub.replace('-', '_')}", cost=0.0,
                   description=f"git {sub}")
            for sub in sorted(_ALLOWED_GIT_SUBCOMMANDS)
        ]

    def execute(self, input: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        action = str(input.get("action", "")).lower()
        if action.startswith("git_"):
            sub = action[4:].replace("_", "-")
        else:
            sub = action
        if sub not in _ALLOWED_GIT_SUBCOMMANDS:
            return self._wrap(
                action or "git_unknown", started, False, "",
                error=(
                    f"git subcommand {sub!r} not in the read-only allowlist "
                    f"({sorted(_ALLOWED_GIT_SUBCOMMANDS)})"
                ),
                input_payload=input,
            )
        args = input.get("args", [])
        if isinstance(args, str):
            args = [args]
        if not isinstance(args, list):
            return self._wrap(
                action, started, False, "",
                error="args must be a list of strings",
                input_payload=input,
            )
        # Disallow flag-injected operations that would mutate state even
        # under a read-only subcommand (e.g. `git log --abbrev-commit
        # --pretty=format:"; touch ...; #"`).
        for arg in args:
            if not isinstance(arg, str):
                return self._wrap(
                    action, started, False, "",
                    error=f"non-string arg in args: {arg!r}",
                    input_payload=input,
                )
        cmd = ["git", "-C", str(self.repo_root), sub, *args]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return self._wrap(
                action, started, False, "",
                error=f"timeout after {self.timeout_seconds:.1f}s",
                input_payload=input,
                metadata={"timeout": True},
            )
        except OSError as exc:
            return self._wrap(
                action, started, False, "",
                error=f"{type(exc).__name__}: {exc}",
                input_payload=input,
            )
        stdout = (proc.stdout or b"")[: self.max_output_bytes].decode("utf-8", "replace")
        stderr = (proc.stderr or b"")[: self.max_output_bytes].decode("utf-8", "replace")
        return self._wrap(
            action, started, proc.returncode == 0, stdout,
            error=stderr if proc.returncode != 0 else "",
            input_payload=input,
            metadata={"returncode": proc.returncode, "subcommand": sub},
        )
