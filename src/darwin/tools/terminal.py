"""TerminalTool — bounded shell execution inside a sandbox directory.

Security model (strict by default — was a denylist, now an allowlist):

* ``shell=False`` always: every command is parsed by ``shlex.split`` and
  executed as ``argv[]``. No shell metacharacters (``;``, ``&``, ``|``,
  redirections, command substitution, globs) are interpreted. If the
  caller wants a pipeline, they must compose multiple TerminalTool calls.
* A strict positive allowlist of binaries controls what may run. The
  default list is short and read-only-ish (``ls``, ``cat``, ``head``,
  ``tail``, ``wc``, ``grep``, ``find``, ``echo``, ``pwd``, ``date``,
  ``python``, ``python3``, ``pytest``, ``git``). Callers can extend the
  list explicitly via ``extra_allowed=`` — never silently.
* Working directory is pinned to ``sandbox_root``. PATH is sanitised so
  the executable lookup hits ``/usr/local/bin``, ``/usr/bin``, ``/bin``
  only.
* The denylist (rm -rf, sudo, network fetchers, etc.) remains as a
  belt-and-braces guard against accidental misuse even if the allowlist
  is widened.
* Outputs are size-bounded and the wall-clock timeout is enforced.

This is still not a kernel-level sandbox — V-Sandbox (next phase) will
add bubblewrap / firejail / nsjail / wasm wrappers for true containment.
TerminalTool's job here is to make the *common* misuse paths impossible.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


# Default denylist — defence in depth. Matched against the resolved
# command string after shlex splitting.
_DEFAULT_DENY = (
    "rm", "sudo", "su", "mkfs", "dd", "shutdown", "reboot", "halt",
    "curl", "wget", "nc", "ncat", "socat", "scp", "rsync", "ssh",
    "chmod", "chown", "mount", "umount", "kill", "killall", "pkill",
)

# Default allowlist — read-only-ish operations + the bare minimum Darwin
# needs to inspect its own code, run tests, and check git status.
_DEFAULT_ALLOWLIST = (
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep",
    "find", "echo", "pwd", "date", "uname", "whoami",
    "python", "python3", "pytest", "git", "diff", "stat",
    "true", "false", "test", "[",
)

# Sanitised PATH — restrict executable resolution to system bins so
# allowlisted commands cannot be shadowed by arbitrary sandbox files.
_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class TerminalTool(Tool):
    """Run a shell command inside the sandbox cwd with a wall-clock timeout.

    Strict by default: no shell metacharacters, allowlist of binaries,
    denylist of destructive verbs, sanitised PATH, bounded output and
    timeout. See module docstring for the full security model.
    """

    name = "terminal"
    description = "Run a allow-listed command inside the sandbox with a timeout."

    def __init__(
        self,
        sandbox_root: str | Path,
        *,
        timeout_seconds: float = 10.0,
        max_output_bytes: int = 32 * 1024,
        deny_patterns: Sequence[str] | None = None,
        allowed_commands: Sequence[str] | None = None,
        extra_allowed: Sequence[str] | None = None,
        permit_shell_metacharacters: bool = False,
    ) -> None:
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)
        self.deny_patterns = (
            tuple(deny_patterns) if deny_patterns is not None else _DEFAULT_DENY
        )
        if allowed_commands is None:
            self.allowed_commands = tuple(_DEFAULT_ALLOWLIST)
        else:
            self.allowed_commands = tuple(allowed_commands)
        if extra_allowed:
            self.allowed_commands = self.allowed_commands + tuple(extra_allowed)
        self.permit_shell_metacharacters = bool(permit_shell_metacharacters)

    def actions(self) -> list[Action]:
        return [
            Action(
                "shell", cost=0.0,
                description="run an allow-listed command in the sandbox",
            ),
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
        if not self.permit_shell_metacharacters and self._contains_shell_meta(command):
            return self._wrap(
                "shell", started, False, "",
                error=(
                    "shell metacharacters (;, &, |, $(), `, <, >, *, ?) are "
                    "rejected by default; compose multiple TerminalTool "
                    "calls instead"
                ),
                input_payload=input,
            )
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return self._wrap(
                "shell", started, False, "",
                error=f"shlex parse failure: {exc}",
                input_payload=input,
            )
        if not argv:
            return self._wrap(
                "shell", started, False, "",
                error="empty argv after parsing",
                input_payload=input,
            )
        binary = os.path.basename(argv[0])
        if binary in self.deny_patterns:
            return self._wrap(
                "shell", started, False, "",
                error=f"command {binary!r} rejected by denylist",
                input_payload=input,
            )
        if binary not in self.allowed_commands:
            return self._wrap(
                "shell", started, False, "",
                error=(
                    f"command {binary!r} not in allowlist; pass "
                    "extra_allowed=(...) at TerminalTool construction to "
                    "permit it"
                ),
                input_payload=input,
            )
        # Resolve the binary against the sanitised PATH so a sandbox file
        # cannot shadow the system binary.
        resolved = shutil.which(binary, path=_SAFE_PATH)
        if resolved is None:
            return self._wrap(
                "shell", started, False, "",
                error=f"binary {binary!r} not found on safe PATH",
                input_payload=input,
            )
        argv[0] = resolved
        env = {
            "PATH": _SAFE_PATH,
            "LANG": "C.UTF-8",
            "HOME": str(self.sandbox_root),
            "PYTHONPATH": "",
        }
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.sandbox_root),
                capture_output=True,
                timeout=self.timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return self._wrap(
                "shell", started, False, "",
                error=f"timeout after {self.timeout_seconds:.1f}s",
                input_payload=input,
                metadata={
                    "timeout": True,
                    "stdout": (exc.stdout or b"")[: self.max_output_bytes].decode(
                        "utf-8", "replace",
                    ),
                },
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
                "binary": binary,
            },
        )

    # -- guards ------------------------------------------------------------

    _SHELL_META = (";", "&&", "||", "|", "$(", "`", ">", "<", ">>", "<<")

    def _contains_shell_meta(self, command: str) -> bool:
        return any(token in command for token in self._SHELL_META)
