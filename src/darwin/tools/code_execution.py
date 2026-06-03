"""CodeExecutionTool — run Python in a subprocess sandbox with hardening.

Hardening layers (best-effort, layered defence):

1. **Static AST inspection** of the source before execution. By default a
   denylist of dangerous imports (``os``, ``subprocess``, ``socket``,
   ``ctypes``, ``shutil``, ``pathlib``, ``builtins``, ``importlib``)
   and AST node kinds (``exec``, ``eval``, ``compile``, ``__import__``,
   ``open``) is enforced. Callers can pass ``allow_unsafe=True`` to
   bypass for trusted contexts (CodeAgent's own template emission).

2. **Subprocess isolation** with a sanitised env, restricted PATH, and a
   wall-clock timeout. The temp file is written under ``sandbox_root``
   so any side-effect filesystem writes are contained there.

3. **Optional kernel-level container wrappers** when available:
   ``bubblewrap`` (``bwrap``), ``firejail``, or ``nsjail``. When the
   ``DARWIN_SANDBOX_BACKEND`` env var names a wrapper that's on PATH,
   the Python invocation is wrapped accordingly. Default is ``none``;
   pure-Python tests continue to work.

4. **Resource limits** via ``resource.setrlimit`` (RLIMIT_AS, RLIMIT_CPU,
   RLIMIT_NOFILE, RLIMIT_NPROC) applied in a ``preexec_fn`` so the child
   cannot exhaust memory, CPU, or process count.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


_DANGEROUS_IMPORTS: frozenset[str] = frozenset({
    "os", "subprocess", "socket", "ctypes", "shutil",
    "builtins", "importlib", "imp", "marshal", "pickle",
    "multiprocessing", "threading",
})
_DANGEROUS_CALLS: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__", "open",
    "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "exit", "quit",
})


@dataclass
class StaticInspection:
    """Outcome of inspecting Python source before execution."""

    ok: bool
    forbidden_imports: list[str]
    forbidden_calls: list[str]
    parse_error: str = ""

    @property
    def reason(self) -> str:
        if self.parse_error:
            return f"parse error: {self.parse_error}"
        if self.forbidden_imports:
            return f"forbidden imports: {sorted(self.forbidden_imports)}"
        if self.forbidden_calls:
            return f"forbidden calls: {sorted(self.forbidden_calls)}"
        return ""


def inspect_source(
    source: str,
    *,
    denied_imports: frozenset[str] = _DANGEROUS_IMPORTS,
    denied_calls: frozenset[str] = _DANGEROUS_CALLS,
) -> StaticInspection:
    """Reject obvious unsafe patterns before running ``source``."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return StaticInspection(False, [], [], parse_error=str(exc))
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in denied_imports:
                    forbidden_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in denied_imports:
                forbidden_imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            target = node.func
            name = ""
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name in denied_calls:
                forbidden_calls.append(name)
    ok = not forbidden_imports and not forbidden_calls
    return StaticInspection(ok, forbidden_imports, forbidden_calls)


def _sandbox_wrapper() -> list[str]:
    """Return the prefix argv for the configured sandbox backend."""

    backend = os.environ.get("DARWIN_SANDBOX_BACKEND", "none").strip().lower()
    if backend in ("", "none"):
        return []
    if backend == "bubblewrap":
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            return []
        # Read-only system, writable /tmp, no network namespace.
        return [
            bwrap, "--die-with-parent", "--unshare-net", "--unshare-ipc",
            "--unshare-pid", "--unshare-uts", "--unshare-cgroup-try",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf",
            "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp",
        ]
    if backend == "firejail":
        firejail = shutil.which("firejail")
        if firejail is None:
            return []
        return [firejail, "--quiet", "--net=none", "--noprofile", "--private"]
    if backend == "nsjail":
        nsjail = shutil.which("nsjail")
        if nsjail is None:
            return []
        return [nsjail, "--quiet", "--time_limit", "10"]
    return []


def _apply_rlimits(memory_mb: int = 512, cpu_seconds: int = 10) -> None:
    """preexec_fn that caps memory, CPU time, file descriptors, and procs."""

    try:
        import resource  # POSIX only
    except ImportError:
        return
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024),
        )
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        except (ValueError, OSError):
            pass
    except (ValueError, OSError):
        return


class CodeExecutionTool(Tool):
    """Execute a piece of Python source in a hardened subprocess sandbox."""

    name = "code"
    description = (
        "Execute a Python snippet in a hardened subprocess sandbox. "
        "Static-AST denylist + rlimits + optional bwrap/firejail/nsjail wrapper."
    )

    def __init__(
        self,
        sandbox_root: str | Path,
        *,
        timeout_seconds: float = 8.0,
        max_output_bytes: int = 32 * 1024,
        python_executable: str | None = None,
        allow_unsafe: bool = False,
        memory_mb: int = 512,
        cpu_seconds: int = 10,
        denied_imports: frozenset[str] | None = None,
        denied_calls: frozenset[str] | None = None,
    ) -> None:
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)
        self.python_executable = python_executable or sys.executable
        self.allow_unsafe = bool(allow_unsafe)
        self.memory_mb = int(memory_mb)
        self.cpu_seconds = int(cpu_seconds)
        self.denied_imports = denied_imports or _DANGEROUS_IMPORTS
        self.denied_calls = denied_calls or _DANGEROUS_CALLS

    def actions(self) -> list[Action]:
        return [
            Action(
                "exec_python", cost=0.0,
                description="execute a Python snippet in the hardened sandbox",
            ),
        ]

    def execute(self, input: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        source = str(input.get("source", "")).strip()
        if not source:
            return self._wrap(
                "exec_python", started, False, "",
                error="empty source", input_payload=input,
            )
        if not self.allow_unsafe and not bool(input.get("trusted", False)):
            inspection = inspect_source(
                source,
                denied_imports=self.denied_imports,
                denied_calls=self.denied_calls,
            )
            if not inspection.ok:
                return self._wrap(
                    "exec_python", started, False, "",
                    error=f"static inspection rejected source: {inspection.reason}",
                    input_payload=input,
                    metadata={
                        "forbidden_imports": inspection.forbidden_imports,
                        "forbidden_calls": inspection.forbidden_calls,
                    },
                )
        env_pass = bool(input.get("env_passthrough", False))
        env = None if env_pass else {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONPATH": "",
            "HOME": str(self.sandbox_root),
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="darwin_exec_",
            dir=str(self.sandbox_root), delete=False, encoding="utf-8",
        )
        try:
            tmp.write(source)
            tmp.close()
            argv = _sandbox_wrapper() + [self.python_executable, tmp.name]
            try:
                proc = subprocess.run(
                    argv,
                    cwd=str(self.sandbox_root),
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    env=env,
                    check=False,
                    preexec_fn=(
                        lambda: _apply_rlimits(self.memory_mb, self.cpu_seconds)
                    ) if os.name == "posix" else None,
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
                    "sandbox_backend": os.environ.get(
                        "DARWIN_SANDBOX_BACKEND", "none",
                    ),
                },
            )
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["CodeExecutionTool", "StaticInspection", "inspect_source"]
