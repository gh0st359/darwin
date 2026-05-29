"""Tool harness — the contract every real-world adapter implements.

Darwin's tools are sandboxed adapters to real-world capabilities:
filesystem, terminal, code execution, web fetching, git, sqlite. Each
tool exposes one or more named *actions*; the planner can choose any
action exactly as it chooses any other v5 Action; the result of an
``execute`` call becomes a Transition that flows back into the causal
model, the memory tiers, and (when used inside chat) the discourse
plan.

Every tool is bounded:

  * **Timeouts.** Every execute has a max wall-clock duration; on
    expiry the tool returns a failure ToolResult and never blocks the
    cognition loop.
  * **Sandbox roots.** Filesystem-touching tools take a sandbox root
    and resolve every supplied path through it; any path that escapes
    the sandbox raises ``SandboxEscape`` and never touches disk.
  * **Provenance.** Every ToolResult carries the tool name, the input,
    the output (truncated to a bounded size), and the elapsed time.
    The runtime records this as a Transition with
    ``metadata["origin"] = "tool"`` so downstream observers know the
    transition is real-world, not simulated.

Tools never bypass the meta-gate or quarantine. A self-modification
proposed by Darwin still goes through ``SelfModificationEngine`` before
landing; tools are how Darwin *gains experience*, not how it edits
itself.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darwin.types import Action


class SandboxEscape(Exception):
    """Raised when a tool input would escape its declared sandbox root."""


class ToolError(Exception):
    """Generic tool failure (timeout, missing dependency, malformed input).

    Tools catch this internally and return a failed ToolResult; it is
    raised only when callers explicitly invoke the unsafe variant.
    """


@dataclass
class ToolResult:
    """Outcome of one tool invocation."""

    success: bool
    output: str
    tool: str = ""
    action: str = ""
    error: str = ""
    duration_ms: float = 0.0
    input: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tool": self.tool,
            "action": self.action,
            "output": self.output[:2000],
            "error": self.error[:1000],
            "duration_ms": round(self.duration_ms, 2),
            "input": dict(self.input),
            "metadata": dict(self.metadata),
        }


def resolve_sandboxed(root: Path, candidate: str | Path) -> Path:
    """Return ``candidate`` resolved against ``root`` if it stays inside it.

    Raises ``SandboxEscape`` on traversal, absolute paths that fall
    outside, or symlink resolution that exits the sandbox.
    """

    root = Path(root).resolve()
    raw = Path(candidate)
    base = (root / raw) if not raw.is_absolute() else raw
    try:
        resolved = base.resolve()
    except OSError as exc:
        raise SandboxEscape(f"could not resolve {candidate!r}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SandboxEscape(
            f"{candidate!r} resolves to {resolved}, which is outside the "
            f"sandbox root {root}"
        )
    return resolved


class Tool(ABC):
    """Base class every real-world adapter inherits from."""

    name: str = "tool"
    description: str = ""

    @abstractmethod
    def execute(self, input: dict[str, Any]) -> ToolResult:
        """Run the tool's primary action. Must never raise on input errors;
        return a failed ToolResult instead."""

    def actions(self) -> list[Action]:
        """Map the tool's behavior(s) to v5 Action objects the planner can
        select. Default: a single Action sharing the tool's name and
        description, with cost 0."""

        return [Action(name=self.name, cost=0.0, description=self.description)]

    # -- helpers -----------------------------------------------------------

    def _wrap(self, action_name: str, started: float, success: bool,
              output: str, *, error: str = "",
              input_payload: dict[str, Any] | None = None,
              metadata: dict[str, Any] | None = None) -> ToolResult:
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        return ToolResult(
            success=success,
            output=output,
            tool=self.name,
            action=action_name,
            error=error,
            duration_ms=duration_ms,
            input=dict(input_payload or {}),
            metadata=dict(metadata or {}),
        )
