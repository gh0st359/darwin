"""ToolWorld — present a ToolRegistry as a v5 World.

Darwin's planner ranks Actions; the World protocol's apply() turns a chosen
Action into a (next_state, reward) pair. ToolWorld wraps a ToolRegistry so
each registered tool action becomes a planner-selectable Action, and a
successful tool execution becomes a positive-reward transition that flows
back into the causal model.

Rewards are intentionally *small and epistemic*: a successful tool call
that produced fresh output earns a modest positive reward; a tool call
that errored earns a slight negative reward. The cognition substrate
learns "doing X tends to succeed when conditions Y" without the planner
ever being told what success "means" in advance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from darwin.tools.base import ToolResult
from darwin.tools.registry import ToolRegistry
from darwin.types import Action, State, Transition


@dataclass
class _Focus:
    last_action: str = ""
    last_success: bool = False
    last_output_size: int = 0
    last_error: str = ""
    last_tool: str = ""


class ToolWorld:
    """Wrap a ToolRegistry as a v5 World.

    The "observation" is a compact summary of the last tool invocation
    plus the current registry inventory. The "actions" are every action
    every registered tool exposes. The "apply" dispatches through the
    registry, builds a Transition, and emits the standard tracking
    metadata (origin=tool, tool name, action name, success).

    ``default_input`` lets callers parametrize the input each Action will
    receive when no per-call payload is supplied: useful for fully
    autonomous tool-driven loops (e.g. always probe the same directory).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        default_input: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.registry = registry
        self.default_input: dict[str, dict[str, Any]] = default_input or {}
        self._focus = _Focus()
        self._step = 0

    # -- World protocol -----------------------------------------------------

    def observe(self) -> State:
        return {
            "tool_step": self._step,
            "last_tool": self._focus.last_tool,
            "last_action": self._focus.last_action,
            "last_success": self._focus.last_success,
            "last_output_size": self._focus.last_output_size,
            "last_error_preview": self._focus.last_error[:120],
            "registered_tool_count": len(self.registry.names()),
        }

    def possible_actions(self) -> list[Action]:
        return self.registry.actions()

    def apply(self, action: Action) -> tuple[State, float]:
        self._step += 1
        payload = dict(self.default_input.get(action.name, {}))
        payload.setdefault("action", action.name)
        result = self.registry.dispatch(action.name, payload)
        reward = self._score(result)
        self._focus = _Focus(
            last_action=action.name,
            last_success=result.success,
            last_output_size=len(result.output or ""),
            last_error=result.error or "",
            last_tool=result.tool or "",
        )
        return self.observe(), reward

    def make_transition(
        self, before: State, after: State, *, reward: float
    ) -> Transition:
        return Transition(
            before=before,
            action=self._focus.last_action,
            after=after,
            reward=reward,
            t=self._step,
            metadata={
                "track": "grounded",
                "origin": "tool",
                "tool": self._focus.last_tool,
                "success": self._focus.last_success,
                "output_size": self._focus.last_output_size,
            },
        )

    # -- helpers -----------------------------------------------------------

    def _score(self, result: ToolResult) -> float:
        if not result.success:
            return -0.05
        # Small positive reward for any successful tool call, with a tiny
        # bonus for non-empty output (it conveyed information).
        bonus = 0.0
        if result.output:
            bonus = min(0.1, len(result.output) / 4000.0)
        return 0.05 + bonus

    # -- introspection -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "registry": self.registry.summary(),
            "focus": self._focus.__dict__,
        }
