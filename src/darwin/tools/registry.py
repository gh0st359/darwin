"""ToolRegistry — central wiring of tools into the runtime.

A Darwin runtime carries one ToolRegistry. Tools register themselves
(or are registered by the brain bootstrap) under their canonical
``name``. The registry maps action names to (tool, default-input-fields)
pairs so the planner can choose an Action and the runtime knows how to
dispatch the execute call.

The registry also exposes a summary so introspection (/tools) can report
what's currently wired in.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


@dataclass
class _ActionBinding:
    tool: Tool
    action_name: str


class ToolRegistry:
    """Thread-safe registry of named tools and their actions."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._bindings: dict[str, _ActionBinding] = {}
        self._lock = threading.RLock()
        self._history: list[ToolResult] = []
        self._history_cap = 256

    def register(self, tool: Tool) -> Tool:
        with self._lock:
            self._tools[tool.name] = tool
            for action in tool.actions():
                self._bindings[action.name] = _ActionBinding(tool=tool, action_name=action.name)
        return tool

    def unregister(self, name: str) -> None:
        with self._lock:
            tool = self._tools.pop(name, None)
            if tool is None:
                return
            for action_name, binding in list(self._bindings.items()):
                if binding.tool is tool:
                    self._bindings.pop(action_name, None)

    def tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def tool_for_action_exists(self, action_name: str) -> bool:
        return action_name in self._bindings

    def actions(self) -> list[Action]:
        out: list[Action] = []
        with self._lock:
            for tool in self._tools.values():
                out.extend(tool.actions())
        return out

    def names(self) -> list[str]:
        return list(self._tools)

    def dispatch(self, action_name: str, input: dict[str, Any]) -> ToolResult:
        binding = self._bindings.get(action_name)
        if binding is None:
            return ToolResult(
                success=False,
                output="",
                tool="(unbound)",
                action=action_name,
                error=f"no tool registered for action {action_name!r}",
                input=dict(input),
            )
        payload = dict(input)
        payload.setdefault("action", action_name)
        result = binding.tool.execute(payload)
        with self._lock:
            self._history.append(result)
            if len(self._history) > self._history_cap:
                self._history = self._history[-self._history_cap:]
        return result

    def history(self, limit: int = 16) -> list[ToolResult]:
        with self._lock:
            return list(self._history[-limit:])

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tools": [
                    {"name": tool.name, "description": tool.description,
                     "actions": [a.name for a in tool.actions()]}
                    for tool in self._tools.values()
                ],
                "history_size": len(self._history),
            }
