"""Tests for ToolRegistry + ToolWorld + AutonomousRunner."""

from __future__ import annotations

from pathlib import Path

from darwin.tools import (
    AutonomousRunner,
    AutonomousTask,
    FilesystemTool,
    ToolRegistry,
    ToolWorld,
)
from darwin.tools.base import Tool, ToolResult
from darwin.types import Action


class _StubTool(Tool):
    name = "stub"
    description = "stub tool for tests"

    def actions(self):
        return [Action("stub_a", cost=0.0), Action("stub_b", cost=0.0)]

    def execute(self, input):
        action = input.get("action", "")
        return ToolResult(
            success=action == "stub_a",
            output="A out" if action == "stub_a" else "",
            tool=self.name,
            action=action,
            error="" if action == "stub_a" else "stub failed",
        )


def test_registry_registers_and_dispatches() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool())
    assert "stub" in registry.names()
    result = registry.dispatch("stub_a", {})
    assert result.success
    assert result.output == "A out"


def test_registry_dispatches_to_unbound_action_returns_failure() -> None:
    registry = ToolRegistry()
    result = registry.dispatch("does_not_exist", {})
    assert not result.success
    assert "no tool registered" in result.error


def test_registry_history_bounded() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool())
    for _ in range(300):
        registry.dispatch("stub_a", {})
    history = registry.history(limit=512)
    assert len(history) <= 256


def test_registry_summary_lists_each_tool() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool())
    summary = registry.summary()
    assert summary["tools"]
    assert summary["tools"][0]["name"] == "stub"


def test_filesystem_tool_via_world_emits_transition(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(FilesystemTool(tmp_path))
    world = ToolWorld(
        registry,
        default_input={
            "fs_write": {"path": "x.txt", "content": "hi"},
            "fs_read": {"path": "x.txt"},
        },
    )
    write_action = next(a for a in world.possible_actions() if a.name == "fs_write")
    state_after, reward = world.apply(write_action)
    assert reward > 0
    assert state_after["last_success"] is True
    before = state_after
    read_action = next(a for a in world.possible_actions() if a.name == "fs_read")
    state_after2, reward2 = world.apply(read_action)
    transition = world.make_transition(before, state_after2, reward=reward2)
    assert transition.metadata["origin"] == "tool"
    assert transition.metadata["tool"] == "filesystem"
    assert transition.metadata["track"] == "grounded"


def test_world_apply_failing_action_yields_negative_reward() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool())
    world = ToolWorld(registry)
    fail_action = next(a for a in world.possible_actions() if a.name == "stub_b")
    state, reward = world.apply(fail_action)
    assert reward < 0
    assert state["last_success"] is False


def test_autonomous_runner_stops_on_predicate(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(FilesystemTool(tmp_path))
    world = ToolWorld(
        registry,
        default_input={
            "fs_write": {"path": "marker.txt", "content": "done"},
        },
    )
    runner = AutonomousRunner(world)
    # Predicate: stop when last action was successful.
    task = AutonomousTask(
        goal="write a marker file",
        max_steps=8,
        success_predicate=lambda state: state.get("last_success") is True,
    )
    runner.run(task)
    assert task.success is True
    assert task.reason_stopped == "predicate satisfied"
    assert task.steps


def test_autonomous_runner_respects_max_steps(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(_StubTool())
    world = ToolWorld(registry)
    runner = AutonomousRunner(world)
    task = AutonomousTask(
        goal="never satisfied",
        max_steps=3,
        success_predicate=lambda state: False,
    )
    runner.run(task)
    assert task.success is False
    assert len(task.steps) == 3
    assert "max_steps" in task.reason_stopped
