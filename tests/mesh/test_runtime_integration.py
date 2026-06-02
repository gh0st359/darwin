"""Tests that V-Mesh wires into DarwinRuntime correctly."""

from __future__ import annotations

from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


def _runtime(tmp_path: Path) -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=7)
    adapter = RoomSimulationAdapter(world)
    store = PersistentStore(tmp_path / "memory.sqlite3")
    actions = ensure_chat_action(adapter.possible_actions())
    darwin = Darwin(actions=actions, store=store, seed=7, exploration_rate=0.1)
    goal = Goal(desired={"room_bright": True})
    return DarwinRuntime(
        darwin=darwin, adapter=adapter, goal=goal, store=store, interval=0.1,
    )


def test_runtime_constructs_mesh_substrate(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert runtime.cortical_mesh is not None
    assert runtime.mesh_coupling is not None
    assert runtime.mesh_plasticity is not None
    assert runtime.mesh_persistence is not None


def test_mesh_syncs_with_universe_concepts_at_construction(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    universe_summary = runtime.universe.summary()
    # The coupling sync runs in __init__; every universe concept gets a cell.
    assert len(runtime.cortical_mesh) >= universe_summary["concepts"]


def test_chat_does_not_break_when_mesh_is_active(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    # Just calling chat should not raise; mesh should remain consistent.
    reply = runtime.chat("hello")
    assert isinstance(reply, str)
    assert len(runtime.cortical_mesh) >= 1


def test_loop_mesh_runs_without_error(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    event = runtime._loop_mesh()
    assert event is not None
    assert event.kind == "mesh"


def test_mesh_state_persists_across_runtime_construction(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.cortical_mesh.add_cell("custom_marker", salience=2.5)
    runtime.mesh_persistence.save(runtime.cortical_mesh)
    # Build a second runtime sharing the same DARWIN_DATA_DIR (the test
    # isolation fixture pins it for the entire test). The new runtime
    # should load the marker.
    runtime2 = _runtime(tmp_path / "memory_secondary.sqlite3" if False else tmp_path)
    assert runtime2.cortical_mesh.has("custom_marker")
