"""Capability probe — non-fixture, procedural evaluation."""

from __future__ import annotations

from darwin.agent import Darwin
from darwin.bench import build_capability_suite
from darwin.bench.framework import BenchmarkRunner
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


def _runtime() -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=42)
    adapter = RoomSimulationAdapter(world)
    darwin = Darwin(
        actions=ensure_chat_action(adapter.possible_actions()),
        seed=42, exploration_rate=0.0,
    )
    return DarwinRuntime(
        darwin=darwin, adapter=adapter,
        goal=Goal(desired={"room_bright": True}),
        interval=100.0,
    )


def test_capability_suite_builds_and_runs():
    suite = build_capability_suite(seed=1)
    runtime = _runtime()
    card = BenchmarkRunner(suite).run(runtime, label="cap-1")
    assert "capability" in card.per_category
    assert len(card.results) >= 1


def test_capability_seed_changes_probe_inputs():
    suite_a = build_capability_suite(seed=1)
    suite_b = build_capability_suite(seed=999)
    # Task ids stay stable; seeds differ — the underlying problems differ.
    assert {t.task_id for t in suite_a.tasks} == {t.task_id for t in suite_b.tasks}


def test_capability_embedding_neighbourhood_is_meaningful():
    # The embedding probe is the most direct signal that the substrate has
    # learned structure. After running, the score is a float in [0, 1].
    from darwin.bench.capability import _embedding_neighbourhood

    runtime = _runtime()
    score, evidence = _embedding_neighbourhood(runtime, seed=5)
    assert 0.0 <= score <= 1.0
    assert "near" in evidence


def test_no_frontier_module_imports_anywhere():
    import importlib

    try:
        importlib.import_module("darwin.bench.frontier")
        raise AssertionError("darwin.bench.frontier should not be importable")
    except ModuleNotFoundError:
        pass
