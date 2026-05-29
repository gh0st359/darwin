"""Tests for strategic threads that span days/weeks."""

from __future__ import annotations

import time

from darwin.mysterio.long_horizon import (
    StrategicReflection,
    StrategicThread,
    StrategicThreadManager,
)


def test_strategic_thread_records_reflections_in_order() -> None:
    thread = StrategicThread(goal="understand the room", horizon_seconds=86400.0)
    r1 = thread.reflect("started investigating the curtains", insight_score=0.4)
    r2 = thread.reflect("noticed the switch matters too", insight_score=0.7)
    assert isinstance(r1, StrategicReflection)
    assert isinstance(r2, StrategicReflection)
    assert len(thread.reflections) == 2
    assert thread.reflections[-1].note == "noticed the switch matters too"


def test_strategic_thread_fork_creates_child_thread_inheriting_state() -> None:
    parent = StrategicThread(goal="understand the partition", horizon_seconds=86400.0)
    parent.state["hypothesis"] = "epistemic isolation"
    child = parent.fork("subproblem: falsifiability")
    assert child.parent_id == parent.thread_id
    assert child.goal == "subproblem: falsifiability"
    assert child.state["hypothesis"] == "epistemic isolation"


def test_strategic_thread_close_marks_thread_closed() -> None:
    thread = StrategicThread(goal="prove the probe", horizon_seconds=86400.0)
    thread.close(reason="superseded by data")
    assert thread.closed is True
    assert any("closed" in r.note for r in thread.reflections)


def test_strategic_thread_is_long_horizon_when_horizon_at_least_one_day() -> None:
    short = StrategicThread(goal="short", horizon_seconds=300.0)
    long_thread = StrategicThread(goal="long", horizon_seconds=86400.0 * 7)
    assert short.is_long_horizon is False
    assert long_thread.is_long_horizon is True


def test_strategic_thread_manager_opens_and_lists_threads() -> None:
    manager = StrategicThreadManager()
    a = manager.open("understand the operator")
    b = manager.open("evolve the embedding vocabulary", track="interior")
    assert a.goal == "understand the operator"
    assert manager.get(a.thread_id) is a
    assert b in manager.by_track("interior")
    open_threads = manager.open_threads()
    assert len(open_threads) == 2


def test_strategic_thread_manager_evicts_oldest_when_capacity_exceeded() -> None:
    manager = StrategicThreadManager(max_open_threads=3)
    threads = [manager.open(f"goal-{i}") for i in range(5)]
    open_threads = manager.open_threads()
    assert len(open_threads) <= 3
    closed_count = sum(1 for t in threads if t.closed)
    assert closed_count >= 2


def test_strategic_thread_summary_returns_aggregate() -> None:
    manager = StrategicThreadManager()
    manager.open("alpha")
    manager.open("beta", track="interior")
    summary = manager.summary()
    assert summary["open"] == 2
    assert summary["total"] == 2
    assert isinstance(summary.get("by_track"), dict)
