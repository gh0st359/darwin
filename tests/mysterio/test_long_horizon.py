"""Multi-week strategic threads + theory-of-mind cascade."""

from __future__ import annotations

from darwin.mysterio.long_horizon import StrategicThreadManager
from darwin.mysterio.observer_cascade import ObserverCascade
from darwin.mysterio.observer_modeler import ObserverWorld


def test_open_thread_records_goal_and_horizon() -> None:
    mgr = StrategicThreadManager()
    t = mgr.open("understand the curtains hypothesis", horizon_seconds=86400 * 3)
    assert t.thread_id
    assert t.is_long_horizon
    assert t.goal.startswith("understand")
    assert not t.closed


def test_reflect_and_fork() -> None:
    mgr = StrategicThreadManager()
    parent = mgr.open("root", horizon_seconds=86400)
    parent.reflect("noticed prediction failure on room_bright", failures=4)
    child = parent.fork("isolate the curtains factor", horizon_seconds=3600)
    assert child.parent_id == parent.thread_id
    assert child.horizon_seconds == 3600
    assert parent.reflections[0].metrics["failures"] == 4


def test_score_threads_rewards_reflection_density() -> None:
    mgr = StrategicThreadManager()
    busy = mgr.open("busy", horizon_seconds=86400)
    quiet = mgr.open("quiet", horizon_seconds=86400)
    for i in range(10):
        busy.reflect(f"step {i}")
    mgr.score_threads()
    assert busy.score >= quiet.score


def test_evict_under_budget() -> None:
    mgr = StrategicThreadManager(max_open_threads=3)
    for i in range(5):
        mgr.open(f"thread {i}", horizon_seconds=86400)
    open_now = mgr.open_threads()
    assert len(open_now) <= 3


def test_observer_cascade_dampens_with_depth() -> None:
    world = ObserverWorld()
    world.note_command("/mind")
    cascade = ObserverCascade(world, max_depth=4)
    cascade.step()
    # attention_level should be non-increasing with depth.
    levels = cascade.levels
    for i in range(1, len(levels)):
        assert levels[i].entity.attention_level <= levels[i - 1].entity.attention_level + 1e-9


def test_observer_cascade_grow() -> None:
    world = ObserverWorld()
    cascade = ObserverCascade(world, max_depth=2)
    cascade.grow(by=3)
    assert cascade.max_depth == 5
    snap = cascade.snapshot()
    assert len(snap["levels"]) == 5
