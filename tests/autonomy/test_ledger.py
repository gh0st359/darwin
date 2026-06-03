"""Tests for GoalLedger persistence."""

from __future__ import annotations

from darwin.autonomy.goal import Goal, GoalStatus, TaskNode, TaskStatus
from darwin.autonomy.ledger import GoalLedger


def test_add_and_retrieve_goal() -> None:
    ledger = GoalLedger()
    goal = Goal.make("Test goal")
    ledger.add_goal(goal)
    assert ledger.goal(goal.goal_id) is goal


def test_save_and_reload_round_trip() -> None:
    ledger = GoalLedger()
    goal = Goal.make("Persist me")
    task = TaskNode(
        task_id="t1", goal_id=goal.goal_id,
        description="x", agent_name="dialogue",
    )
    goal.task_ids = [task.task_id]
    ledger.add_goal(goal)
    ledger.add_task(task)
    ledger.save()
    # Rehydrate.
    fresh = GoalLedger(path=ledger.path)
    assert fresh.goal(goal.goal_id) is not None
    assert fresh.task(task.task_id) is not None
    assert fresh.goal(goal.goal_id).description == "Persist me"


def test_open_goals_filters_completed() -> None:
    ledger = GoalLedger()
    open_g = Goal.make("Open")
    done_g = Goal.make("Done")
    done_g.status = GoalStatus.SUCCEEDED
    ledger.add_goal(open_g)
    ledger.add_goal(done_g)
    open_ids = {g.goal_id for g in ledger.open_goals()}
    assert open_g.goal_id in open_ids
    assert done_g.goal_id not in open_ids


def test_cancel_goal_cascades_to_tasks() -> None:
    ledger = GoalLedger()
    goal = Goal.make("Cancel me")
    task = TaskNode(task_id="t1", goal_id=goal.goal_id, description="x", agent_name="dialogue")
    goal.task_ids = [task.task_id]
    ledger.add_goal(goal)
    ledger.add_task(task)
    ledger.cancel_goal(goal.goal_id)
    assert ledger.goal(goal.goal_id).status == GoalStatus.CANCELLED
    assert ledger.task("t1").status == TaskStatus.CANCELLED


def test_summary_counts() -> None:
    ledger = GoalLedger()
    for status in (
        GoalStatus.OPEN, GoalStatus.SUCCEEDED, GoalStatus.FAILED,
    ):
        g = Goal.make(f"goal_{status.value}")
        g.status = status
        ledger.add_goal(g)
    s = ledger.summary()
    assert s["goal_count"] == 3
    assert s["succeeded_goal_count"] == 1
    assert s["failed_goal_count"] == 1
