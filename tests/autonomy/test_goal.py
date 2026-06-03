"""Tests for Goal and TaskNode dataclasses."""

from __future__ import annotations

from darwin.autonomy.goal import Goal, GoalStatus, TaskNode, TaskStatus


def test_goal_factory_assigns_id() -> None:
    goal = Goal.make("Solve the linguistics puzzle.")
    assert goal.goal_id.startswith("g_")
    assert goal.status == GoalStatus.OPEN


def test_task_can_run_requires_dependencies_complete() -> None:
    task = TaskNode(
        task_id="t1", goal_id="g1", description="x",
        agent_name="dialogue", depends_on=["t0"],
    )
    assert task.can_run(set()) is False
    assert task.can_run({"t0"}) is True


def test_task_cannot_run_when_attempts_exhausted() -> None:
    task = TaskNode(
        task_id="t1", goal_id="g1", description="x",
        agent_name="dialogue", max_attempts=2,
    )
    task.attempts = 2
    assert task.can_run(set()) is False


def test_task_mark_updates_state() -> None:
    task = TaskNode(task_id="t1", goal_id="g1", description="x", agent_name="dialogue")
    task.mark(TaskStatus.DONE, summary="ok", confidence=0.9)
    assert task.status == TaskStatus.DONE
    assert task.result_summary == "ok"
    assert task.confidence == 0.9


def test_goal_serialization_roundtrip() -> None:
    goal = Goal.make("Test goal")
    goal.task_ids = ["t1", "t2"]
    record = goal.to_record()
    rehydrated = Goal.from_record(record)
    assert rehydrated.goal_id == goal.goal_id
    assert rehydrated.task_ids == ["t1", "t2"]


def test_task_serialization_roundtrip() -> None:
    task = TaskNode(
        task_id="t1", goal_id="g1", description="solve x",
        agent_name="math", payload={"problem": "1+1"},
        depends_on=["t0"], attempts=1,
    )
    record = task.to_record()
    rehydrated = TaskNode.from_record(record)
    assert rehydrated.task_id == "t1"
    assert rehydrated.payload == {"problem": "1+1"}
    assert rehydrated.depends_on == ["t0"]
