"""Tests for TaskExecutor."""

from __future__ import annotations

from types import SimpleNamespace

from darwin.agents.registry import AgentRegistry
from darwin.autonomy.executor import TaskExecutor
from darwin.autonomy.goal import TaskNode, TaskStatus


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(agent_registry=AgentRegistry())


def test_math_task_executes_successfully() -> None:
    executor = TaskExecutor(_runtime())
    task = TaskNode(
        task_id="t1", goal_id="g1",
        description="What is 2 + 2?",
        agent_name="math",
        payload={"problem": "What is 2 + 2?"},
    )
    report = executor.execute(task)
    assert report.status == TaskStatus.DONE
    assert task.status == TaskStatus.DONE
    assert task.result_summary == "4"


def test_failed_task_retries_then_blocks() -> None:
    runtime = SimpleNamespace(agent_registry=None)
    executor = TaskExecutor(runtime)
    task = TaskNode(
        task_id="t1", goal_id="g1",
        description="anything",
        agent_name="math",
        max_attempts=2,
    )
    executor.execute(task)
    assert task.status == TaskStatus.FAILED
    executor.execute(task)
    assert task.status == TaskStatus.BLOCKED


def test_dialogue_task_falls_back_cleanly() -> None:
    executor = TaskExecutor(_runtime())
    task = TaskNode(
        task_id="t1", goal_id="g1",
        description="hello",
        agent_name="dialogue",
        payload={"message": "hello"},
    )
    report = executor.execute(task)
    assert report.status == TaskStatus.DONE


def test_unknown_agent_marks_task_failed() -> None:
    executor = TaskExecutor(_runtime())
    task = TaskNode(
        task_id="t1", goal_id="g1",
        description="x",
        agent_name="quantum_clairvoyant",
    )
    report = executor.execute(task)
    assert report.status == TaskStatus.FAILED
    assert "unavailable" in report.error
