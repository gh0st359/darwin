"""Tests for GoalOrchestrator end-to-end."""

from __future__ import annotations

from types import SimpleNamespace

from darwin.agents.registry import AgentRegistry
from darwin.autonomy.ledger import GoalLedger
from darwin.autonomy.goal import GoalStatus
from darwin.autonomy.orchestrator import GoalOrchestrator


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(agent_registry=AgentRegistry())


def test_submit_creates_goal_and_persists_tasks() -> None:
    ledger = GoalLedger()
    orch = GoalOrchestrator(_runtime(), ledger=ledger)
    goal = orch.submit("Write a function that sums a list.")
    assert ledger.goal(goal.goal_id) is not None
    assert len(ledger.tasks_for(goal.goal_id)) == 2


def test_dialogue_only_goal_runs_to_completion() -> None:
    ledger = GoalLedger()
    orch = GoalOrchestrator(_runtime(), ledger=ledger)
    goal = orch.submit("hello there friend")  # falls back to single dialogue task
    report = orch.run(goal, max_cycles=4)
    assert report.final_status == GoalStatus.SUCCEEDED
    assert len(report.tasks_completed) == 1


def test_solve_list_completes_all() -> None:
    ledger = GoalLedger()
    orch = GoalOrchestrator(_runtime(), ledger=ledger)
    goal = orch.submit(
        "Solve the following:\n- What is 2 + 2?\n- What is 5 * 5?\n",
    )
    report = orch.run(goal, max_cycles=10)
    # All math tasks succeed.
    assert report.final_status == GoalStatus.SUCCEEDED
    assert len(report.tasks_completed) >= 2


def test_dependencies_respected_during_execution() -> None:
    from darwin.autonomy.goal import TaskStatus
    ledger = GoalLedger()
    orch = GoalOrchestrator(_runtime(), ledger=ledger)
    goal = orch.submit("Research neural plasticity briefly.")
    orch.run(goal, max_cycles=20)
    tasks = ledger.tasks_for(goal.goal_id)
    # If any task ran (DONE), all of its declared dependencies must also be DONE.
    for t in tasks:
        if t.status != TaskStatus.DONE:
            continue
        for dep in t.depends_on:
            dep_task = ledger.task(dep)
            assert dep_task is not None
            assert dep_task.status == TaskStatus.DONE, (
                f"task {t.task_id} ran with unfinished dep {dep}"
            )


def test_max_cycles_bounds_execution() -> None:
    ledger = GoalLedger()
    orch = GoalOrchestrator(_runtime(), ledger=ledger)
    goal = orch.submit(
        "Solve the following:\n"
        + "\n".join(f"- What is {i} + 1?" for i in range(10)),
    )
    report = orch.run(goal, max_cycles=3)
    assert report.cycles_run <= 3


def test_goal_persists_across_orchestrator_instances() -> None:
    ledger = GoalLedger()
    orch_a = GoalOrchestrator(_runtime(), ledger=ledger)
    goal = orch_a.submit("hello darwin")
    orch_a.run(goal, max_cycles=4)
    fresh_ledger = GoalLedger(path=ledger.path)
    assert fresh_ledger.goal(goal.goal_id) is not None
    assert fresh_ledger.goal(goal.goal_id).status == GoalStatus.SUCCEEDED
