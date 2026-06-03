"""GoalOrchestrator — drive a Goal to completion across many cycles.

The orchestrator owns:

* A ``GoalDecomposer`` that turns the goal description into a TaskNode
  tree on first run (or rehydrates from disk).
* A ``TaskExecutor`` that runs ready tasks one at a time.
* A retry / blocked-task policy: failed tasks retry up to
  ``max_attempts``; persistent failure marks them BLOCKED and propagates
  to dependants.
* A ``GoalLedger`` (passed in) that persists state after every cycle so
  a fresh runtime can resume.

This is bounded autonomy: each ``run()`` call advances by at most
``max_cycles`` task dispatches and returns. Long-horizon execution is
achieved by repeated calls (the brain loop is one source; a CLI
``darwin work --goal <id>`` is another).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.autonomy.decomposer import GoalDecomposer
from darwin.autonomy.executor import ExecutionReport, TaskExecutor
from darwin.autonomy.goal import Goal, GoalStatus, TaskNode, TaskStatus


@dataclass
class OrchestrationReport:
    """Summary of one orchestrator cycle."""

    goal_id: str
    cycles_run: int = 0
    tasks_completed: list[str] = field(default_factory=list)
    tasks_failed: list[str] = field(default_factory=list)
    final_status: GoalStatus = GoalStatus.OPEN
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "cycles_run": self.cycles_run,
            "tasks_completed": list(self.tasks_completed),
            "tasks_failed": list(self.tasks_failed),
            "final_status": self.final_status.value,
            "notes": self.notes,
        }


class GoalOrchestrator:
    """Coordinate goal-decomposition + task-execution + ledger updates."""

    def __init__(
        self,
        runtime: Any,
        ledger: Any = None,
        *,
        decomposer: GoalDecomposer | None = None,
        executor: TaskExecutor | None = None,
    ) -> None:
        self.runtime = runtime
        self.ledger = ledger
        self.decomposer = decomposer or GoalDecomposer()
        self.executor = executor or TaskExecutor(runtime)

    def submit(self, description: str, *, success_criteria: str = "") -> Goal:
        """Create and persist a goal + its decomposition."""

        goal = Goal.make(description, success_criteria=success_criteria)
        tasks = self.decomposer.decompose(goal, description)
        goal.task_ids = [t.task_id for t in tasks]
        if self.ledger is not None:
            self.ledger.add_goal(goal)
            for task in tasks:
                self.ledger.add_task(task)
            self.ledger.save()
        return goal

    def run(self, goal: Goal, *, max_cycles: int = 12) -> OrchestrationReport:
        """Pump ready tasks through the executor until done or budget exhausted."""

        report = OrchestrationReport(goal_id=goal.goal_id)
        if not goal.task_ids:
            report.final_status = goal.status
            report.notes = "goal has no tasks"
            return report
        if goal.status not in (GoalStatus.OPEN, GoalStatus.RUNNING):
            report.final_status = goal.status
            report.notes = f"goal already {goal.status.value}"
            return report
        goal.status = GoalStatus.RUNNING
        completed: set[str] = self._completed_ids(goal)
        for cycle in range(max_cycles):
            tasks = self._tasks(goal)
            if not tasks:
                break
            ready = [t for t in tasks if t.can_run(completed)]
            if not ready:
                # Check whether everything is done or we're stuck on blocked deps.
                if all(t.status == TaskStatus.DONE for t in tasks):
                    goal.status = GoalStatus.SUCCEEDED
                elif any(t.status == TaskStatus.BLOCKED for t in tasks):
                    goal.status = GoalStatus.FAILED
                    report.notes = "one or more tasks blocked"
                break
            task = self._pick_next(ready)
            exec_report = self.executor.execute(task)
            report.cycles_run += 1
            if exec_report.status == TaskStatus.DONE:
                report.tasks_completed.append(task.task_id)
                completed.add(task.task_id)
            else:
                report.tasks_failed.append(task.task_id)
            if self.ledger is not None:
                self.ledger.update_task(task)
                self.ledger.save()
        # Final status check.
        tasks = self._tasks(goal)
        if all(t.status == TaskStatus.DONE for t in tasks):
            goal.status = GoalStatus.SUCCEEDED
        elif any(t.status == TaskStatus.BLOCKED for t in tasks):
            goal.status = GoalStatus.FAILED
        goal.touch()
        if self.ledger is not None:
            self.ledger.update_goal(goal)
            self.ledger.save()
        report.final_status = goal.status
        return report

    # -- helpers -------------------------------------------------------

    def _tasks(self, goal: Goal) -> list[TaskNode]:
        if self.ledger is None:
            return []
        return [
            t for t in (self.ledger.task(tid) for tid in goal.task_ids)
            if t is not None
        ]

    def _completed_ids(self, goal: Goal) -> set[str]:
        return {
            t.task_id for t in self._tasks(goal)
            if t.status == TaskStatus.DONE
        }

    def _pick_next(self, ready: list[TaskNode]) -> TaskNode:
        # Prefer the task with the most dependants downstream (largest fan-out
        # gets unblocked sooner). Stable by created_at for determinism.
        ready_sorted = sorted(ready, key=lambda t: (t.created_at, t.task_id))
        return ready_sorted[0]


__all__ = ["GoalOrchestrator", "OrchestrationReport"]
