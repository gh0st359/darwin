"""GoalLedger — durable storage for Goals and TaskNodes.

The ledger persists every goal and task to ``data_dir() / "darwin_goals.json"``
(routed through the autouse ``DARWIN_DATA_DIR`` fixture in tests). A
session restart rehydrates open goals + their task trees, so long-horizon
work survives process death.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darwin.autonomy.goal import Goal, GoalStatus, TaskNode, TaskStatus
from darwin.paths import data_dir


@dataclass
class GoalLedger:
    """In-memory + on-disk store of goals and tasks."""

    path: Path = field(default_factory=lambda: data_dir() / "darwin_goals.json")
    _goals: dict[str, Goal] = field(default_factory=dict)
    _tasks: dict[str, TaskNode] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self.load()

    # -- mutate --------------------------------------------------------

    def add_goal(self, goal: Goal) -> None:
        self._goals[goal.goal_id] = goal

    def update_goal(self, goal: Goal) -> None:
        self._goals[goal.goal_id] = goal

    def add_task(self, task: TaskNode) -> None:
        self._tasks[task.task_id] = task

    def update_task(self, task: TaskNode) -> None:
        self._tasks[task.task_id] = task

    def cancel_goal(self, goal_id: str) -> None:
        goal = self._goals.get(goal_id)
        if goal is None:
            return
        goal.status = GoalStatus.CANCELLED
        for tid in goal.task_ids:
            task = self._tasks.get(tid)
            if task is not None and task.status in (
                TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED,
            ):
                task.status = TaskStatus.CANCELLED

    # -- query ---------------------------------------------------------

    def goal(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def task(self, task_id: str) -> TaskNode | None:
        return self._tasks.get(task_id)

    def open_goals(self) -> list[Goal]:
        return [
            g for g in self._goals.values()
            if g.status in (GoalStatus.OPEN, GoalStatus.RUNNING)
        ]

    def all_goals(self) -> list[Goal]:
        return list(self._goals.values())

    def tasks_for(self, goal_id: str) -> list[TaskNode]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return []
        return [self._tasks[tid] for tid in goal.task_ids if tid in self._tasks]

    def summary(self) -> dict[str, Any]:
        return {
            "goal_count": len(self._goals),
            "task_count": len(self._tasks),
            "open_goal_count": len(self.open_goals()),
            "succeeded_goal_count": sum(
                1 for g in self._goals.values()
                if g.status == GoalStatus.SUCCEEDED
            ),
            "failed_goal_count": sum(
                1 for g in self._goals.values()
                if g.status == GoalStatus.FAILED
            ),
        }

    # -- persistence ---------------------------------------------------

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "goals": [g.to_record() for g in self._goals.values()],
                "tasks": [t.to_record() for t in self._tasks.values()],
            }
            # Atomic write so concurrent readers never see a half file.
            fd, tmp_path = tempfile.mkstemp(
                prefix=".darwin_goals_", suffix=".json",
                dir=str(self.path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"))
                os.replace(tmp_path, str(self.path))
                return True
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return False
        except OSError:
            return False

    def load(self) -> bool:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        self._goals.clear()
        self._tasks.clear()
        for record in payload.get("goals", []):
            if not isinstance(record, dict):
                continue
            try:
                goal = Goal.from_record(record)
                self._goals[goal.goal_id] = goal
            except Exception:
                continue
        for record in payload.get("tasks", []):
            if not isinstance(record, dict):
                continue
            try:
                task = TaskNode.from_record(record)
                self._tasks[task.task_id] = task
            except Exception:
                continue
        return True


__all__ = ["GoalLedger"]
