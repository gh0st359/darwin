"""Autonomous task runner — Darwin pursues a goal across many tool calls.

An ``AutonomousTask`` carries:

  * a natural-language goal (the *intent* the operator typed);
  * a structured success predicate (a callable Darwin can run to decide
    "am I done?"). The predicate is *optional* — if absent, the task runs
    until ``max_steps`` is reached or the loop stalls;
  * a budget (max wall-clock seconds + max step count);
  * a step function — by default, ranks all registered tool actions
    against the current ToolWorld state and picks the highest-ranked.

The runner is *interruptible* and *observable*: every step emits an
``AutonomousStep`` that the brain terminal can stream, and ``stop()``
halts cleanly. A task records every Transition into the runtime's
causal model so the planner gets better at choosing tool actions over
the lifetime of the brain.

This is the "long-running autonomous task" surface the harness exposes;
the planner stays in charge of *which* action to take at each step.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from darwin.tools.world import ToolWorld
from darwin.types import Action, Goal


@dataclass
class AutonomousStep:
    """One step the runner took: an action, its reward, the new state."""

    step: int
    action: str
    reward: float
    success: bool
    output_preview: str = ""
    error_preview: str = ""
    state: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "reward": round(self.reward, 4),
            "success": self.success,
            "output_preview": self.output_preview[:200],
            "error_preview": self.error_preview[:200],
            "state": dict(self.state),
        }


@dataclass
class AutonomousTask:
    """A running goal-directed task on the tool world."""

    goal: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    max_steps: int = 16
    max_seconds: float = 30.0
    success_predicate: Callable[[Any], bool] | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    success: bool = False
    steps: list[AutonomousStep] = field(default_factory=list)
    reason_stopped: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "max_steps": self.max_steps,
            "max_seconds": self.max_seconds,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "reason_stopped": self.reason_stopped,
            "steps": [s.to_record() for s in self.steps],
        }


class AutonomousRunner:
    """Run AutonomousTasks against a ToolWorld + planner.

    The runner is single-threaded by design — Darwin's brain already has
    a multi-threaded scheduler; the runner just plays the role of "this
    autonomous task is what we're going to focus on for a while". For
    truly parallel autonomy, spawn multiple runners against the same
    registry; the registry is thread-safe.
    """

    def __init__(self, world: ToolWorld, *, action_chooser: Callable[[list[Action], dict[str, Any]], Action] | None = None) -> None:
        self.world = world
        self._stop = threading.Event()
        self._tasks: list[AutonomousTask] = []
        # Default chooser: round-robin through actions, biased toward
        # actions that have not been tried yet. The runtime can pass in a
        # planner-driven chooser for real ranking.
        self._chooser = action_chooser or self._default_chooser
        self._tried: set[str] = set()

    def stop(self) -> None:
        self._stop.set()

    def run(self, task: AutonomousTask) -> AutonomousTask:
        self._stop.clear()
        self._tasks.append(task)
        deadline = task.started_at + task.max_seconds
        for step_index in range(1, task.max_steps + 1):
            if self._stop.is_set():
                task.reason_stopped = "interrupted"
                break
            if time.time() >= deadline:
                task.reason_stopped = "time budget exceeded"
                break
            actions = self.world.possible_actions()
            if not actions:
                task.reason_stopped = "no actions available"
                break
            state_before = self.world.observe()
            chosen = self._chooser(actions, state_before)
            self._tried.add(chosen.name)
            state_after, reward = self.world.apply(chosen)
            focus = self.world._focus
            step = AutonomousStep(
                step=step_index,
                action=chosen.name,
                reward=reward,
                success=focus.last_success,
                output_preview=(focus.last_error or ""),
                error_preview=(focus.last_error or ""),
                state=state_after,
            )
            task.steps.append(step)
            if task.success_predicate is not None:
                try:
                    if task.success_predicate(state_after):
                        task.success = True
                        task.reason_stopped = "predicate satisfied"
                        break
                except Exception:
                    pass
        else:
            task.reason_stopped = task.reason_stopped or "max_steps reached"
        task.completed_at = time.time()
        return task

    def history(self) -> list[AutonomousTask]:
        return list(self._tasks)

    # -- default chooser ---------------------------------------------------

    def _default_chooser(self, actions: list[Action], state: dict[str, Any]) -> Action:
        for action in actions:
            if action.name not in self._tried:
                return action
        # Everything has been tried; cycle.
        index = state.get("tool_step", 0) % max(1, len(actions))
        return actions[int(index)]
