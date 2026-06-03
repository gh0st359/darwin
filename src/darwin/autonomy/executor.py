"""TaskExecutor — pump a TaskNode through its agent and capture the result.

Dispatches each task to the right V-Agent based on ``task.agent_name``
and ``task.payload``. Records the agent's ``Solution`` back onto the
task: ``result_summary``, ``confidence``, ``status`` (DONE / FAILED /
BLOCKED). The executor never raises into the caller — failures stay
inside the task as a recoverable state so the runner can retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darwin.autonomy.goal import TaskNode, TaskStatus


@dataclass
class ExecutionReport:
    """Outcome of one executor call."""

    task_id: str
    status: TaskStatus
    summary: str = ""
    confidence: float = 0.0
    error: str = ""


class TaskExecutor:
    """Dispatch a TaskNode to its target V-Agent and absorb the result."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def execute(self, task: TaskNode) -> ExecutionReport:
        task.attempts += 1
        task.status = TaskStatus.RUNNING
        registry = (
            getattr(self.runtime, "agent_registry", None) if self.runtime else None
        )
        if registry is None:
            return self._fail(task, "no agent_registry on runtime")
        agent = getattr(registry, task.agent_name, None)
        if agent is None:
            return self._fail(task, f"agent {task.agent_name!r} unavailable")
        problem = self._payload_to_problem(task)
        try:
            solution = agent.solve(problem)
        except Exception as exc:
            return self._fail(task, f"{type(exc).__name__}: {exc}")
        if not solution.succeeded:
            task.mark(TaskStatus.FAILED, summary=solution.notes or "agent did not succeed",
                      confidence=solution.confidence)
            return ExecutionReport(
                task_id=task.task_id, status=task.status,
                summary=task.result_summary, confidence=task.confidence,
                error=solution.notes,
            )
        task.mark(
            TaskStatus.DONE,
            summary=str(solution.answer)[:240],
            confidence=solution.confidence,
        )
        return ExecutionReport(
            task_id=task.task_id, status=task.status,
            summary=task.result_summary, confidence=task.confidence,
        )

    # -- helpers -------------------------------------------------------

    def _fail(self, task: TaskNode, reason: str) -> ExecutionReport:
        if task.attempts >= task.max_attempts:
            task.mark(TaskStatus.BLOCKED, summary=reason)
        else:
            task.mark(TaskStatus.FAILED, summary=reason)
        return ExecutionReport(
            task_id=task.task_id, status=task.status,
            summary=reason, confidence=0.0, error=reason,
        )

    def _payload_to_problem(self, task: TaskNode) -> Any:
        """Translate a generic task payload into the agent's expected problem type."""

        name = task.agent_name
        payload = task.payload or {}
        if name == "code":
            from darwin.agents.code_agent import CodeProblem
            return CodeProblem(
                prompt=str(payload.get("prompt", task.description)),
                function_name=str(payload.get("function_name", "solve")),
                examples=list(payload.get("examples", [])),
            )
        if name == "math":
            return str(payload.get("problem", task.description))
        if name == "science":
            from darwin.agents.science_agent import ScienceProblem
            return ScienceProblem(
                question=str(payload.get("question", task.description)),
                choices=list(payload.get("choices", [])),
            )
        if name == "planning":
            from darwin.agents.planning_agent import PlanningProblem
            return PlanningProblem(
                examples=list(payload.get("examples", [])),
                test_input=payload.get("test_input"),
            )
        if name == "research":
            from darwin.agents.research_agent import ResearchProblem
            return ResearchProblem(
                passage=str(payload.get("passage", "")),
                question=str(payload.get("question", task.description)),
            )
        # dialogue / unknown
        from darwin.agents.dialogue_agent import DialogueProblem
        return DialogueProblem(
            message=str(payload.get("message", task.description)),
        )


__all__ = ["ExecutionReport", "TaskExecutor"]
