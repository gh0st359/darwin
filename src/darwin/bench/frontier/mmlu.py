"""MMLU adapter — multi-choice questions dispatched to the right agent."""

from __future__ import annotations

from typing import Any

from darwin.bench.framework import BenchmarkTask
from darwin.bench.frontier.loaders import load_benchmark


def _runner(runtime: Any) -> tuple[float, dict[str, Any]]:
    result = load_benchmark("mmlu")
    if not result.loaded:
        return 0.0, {
            "error": f"dataset_not_provisioned:{result.status}",
            "items_seen": 0,
        }
    items = result.items
    if not items:
        return 0.0, {"error": "empty_dataset", "items_seen": 0}
    registry = getattr(runtime, "agent_registry", None)
    if registry is None:
        return 0.0, {"error": "no_agent_registry", "items_seen": len(items)}
    correct = 0
    for item in items:
        question = str(item.get("question", ""))
        choices = list(item.get("choices", []))
        gold = str(item.get("answer", "")).strip().lower()
        subject = str(item.get("subject", "")).lower()
        agent = _route_by_subject(registry, subject)
        try:
            from darwin.agents.science_agent import ScienceProblem
            sol = agent.solve(ScienceProblem(question=question, choices=choices))
            if str(sol.answer).strip().lower() == gold:
                correct += 1
        except Exception:
            continue
    score = correct / max(1, len(items))
    return score, {
        "items_seen": len(items),
        "correct": correct,
    }


def _route_by_subject(registry: Any, subject: str) -> Any:
    if "math" in subject or "algebra" in subject or "calculus" in subject:
        return registry.math
    if "code" in subject or "program" in subject or "computer" in subject:
        return registry.code
    return registry.science


def task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="frontier.mmlu",
        category="frontier",
        description="MMLU multi-choice subject knowledge.",
        runner=_runner,
        weight=1.0,
        timeout_seconds=60.0,
        metadata={"benchmark": "mmlu"},
    )


__all__ = ["task"]
