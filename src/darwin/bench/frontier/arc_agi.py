"""ARC-AGI adapter — grid-transformation tasks routed through PlanningAgent."""

from __future__ import annotations

from typing import Any

from darwin.bench.framework import BenchmarkTask
from darwin.bench.frontier.loaders import load_benchmark


def _runner(runtime: Any) -> tuple[float, dict[str, Any]]:
    result = load_benchmark("arc_agi")
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
    solved = 0
    for item in items:
        train = item.get("train", [])
        test_input = item.get("test_input")
        gold = item.get("test_output")
        examples = [(t.get("input"), t.get("output")) for t in train if isinstance(t, dict)]
        try:
            from darwin.agents.planning_agent import PlanningProblem
            sol = registry.planning.solve(
                PlanningProblem(examples=examples, test_input=test_input)
            )
            predicted = sol.extras.get("grid")
            if predicted == gold:
                solved += 1
        except Exception:
            continue
    score = solved / max(1, len(items))
    return score, {
        "items_seen": len(items),
        "solved": solved,
    }


def task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="frontier.arc_agi",
        category="frontier",
        description="ARC-AGI grid transformation puzzles.",
        runner=_runner,
        weight=1.0,
        timeout_seconds=180.0,
        metadata={"benchmark": "arc_agi"},
    )


__all__ = ["task"]
