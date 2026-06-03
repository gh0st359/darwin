"""HumanEval adapter — code-generation tasks routed through CodeAgent."""

from __future__ import annotations

from typing import Any

from darwin.bench.framework import BenchmarkTask
from darwin.bench.frontier.loaders import load_benchmark


def _runner(runtime: Any) -> tuple[float, dict[str, Any]]:
    result = load_benchmark("humaneval")
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
    passed = 0
    for item in items:
        prompt = str(item.get("prompt", ""))
        func_name = str(item.get("function_name", "solve"))
        examples = list(item.get("examples", []))
        normalised = [(p[0], p[1])
                      for p in examples if isinstance(p, (list, tuple)) and len(p) == 2]
        try:
            from darwin.agents.code_agent import CodeProblem
            problem = CodeProblem(
                prompt=prompt, function_name=func_name, examples=normalised,
            )
            sol = registry.code.solve(problem)
            if sol.succeeded:
                passed += 1
        except Exception:
            continue
    score = passed / max(1, len(items))
    return score, {
        "items_seen": len(items),
        "passed": passed,
    }


def task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="frontier.humaneval",
        category="frontier",
        description="HumanEval Python function synthesis.",
        runner=_runner,
        weight=1.0,
        timeout_seconds=120.0,
        metadata={"benchmark": "humaneval"},
    )


__all__ = ["task"]
