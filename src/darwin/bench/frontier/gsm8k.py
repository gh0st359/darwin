"""GSM8K adapter — grade-school math word problems routed through MathAgent."""

from __future__ import annotations

import re
from typing import Any

from darwin.bench.framework import BenchmarkTask
from darwin.bench.frontier.loaders import load_benchmark


_NUMERIC_RX = re.compile(r"-?\d+(?:\.\d+)?")


def _runner(runtime: Any) -> tuple[float, dict[str, Any]]:
    result = load_benchmark("gsm8k")
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
        gold = _extract_numeric(str(item.get("answer", "")))
        try:
            sol = registry.math.solve(question)
            predicted = _extract_numeric(sol.answer)
            if predicted is not None and gold is not None and abs(predicted - gold) < 1e-6:
                correct += 1
        except Exception:
            continue
    score = correct / max(1, len(items))
    return score, {
        "items_seen": len(items),
        "correct": correct,
    }


def _extract_numeric(text: str) -> float | None:
    matches = _NUMERIC_RX.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="frontier.gsm8k",
        category="frontier",
        description="GSM8K grade-school arithmetic word problems.",
        runner=_runner,
        weight=1.0,
        timeout_seconds=60.0,
        metadata={"benchmark": "gsm8k"},
    )


__all__ = ["task"]
