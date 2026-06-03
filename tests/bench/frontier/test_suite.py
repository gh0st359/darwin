"""Tests for the frontier benchmark suite."""

from __future__ import annotations

import json
from types import SimpleNamespace

from darwin.agents.registry import AgentRegistry
from darwin.bench.framework import BenchmarkRunner
from darwin.bench.frontier.suite import build_frontier_suite
from darwin.paths import data_dir


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(agent_registry=AgentRegistry(), universe=None)


def test_suite_assembles_six_tasks() -> None:
    suite = build_frontier_suite()
    assert len(suite.tasks) == 6
    ids = {t.task_id for t in suite.tasks}
    assert ids == {
        "frontier.mmlu",
        "frontier.humaneval",
        "frontier.gpqa",
        "frontier.arc_agi",
        "frontier.gsm8k",
        "frontier.math",
    }


def test_all_tasks_in_frontier_category() -> None:
    suite = build_frontier_suite()
    assert all(t.category == "frontier" for t in suite.tasks)


def test_unprovisioned_runs_score_zero_with_evidence() -> None:
    suite = build_frontier_suite()
    runner = BenchmarkRunner(suite)
    card = runner.run(_runtime(), label="unprovisioned-test")
    # All six should score 0 because the datasets are absent.
    for r in card.results:
        assert r.score == 0.0
        assert "not_provisioned" in r.evidence.get("error", "")
    # ScoreCard should still record the run.
    assert card.label == "unprovisioned-test"
    assert len(card.results) == 6


def test_provisioned_humaneval_scores_correctly() -> None:
    root = data_dir() / "benchmarks" / "humaneval"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.jsonl").write_text(
        json.dumps({
            "prompt": "Return the sum of a list.",
            "function_name": "solve",
            "examples": [[[1, 2, 3], 6], [[10, 20], 30]],
        }) + "\n",
        encoding="utf-8",
    )
    from darwin.bench.frontier.humaneval import task as humaneval_task
    runtime = _runtime()
    score, evidence = humaneval_task().runner(runtime)
    assert score > 0.0
    assert evidence["passed"] >= 1


def test_provisioned_gsm8k_scores_correctly() -> None:
    root = data_dir() / "benchmarks" / "gsm8k"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.jsonl").write_text(
        json.dumps({"question": "What is 3 + 4?", "answer": "7"}) + "\n"
        + json.dumps({"question": "What is 2 * 5?", "answer": "10"}) + "\n",
        encoding="utf-8",
    )
    from darwin.bench.frontier.gsm8k import task as gsm8k_task
    runtime = _runtime()
    score, evidence = gsm8k_task().runner(runtime)
    assert score == 1.0
    assert evidence["correct"] == 2
