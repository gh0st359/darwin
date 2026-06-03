"""Tests for individual frontier adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

from darwin.agents.registry import AgentRegistry
from darwin.bench.frontier import arc_agi, gpqa, gsm8k, humaneval, math_bench, mmlu
from darwin.paths import data_dir
from darwin.universe.concept_universe import ConceptUniverse


def _runtime() -> SimpleNamespace:
    u = ConceptUniverse()
    u.add_relation("photon", "light", "is_a", ensure_concepts=True)
    return SimpleNamespace(agent_registry=AgentRegistry(), universe=u)


def _write_jsonl(name: str, items: list[dict]) -> None:
    root = data_dir() / "benchmarks" / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n",
        encoding="utf-8",
    )


def test_mmlu_routes_math_subject_to_math_agent() -> None:
    _write_jsonl("mmlu", [
        {
            "subject": "math",
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5"],
            "answer": "4",
        },
    ])
    runtime = _runtime()
    runtime.agent_registry.math.solve("What is 2 + 2?")
    score, evidence = mmlu.task().runner(runtime)
    assert evidence["items_seen"] == 1


def test_humaneval_unprovisioned() -> None:
    score, evidence = humaneval.task().runner(_runtime())
    assert score == 0.0
    assert "not_provisioned" in evidence["error"]


def test_gpqa_handles_provisioned_choices() -> None:
    _write_jsonl("gpqa", [
        {
            "question": "What is photon?",
            "choices": ["light", "mass"],
            "answer": "light",
        },
    ])
    runtime = _runtime()
    score, evidence = gpqa.task().runner(runtime)
    assert evidence["items_seen"] == 1


def test_arc_agi_unprovisioned() -> None:
    score, evidence = arc_agi.task().runner(_runtime())
    assert score == 0.0
    assert "not_provisioned" in evidence["error"]


def test_arc_agi_provisioned_grid_identity() -> None:
    _write_jsonl("arc_agi", [
        {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test_input": [[3, 4]],
            "test_output": [[3, 4]],
        },
    ])
    score, evidence = arc_agi.task().runner(_runtime())
    assert score == 1.0
    assert evidence["solved"] == 1


def test_gsm8k_unprovisioned() -> None:
    score, evidence = gsm8k.task().runner(_runtime())
    assert score == 0.0
    assert "not_provisioned" in evidence["error"]


def test_math_bench_unprovisioned() -> None:
    score, evidence = math_bench.task().runner(_runtime())
    assert score == 0.0


def test_math_bench_provisioned() -> None:
    _write_jsonl("math", [
        {"problem": "What is 6 / 2?", "solution": "3"},
    ])
    score, evidence = math_bench.task().runner(_runtime())
    assert score == 1.0
