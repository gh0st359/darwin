"""Tests for the frontier dataset loaders."""

from __future__ import annotations

import json

from darwin.bench.frontier.loaders import load_benchmark
from darwin.paths import data_dir


def test_missing_dataset_returns_not_provisioned() -> None:
    result = load_benchmark("nonexistent_dataset")
    assert result.status == "not_provisioned"
    assert result.loaded is False


def test_jsonl_dataset_loads() -> None:
    root = data_dir() / "benchmarks" / "tiny_jsonl"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tasks.jsonl"
    path.write_text(
        '{"q": "a"}\n{"q": "b"}\n',
        encoding="utf-8",
    )
    result = load_benchmark("tiny_jsonl")
    assert result.loaded
    assert len(result.items) == 2
    assert result.items[0]["q"] == "a"


def test_json_array_dataset_loads() -> None:
    root = data_dir() / "benchmarks" / "tiny_json"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.json").write_text(
        json.dumps([{"k": 1}, {"k": 2}, {"k": 3}]),
        encoding="utf-8",
    )
    result = load_benchmark("tiny_json")
    assert result.loaded
    assert len(result.items) == 3


def test_corrupt_lines_skipped() -> None:
    root = data_dir() / "benchmarks" / "noisy_jsonl"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.jsonl").write_text(
        '{"ok": 1}\nnot json\n{"ok": 2}\n',
        encoding="utf-8",
    )
    result = load_benchmark("noisy_jsonl")
    assert result.loaded
    assert len(result.items) == 2
