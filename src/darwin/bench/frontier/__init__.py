"""Frontier benchmark adapters: MMLU / HumanEval / GPQA / ARC-AGI / GSM8K / MATH."""

from __future__ import annotations

from darwin.bench.frontier.loaders import LoadResult, load_benchmark
from darwin.bench.frontier.suite import build_frontier_suite

__all__ = [
    "LoadResult",
    "build_frontier_suite",
    "load_benchmark",
]
