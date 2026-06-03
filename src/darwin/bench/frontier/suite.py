"""Build the full frontier benchmark suite."""

from __future__ import annotations

from darwin.bench.framework import BenchmarkSuite
from darwin.bench.frontier import arc_agi, gpqa, gsm8k, humaneval, math_bench, mmlu


def build_frontier_suite() -> BenchmarkSuite:
    """Assemble the six frontier benchmarks into a single suite."""

    suite = BenchmarkSuite(name="frontier")
    suite.add(mmlu.task())
    suite.add(humaneval.task())
    suite.add(gpqa.task())
    suite.add(arc_agi.task())
    suite.add(gsm8k.task())
    suite.add(math_bench.task())
    return suite


__all__ = ["build_frontier_suite"]
