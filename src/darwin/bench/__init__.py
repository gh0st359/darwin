"""Longitudinal intelligence measurement framework.

A frontier system has to be able to *prove* it's getting better. This
package gives Darwin a structured benchmark surface across seven
categories (coding, memory, learning, adaptation, planning, reasoning,
task_completion). Each category has one or more :class:`BenchmarkTask`s
with a deterministic run procedure and a 0..1 score.

The :class:`BenchmarkRunner` runs an entire :class:`BenchmarkSuite`
against either a fresh runtime (regression mode) or against an already-
running runtime (longitudinal mode) and produces a :class:`ScoreCard`.
:class:`BenchmarkComparison` diffs two scorecards so the operator can
see "Darwin v <today> beats Darwin v <last week> on reasoning by 12%,
loses on memory by 4%".

Scorecards persist to disk by default, so a single Darwin instance can
prove its own trajectory: yesterday's scorecard sitting next to today's
scorecard, with the trend computable from the timestamps.
"""

from darwin.bench.framework import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    ScoreCard,
    compare_scorecards,
    load_scorecard,
    save_scorecard,
)
from darwin.bench.suites import build_default_suite


__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTask",
    "ScoreCard",
    "build_default_suite",
    "compare_scorecards",
    "load_scorecard",
    "save_scorecard",
]
