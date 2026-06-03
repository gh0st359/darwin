"""Tests for the benchmark framework (task / runner / scorecard / comparison)."""

from __future__ import annotations

from pathlib import Path

import pytest

from darwin.bench.framework import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    CATEGORIES,
    Comparison,
    compare_scorecards,
    load_scorecard,
    save_scorecard,
)


# --------------------------------------------------------------------------- #
# Suite + Runner
# --------------------------------------------------------------------------- #


def _runner(score: float = 0.7):
    return lambda runtime: (score, {"note": "stub"})


def test_suite_add_rejects_unknown_category() -> None:
    suite = BenchmarkSuite(name="x")
    with pytest.raises(ValueError):
        suite.add(BenchmarkTask("t", "not_a_category", "d", _runner()))


def test_runner_produces_scorecard_with_one_result_per_task() -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", _runner(0.5)))
    suite.add(BenchmarkTask("b", "memory", "d", _runner(0.8)))
    card = BenchmarkRunner(suite).run(None, label="r1")
    assert len(card.results) == 2
    assert card.label == "r1"


def test_runner_aggregates_overall_by_weighted_average() -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", _runner(0.5), weight=1.0))
    suite.add(BenchmarkTask("b", "memory", "d", _runner(1.0), weight=3.0))
    card = BenchmarkRunner(suite).run(None)
    # Weighted average: (0.5*1 + 1.0*3) / 4 = 0.875
    assert card.overall == pytest.approx(0.875, rel=1e-3)


def test_runner_aggregates_per_category() -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", _runner(0.5)))
    suite.add(BenchmarkTask("b", "coding", "d", _runner(0.9)))
    suite.add(BenchmarkTask("c", "memory", "d", _runner(0.0)))
    card = BenchmarkRunner(suite).run(None)
    assert card.per_category["coding"] == pytest.approx(0.7, rel=1e-3)
    assert card.per_category["memory"] == 0.0


def test_runner_clamps_scores_to_unit_range() -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", lambda r: (5.0, {})))
    suite.add(BenchmarkTask("b", "memory", "d", lambda r: (-2.0, {})))
    card = BenchmarkRunner(suite).run(None)
    assert card.results[0].score == 1.0
    assert card.results[1].score == 0.0


def test_runner_records_error_on_runner_exception() -> None:
    suite = BenchmarkSuite(name="x")
    def boom(runtime):
        raise RuntimeError("kaboom")
    suite.add(BenchmarkTask("a", "coding", "d", boom))
    card = BenchmarkRunner(suite).run(None)
    assert card.results[0].score == 0.0
    assert "kaboom" in card.results[0].error


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", _runner(0.5)))
    card = BenchmarkRunner(suite).run(None, label="rt")
    path = tmp_path / "c.json"
    assert save_scorecard(card, path)
    loaded = load_scorecard(path)
    assert loaded is not None
    assert loaded.label == "rt"
    assert len(loaded.results) == 1
    assert loaded.results[0].task_id == "a"


def test_load_nonexistent_returns_none(tmp_path: Path) -> None:
    assert load_scorecard(tmp_path / "missing.json") is None


def test_load_malformed_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not valid json")
    assert load_scorecard(path) is None


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_compare_picks_later_when_overall_improved() -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", _runner(0.5)))
    earlier = BenchmarkRunner(suite).run(None, label="old")
    suite2 = BenchmarkSuite(name="x")
    suite2.add(BenchmarkTask("a", "coding", "d", _runner(0.9)))
    later = BenchmarkRunner(suite2).run(None, label="new")
    cmp = compare_scorecards(earlier, later)
    assert cmp.winner == "later"
    assert cmp.overall_delta == pytest.approx(0.4, rel=1e-3)


def test_compare_picks_earlier_when_overall_regressed() -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", _runner(0.9)))
    earlier = BenchmarkRunner(suite).run(None)
    suite2 = BenchmarkSuite(name="x")
    suite2.add(BenchmarkTask("a", "coding", "d", _runner(0.3)))
    later = BenchmarkRunner(suite2).run(None)
    cmp = compare_scorecards(earlier, later)
    assert cmp.winner == "earlier"
    assert cmp.overall_delta < 0


def test_compare_picks_tie_when_overall_equal() -> None:
    suite = BenchmarkSuite(name="x")
    suite.add(BenchmarkTask("a", "coding", "d", _runner(0.5)))
    a = BenchmarkRunner(suite).run(None)
    b = BenchmarkRunner(suite).run(None)
    cmp = compare_scorecards(a, b)
    assert cmp.winner == "tie"


def test_compare_per_category_deltas() -> None:
    suite_a = BenchmarkSuite(name="x")
    suite_a.add(BenchmarkTask("a", "coding", "d", _runner(0.4)))
    suite_a.add(BenchmarkTask("b", "memory", "d", _runner(0.6)))
    a = BenchmarkRunner(suite_a).run(None)
    suite_b = BenchmarkSuite(name="x")
    suite_b.add(BenchmarkTask("a", "coding", "d", _runner(0.7)))
    suite_b.add(BenchmarkTask("b", "memory", "d", _runner(0.5)))
    b = BenchmarkRunner(suite_b).run(None)
    cmp = compare_scorecards(a, b)
    by_cat = {d.category: d.delta for d in cmp.per_category}
    assert by_cat["coding"] == pytest.approx(0.3, rel=1e-3)
    assert by_cat["memory"] == pytest.approx(-0.1, rel=1e-3)


def test_categories_constant_includes_all_seven() -> None:
    expected = {
        "coding", "memory", "learning", "adaptation",
        "planning", "reasoning", "task_completion",
    }
    # V-Bench appends "frontier" additively; assert the seven legacy categories
    # remain present alongside it.
    assert expected.issubset(set(CATEGORIES))
