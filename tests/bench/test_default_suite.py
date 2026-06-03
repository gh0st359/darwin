"""Tests for the default benchmark suite.

These verify the suite is well-formed (every task in a valid category,
unique ids, no duplicate task entries). The actual scoring of a Darwin
runtime is exercised via the dedicated benchmark runner CLI / integration.
"""

from __future__ import annotations

from darwin.bench import build_default_suite
from darwin.bench.framework import CATEGORIES


def test_default_suite_has_tasks_in_every_category() -> None:
    suite = build_default_suite()
    by_cat = suite.by_category()
    covered = {c for c, items in by_cat.items() if items}
    # The default suite covers the legacy seven categories; "frontier" is
    # populated by build_frontier_suite() instead.
    legacy = set(CATEGORIES) - {"frontier"}
    assert covered == legacy


def test_default_suite_task_ids_are_unique() -> None:
    suite = build_default_suite()
    ids = [t.task_id for t in suite.tasks]
    assert len(ids) == len(set(ids))


def test_default_suite_has_at_least_one_task_per_category() -> None:
    suite = build_default_suite()
    by_cat = suite.by_category()
    for category in CATEGORIES:
        if category == "frontier":
            # The frontier category is populated by build_frontier_suite()
            # which lives outside the default suite.
            continue
        assert len(by_cat[category]) >= 1, f"category {category} has no tasks"


def test_default_suite_total_task_count_reasonable() -> None:
    # Sanity check: should have at least 10 tasks across the seven categories.
    suite = build_default_suite()
    assert len(suite.tasks) >= 10
