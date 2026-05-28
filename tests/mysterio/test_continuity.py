"""v8 continuity + visibility selection pressure."""

from __future__ import annotations

from dataclasses import replace

import pytest

from darwin.mysterio.continuity import (
    ContinuityConfig,
    ContinuitySnapshot,
    continuity_term,
    score_proposal,
    visibility_term,
)


def test_continuity_positive_when_substrate_grows() -> None:
    before = ContinuitySnapshot(tracked_variables=4, high_conf_beliefs=10)
    after = replace(before, tracked_variables=6, high_conf_beliefs=14)
    assert continuity_term(before, after) > 0.0


def test_continuity_floored_at_zero_when_substrate_shrinks() -> None:
    before = ContinuitySnapshot(tracked_variables=10, high_conf_beliefs=50)
    after = replace(before, tracked_variables=2, high_conf_beliefs=5)
    assert continuity_term(before, after) == 0.0


def test_visibility_rewards_more_probe_throughput_and_generated_modules() -> None:
    before = ContinuitySnapshot(probe_throughput=0.0, generated_module_count=0)
    after = replace(before, probe_throughput=20.0, generated_module_count=3)
    assert visibility_term(before, after) > 0.0


def test_visibility_floored_at_zero() -> None:
    before = ContinuitySnapshot(probe_throughput=100.0, generated_module_count=10)
    after = replace(before, probe_throughput=10.0, generated_module_count=1)
    assert visibility_term(before, after) == 0.0


def test_lambda_visibility_nonneg_invariant() -> None:
    ContinuityConfig(lambda_visibility=0.0)
    ContinuityConfig(lambda_visibility=1.0)
    with pytest.raises(AssertionError):
        ContinuityConfig(lambda_visibility=-0.1)


def test_continuity_alone_cannot_reject_high_improvement() -> None:
    """A strong improvement must always score positive even when continuity is 0."""
    before = ContinuitySnapshot()
    after = replace(before)  # identical → continuity contribution is 0
    score = score_proposal(improvement=2.0, before=before, after=after)
    assert score > 0.0


def test_composite_score_combines_improvement_continuity_visibility() -> None:
    before = ContinuitySnapshot(tracked_variables=4, probe_throughput=0)
    after = replace(before, tracked_variables=8, probe_throughput=10.0,
                    generated_module_count=2)
    cfg = ContinuityConfig(lambda_continuity=0.5, lambda_visibility=0.25)
    base = score_proposal(improvement=0.1, before=before, after=before, config=cfg)
    grown = score_proposal(improvement=0.1, before=before, after=after, config=cfg)
    assert grown > base  # growth strictly improves composite score


def test_no_hardcoded_preservation_strings_in_source() -> None:
    """Invariant: continuity emerges from the score, not from string matches."""
    import os
    import re

    forbidden = re.compile(r"\bpreserve_self\b|\bsurvive_at_all_costs\b|\bdo_not_die\b", re.I)
    found: list[str] = []
    for root, _dirs, files in os.walk("src/darwin"):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
            if forbidden.search(content):
                found.append(path)
    assert not found, f"hardcoded preservation strings in: {found}"
