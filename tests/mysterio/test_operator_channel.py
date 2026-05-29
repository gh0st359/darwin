"""Tests for the interior event-kind taxonomy.

v6 redesign: the former operator/secrecy partition is gone. All event kinds
are now visible to any subscriber that opts in. The taxonomy survives only as
a *topical* grouping the brain terminal can use for display.
"""

from __future__ import annotations

from darwin.mysterio.operator_channel import (
    INTERIOR_EVENT_KINDS,
    OPERATOR_EVENT_KINDS,
    is_interior_kind,
    is_operator_kind,
)


def test_interior_event_kinds_includes_expected_set() -> None:
    expected = {
        "interior_simulation",
        "interior_world",
        "quarantine",
        "divergence",
        "snapshot_diff",
        "meta_proposal",
        "code_gen",
        "narrative",
        "research_finding",
        "subsystem_event",
    }
    assert expected <= set(INTERIOR_EVENT_KINDS)


def test_is_interior_kind_classifies_correctly() -> None:
    assert is_interior_kind("divergence")
    assert is_interior_kind("meta_proposal")
    assert is_interior_kind("interior_simulation")
    assert not is_interior_kind("chat")
    assert not is_interior_kind("self_modification")
    assert not is_interior_kind("simulation")  # grounded sim, not the interior one


def test_legacy_operator_kinds_alias_still_resolves() -> None:
    """Old callers importing OPERATOR_EVENT_KINDS see a superset that includes
    the legacy spellings ``private_simulation`` / ``self_world`` so they do
    not break during the rename window."""

    assert INTERIOR_EVENT_KINDS <= OPERATOR_EVENT_KINDS
    assert "private_simulation" in OPERATOR_EVENT_KINDS
    assert is_operator_kind("private_simulation")
    assert is_operator_kind("interior_simulation")
    assert not is_operator_kind("chat")
