"""Tests for DefeasibleReasoner (defaults with exceptions)."""

from __future__ import annotations

from darwin.reasoning.defeasible import DefeasibleReasoner
from darwin.universe.concept_universe import ConceptUniverse


def test_default_fires_when_no_exception() -> None:
    u = ConceptUniverse()
    u.add_concept("bird")
    reasoner = DefeasibleReasoner(u)
    reasoner.add_default("bird", "can", "fly")
    verdict = reasoner.query("bird", "can", "fly")
    assert verdict is not None
    assert verdict.holds is True
    assert verdict.via_exception is False


def test_exception_preempts_default() -> None:
    u = ConceptUniverse()
    u.add_relation("penguin", "bird", "is_a", ensure_concepts=True)
    reasoner = DefeasibleReasoner(u)
    rule = reasoner.add_default("bird", "can", "fly")
    reasoner.add_exception("penguin", "can", "fly", preempts=rule.rule_id, polarity=False)
    verdict = reasoner.query("penguin", "can", "fly")
    assert verdict is not None
    assert verdict.holds is False
    assert verdict.via_exception is True


def test_default_inherits_via_is_a_chain() -> None:
    u = ConceptUniverse()
    u.add_relation("sparrow", "bird", "is_a", ensure_concepts=True)
    reasoner = DefeasibleReasoner(u)
    reasoner.add_default("bird", "can", "fly")
    verdict = reasoner.query("sparrow", "can", "fly")
    assert verdict is not None
    assert verdict.holds is True
    assert verdict.via_subkind == "bird"


def test_unknown_concept_returns_none() -> None:
    u = ConceptUniverse()
    reasoner = DefeasibleReasoner(u)
    assert reasoner.query("ghost", "can", "fly") is None


def test_verdict_serializes() -> None:
    u = ConceptUniverse()
    u.add_concept("bird")
    reasoner = DefeasibleReasoner(u)
    reasoner.add_default("bird", "can", "fly")
    verdict = reasoner.query("bird", "can", "fly")
    assert verdict is not None
    record = verdict.to_record()
    assert record["subject"] == "bird"
    assert record["holds"] is True
    assert "rule_id" in record


def test_summary_counts_rules() -> None:
    u = ConceptUniverse()
    reasoner = DefeasibleReasoner(u)
    rule = reasoner.add_default("bird", "can", "fly")
    reasoner.add_exception("penguin", "can", "fly", preempts=rule.rule_id)
    s = reasoner.summary()
    assert s["defaults"] == 1
    assert s["exceptions"] == 1


def test_direct_exception_without_default_still_fires() -> None:
    u = ConceptUniverse()
    u.add_concept("ostrich")
    reasoner = DefeasibleReasoner(u)
    reasoner.add_exception("ostrich", "can", "fly", preempts="d_missing", polarity=False)
    verdict = reasoner.query("ostrich", "can", "fly")
    assert verdict is not None
    assert verdict.holds is False
    assert verdict.via_exception is True
