"""Tests for HypotheticalReasoner (universe overlays)."""

from __future__ import annotations

from darwin.reasoning.hypothetical import HypotheticalReasoner
from darwin.universe.concept_universe import ConceptUniverse


def test_overlay_adds_edges_during_context() -> None:
    u = ConceptUniverse()
    u.add_concept("dog")
    u.add_concept("mammal")
    reasoner = HypotheticalReasoner(u)
    with reasoner.overlay([("dog", "is_a", "mammal")]) as result:
        assert any(r.target == "mammal" for r in u.neighbors("dog"))
        assert ("dog", "is_a", "mammal") in result.assumptions


def test_overlay_removes_edges_on_exit() -> None:
    u = ConceptUniverse()
    u.add_concept("dog")
    u.add_concept("mammal")
    before = u.summary()["relations"]
    with HypotheticalReasoner(u).overlay([("dog", "is_a", "mammal")]):
        pass
    after = u.summary()["relations"]
    assert after == before


def test_overlay_preserves_pre_existing_edges() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    # The edge already exists — overlay should not add a duplicate
    # and not remove the original on exit.
    with HypotheticalReasoner(u).overlay([("dog", "is_a", "mammal")]):
        pass
    assert any(r.target == "mammal" for r in u.neighbors("dog"))


def test_overlay_creates_missing_concepts() -> None:
    u = ConceptUniverse()
    reasoner = HypotheticalReasoner(u)
    with reasoner.overlay([("alpha", "is_a", "beta")]):
        assert u.has("alpha") and u.has("beta")


def test_multiple_facts_in_one_overlay() -> None:
    u = ConceptUniverse()
    with HypotheticalReasoner(u).overlay([
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
    ]):
        assert any(r.target == "mammal" for r in u.neighbors("dog"))
        assert any(r.target == "animal" for r in u.neighbors("mammal"))


def test_hypothetical_result_serializes() -> None:
    u = ConceptUniverse()
    with HypotheticalReasoner(u).overlay([("a", "is_a", "b")]) as result:
        record = result.to_record()
        assert "assumptions" in record
