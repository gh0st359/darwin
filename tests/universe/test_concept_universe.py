"""Tests for the ConceptUniverse graph substrate."""

from __future__ import annotations

import pytest

from darwin.universe.concept_universe import (
    Concept,
    ConceptUniverse,
    RELATION_KINDS,
    Relation,
)


def test_add_concept_and_lookup() -> None:
    u = ConceptUniverse()
    u.add_concept("thing", domain="structure", definition="anything referable")
    assert "thing" in u
    concept = u.get("thing")
    assert isinstance(concept, Concept)
    assert concept.domain == "structure"


def test_add_concept_is_idempotent_and_enriching() -> None:
    u = ConceptUniverse()
    u.add_concept("thing", domain="structure")
    again = u.add_concept(
        "thing",
        domain="structure",
        definition="enriched on second add",
        examples=("a rock",),
        aliases=("entity",),
    )
    assert again.definition == "enriched on second add"
    assert "a rock" in again.examples
    assert "entity" in again.aliases


def test_normalization_canonicalizes_case_and_spaces() -> None:
    u = ConceptUniverse()
    u.add_concept(" Free Will ", domain="philosophy")
    assert u.has("free_will")
    assert u.has("FREE WILL")
    assert u.get("free will").name == "free_will"


def test_add_relation_requires_known_concepts_by_default() -> None:
    u = ConceptUniverse()
    u.add_concept("a", domain="x")
    with pytest.raises(KeyError):
        u.add_relation("a", "missing", "is_a")
    with pytest.raises(KeyError):
        u.add_relation("missing", "a", "is_a")


def test_add_relation_with_ensure_concepts_auto_creates() -> None:
    u = ConceptUniverse()
    u.add_relation("alpha", "beta", "is_a", ensure_concepts=True)
    assert u.has("alpha") and u.has("beta")
    rels = u.neighbors("alpha")
    assert any(r.target == "beta" and r.kind == "is_a" for r in rels)


def test_neighbors_filter_by_kind() -> None:
    u = ConceptUniverse()
    for name in ["a", "b", "c", "d"]:
        u.add_concept(name)
    u.add_relation("a", "b", "is_a")
    u.add_relation("a", "c", "related_to")
    u.add_relation("a", "d", "causes")
    is_a = u.neighbors("a", kinds=["is_a"])
    assert {rel.target for rel in is_a} == {"b"}
    multi = u.neighbors("a", kinds=["is_a", "causes"])
    assert {rel.target for rel in multi} == {"b", "d"}


def test_neighborhood_bfs_bounded() -> None:
    u = ConceptUniverse()
    for name in ["root", "a", "b", "c", "d"]:
        u.add_concept(name)
    u.add_relation("root", "a", "is_a")
    u.add_relation("root", "b", "is_a")
    u.add_relation("a", "c", "is_a")
    u.add_relation("b", "d", "is_a")
    nbhd = u.neighborhood("root", hops=2)
    names = [n["name"] for n in nbhd["nodes"]]
    assert "root" in names and "a" in names and "c" in names


def test_shortest_path_returns_relation_chain() -> None:
    u = ConceptUniverse()
    for name in ["music", "harmony", "ratio", "math"]:
        u.add_concept(name)
    u.add_relation("music", "harmony", "part_of")
    u.add_relation("harmony", "ratio", "describes")
    u.add_relation("ratio", "math", "part_of")
    path = u.shortest_path("music", "math")
    assert [rel.source for rel in path] == ["music", "harmony", "ratio"]
    assert [rel.target for rel in path] == ["harmony", "ratio", "math"]


def test_shortest_path_returns_empty_when_unreachable() -> None:
    u = ConceptUniverse()
    u.add_concept("island_a")
    u.add_concept("island_b")
    assert u.shortest_path("island_a", "island_b") == []


def test_walk_increments_visits() -> None:
    u = ConceptUniverse()
    for name in ["a", "b", "c"]:
        u.add_concept(name)
    u.add_relation("a", "b", "related_to")
    u.add_relation("b", "c", "related_to")
    path = u.walk("a", steps=3)
    assert path
    assert u.expect("a").visits >= 1


def test_summary_aggregates_state() -> None:
    u = ConceptUniverse()
    u.add_domain("alpha")
    u.add_domain("beta")
    u.add_concept("x", domain="alpha")
    u.add_concept("y", domain="beta")
    u.add_relation("x", "y", "related_to")
    summary = u.summary()
    assert summary["concepts"] == 2
    assert summary["domains"] == 2
    assert summary["relations"] == 1
    assert summary["domain_sizes"]["alpha"] == 1


def test_relation_kinds_cover_inference_operators() -> None:
    # The canonical relation set must include the operators the reasoner
    # consults when expanding neighborhoods.
    must_have = {"is_a", "part_of", "causes", "analogous_to", "describes", "related_to"}
    assert must_have <= set(RELATION_KINDS)


def test_bulk_add_relations_skips_missing_endpoints() -> None:
    u = ConceptUniverse()
    u.add_concept("a")
    u.add_concept("b")
    n = u.add_relations(
        [
            ("a", "is_a", "b"),
            ("a", "is_a", "c"),  # missing target → skipped
            ("d", "is_a", "b"),  # missing source → skipped
        ]
    )
    assert n == 1
