"""Tests for the CuriosityEngine."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.curiosity import CuriosityEngine, CuriosityProbe
from darwin.universe.primitive_seed import seed_primitives


def test_isolated_concept_surfaces_a_probe() -> None:
    u = ConceptUniverse()
    u.add_concept("lonely", domain="x")  # zero edges
    engine = CuriosityEngine(u)
    probes = engine.probe()
    assert any(p.kind == "isolated_concept" and "lonely" in p.concepts for p in probes)


def test_weak_definition_surfaces_a_probe() -> None:
    u = ConceptUniverse()
    u.add_concept("thin", domain="x", definition="x")  # very short definition
    u.add_concept("supporting", domain="x", definition="rich enough to not be flagged here")
    u.add_relation("thin", "supporting", "related_to")
    engine = CuriosityEngine(u, weak_definition_threshold=8)
    probes = engine.probe()
    kinds = {p.kind for p in probes}
    assert "weak_definition" in kinds


def test_missing_cross_domain_bridge_surfaces_probe() -> None:
    u = ConceptUniverse()
    u.add_concept("a1", domain="alpha")
    u.add_concept("a2", domain="alpha")
    u.add_relation("a1", "a2", "related_to")
    u.add_concept("b1", domain="beta")
    u.add_concept("b2", domain="beta")
    u.add_relation("b1", "b2", "related_to")
    # alpha and beta share no cross-domain edges.
    engine = CuriosityEngine(u)
    probes = engine.probe()
    assert any(
        p.kind == "missing_bridge"
        and set(p.evidence.get("domains", [])) == {"alpha", "beta"}
        for p in probes
    )


def test_cluster_gap_surfaces_for_sibling_concepts() -> None:
    u = ConceptUniverse()
    u.add_concept("parent", domain="x")
    u.add_concept("kid_a", domain="x")
    u.add_concept("kid_b", domain="x")
    u.add_relation("kid_a", "parent", "is_a")
    u.add_relation("kid_b", "parent", "is_a")
    engine = CuriosityEngine(u)
    probes = engine.probe()
    cluster_probes = [p for p in probes if p.kind == "cluster_gap"]
    assert cluster_probes
    target = cluster_probes[0]
    assert {"kid_a", "kid_b", "parent"} <= set(target.concepts)


def test_probes_respect_max_limit() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    engine = CuriosityEngine(u, max_probes=3)
    probes = engine.probe()
    assert len(probes) <= 3


def test_summary_aggregates_probe_kinds() -> None:
    u = ConceptUniverse()
    u.add_concept("a", domain="alpha")
    u.add_concept("b", domain="beta")
    engine = CuriosityEngine(u)
    summary = engine.summary()
    assert "probes" in summary
    assert "kinds" in summary


def test_curiosity_probe_serializes() -> None:
    probe = CuriosityProbe(
        kind="isolated_concept",
        question="?",
        concepts=["x"],
        score=0.8,
        evidence={"a": 1},
    )
    record = probe.to_record()
    assert record["kind"] == "isolated_concept"
    assert record["concepts"] == ["x"]
