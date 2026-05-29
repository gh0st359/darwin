"""Tests for the HypothesisEngine."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.hypothesis import Hypothesis, HypothesisEngine


def _world(*edges: tuple[str, str, str]) -> ConceptUniverse:
    u = ConceptUniverse()
    for source, kind, target in edges:
        u.add_relation(source, target, kind, ensure_concepts=True)
    return u


# -- transitive ----------------------------------------------------------


def test_transitive_closure_hypothesis() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
    )
    engine = HypothesisEngine(u)
    hypotheses = engine.generate()
    # The transitive closure should be proposed.
    assert any(
        h.source == "dog" and h.target == "animal" and h.kind == "is_a"
        and h.pathway == "transitive"
        for h in hypotheses
    )


def test_transitive_does_not_propose_existing_edge() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
        ("dog", "is_a", "animal"),  # already present
    )
    engine = HypothesisEngine(u)
    hypotheses = engine.generate()
    transitive_dog_animal = [
        h for h in hypotheses
        if h.source == "dog" and h.target == "animal" and h.pathway == "transitive"
    ]
    assert transitive_dog_animal == []


# -- analogical ---------------------------------------------------------


def test_analogical_hypothesis_when_neighborhoods_overlap() -> None:
    # hammer and wrench share metal + grip + user; hammer has 'driving'
    # that wrench lacks. The engine should propose wrench drives.
    u = ConceptUniverse()
    for name in ["hammer", "wrench", "metal", "grip", "user", "driving"]:
        u.add_concept(name)
    u.add_relation("hammer", "metal", "part_of")
    u.add_relation("hammer", "grip", "part_of")
    u.add_relation("hammer", "user", "requires")
    u.add_relation("hammer", "driving", "causes")
    u.add_relation("wrench", "metal", "part_of")
    u.add_relation("wrench", "grip", "part_of")
    u.add_relation("wrench", "user", "requires")
    engine = HypothesisEngine(u, analogical_jaccard_threshold=0.4)
    hypotheses = engine.generate()
    # The engine should propose at least one analogical hypothesis
    # pointing from wrench toward driving, or some related cross-link.
    assert any(h.pathway == "analogical" for h in hypotheses)


def test_analogical_hypothesis_carries_jaccard_evidence() -> None:
    u = ConceptUniverse()
    u.add_concept("a")
    u.add_concept("b")
    u.add_concept("x")
    u.add_concept("y")
    u.add_concept("z")
    u.add_relation("a", "x", "is_a")
    u.add_relation("a", "y", "is_a")
    u.add_relation("a", "z", "is_a")
    u.add_relation("b", "x", "is_a")
    u.add_relation("b", "y", "is_a")
    engine = HypothesisEngine(u, analogical_jaccard_threshold=0.3)
    hypotheses = engine.generate()
    analogical = [h for h in hypotheses if h.pathway == "analogical"]
    assert analogical
    assert "jaccard" in analogical[0].evidence


# -- cross-domain -------------------------------------------------------


def test_cross_domain_bridge_hypothesis() -> None:
    """River (physics) and melody (arts) should propose an analogous_to
    edge if they share a relation kind AND at least one neighbor.
    """

    u = ConceptUniverse()
    u.add_concept("river", domain="physics")
    u.add_concept("melody", domain="arts")
    # Shared concept used as common reference point.
    u.add_concept("flow", domain="dynamics")
    u.add_concept("source", domain="structure")
    # Both river and melody is_a flow and require a source — the criterion
    # needs kind overlap AND at least one shared neighbor.
    u.add_relation("river", "flow", "is_a")
    u.add_relation("river", "source", "requires")
    u.add_relation("melody", "flow", "is_a")
    u.add_relation("melody", "source", "requires")
    engine = HypothesisEngine(u)
    hypotheses = engine.generate()
    cross = [h for h in hypotheses if h.pathway == "cross_domain"]
    assert cross
    assert cross[0].kind == "analogous_to"
    # The rationale should mention the shared neighbor(s).
    assert "flow" in cross[0].rationale or "source" in cross[0].rationale


# -- feedback / acceptance ---------------------------------------------


def test_refuted_hypotheses_are_not_re_proposed() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
    )
    engine = HypothesisEngine(u)
    initial = engine.generate()
    # Refute the dog -> animal hypothesis.
    engine.refute("dog", "is_a", "animal")
    after = engine.generate()
    assert any(h.source == "dog" and h.target == "animal" for h in initial)
    assert not any(h.source == "dog" and h.target == "animal" for h in after)


def test_accept_actually_adds_edge_to_universe() -> None:
    u = ConceptUniverse()
    u.add_concept("a")
    u.add_concept("b")
    engine = HypothesisEngine(u)
    hypothesis = Hypothesis(source="a", target="b", kind="is_a", pathway="transitive",
                            rationale="test")
    engine.accept(hypothesis)
    rels = u.neighbors("a", kinds=["is_a"])
    assert any(r.target == "b" for r in rels)


def test_summary_reports_pathway_counts() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
    )
    engine = HypothesisEngine(u)
    engine.generate()
    summary = engine.summary()
    assert summary["total_produced"] >= 1
    assert "by_pathway" in summary


def test_hypothesis_as_question_renders_for_each_kind() -> None:
    h = Hypothesis(source="x", target="y", kind="is_a", pathway="transitive", rationale="r")
    assert "kind of" in h.as_question() or "is a" in h.as_question().lower()
    h2 = Hypothesis(source="x", target="y", kind="causes", pathway="analogical", rationale="r")
    assert "cause" in h2.as_question().lower()
    h3 = Hypothesis(source="x", target="y", kind="analogous_to", pathway="cross_domain", rationale="r")
    assert "analog" in h3.as_question().lower()
