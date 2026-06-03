"""Tests for the curated domain seeds."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.domains import domains_available, load_all, load_domain


EXPECTED_DOMAINS = {
    "biology", "chemistry", "computing", "geography",
    "linguistics", "math", "physics",
}


def test_seven_domains_available() -> None:
    assert set(domains_available()) == EXPECTED_DOMAINS


def test_each_domain_emits_relations() -> None:
    for name in EXPECTED_DOMAINS:
        rels = load_domain(name)
        assert len(rels) > 10, f"domain {name} has only {len(rels)} relations"
        for rel in rels:
            assert len(rel) == 4
            src, kind, tgt, weight = rel
            assert isinstance(src, str) and src
            assert isinstance(kind, str) and kind
            assert isinstance(tgt, str) and tgt
            assert isinstance(weight, (int, float))
            assert 0.0 <= float(weight) <= 1.0


def test_load_all_aggregates() -> None:
    total = load_all()
    assert len(total) > 400, f"only {len(total)} relations across all domains"


def test_unknown_domain_returns_empty() -> None:
    assert load_domain("nonexistent") == []


def test_seeds_populate_universe() -> None:
    u = ConceptUniverse()
    for src, kind, tgt, weight in load_all():
        u.add_relation(src, tgt, kind, weight=weight, ensure_concepts=True)
    summary = u.summary()
    assert summary["concepts"] > 200
    assert summary["relations"] > 400


def test_taxonomy_inferences_resolve() -> None:
    u = ConceptUniverse()
    for src, kind, tgt, weight in load_all():
        u.add_relation(src, tgt, kind, weight=weight, ensure_concepts=True)
    from darwin.reasoning.backward import BackwardChainer
    chainer = BackwardChainer(u)
    # Multi-hop: dog → mammal → animal → organism
    proof = chainer.prove("dog", "organism")
    assert proof is not None
    assert proof.length() >= 3
    # human → primate → mammal → animal
    proof2 = chainer.prove("human", "animal")
    assert proof2 is not None


def test_capital_relations_resolve() -> None:
    u = ConceptUniverse()
    for src, kind, tgt, weight in load_all():
        u.add_relation(src, tgt, kind, weight=weight, ensure_concepts=True)
    rels = [r for r in u.neighbors("paris") if r.kind == "is_capital_of"]
    targets = {r.target for r in rels}
    assert "france" in targets
