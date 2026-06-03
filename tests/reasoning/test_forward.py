"""Tests for ForwardChainer."""

from __future__ import annotations

from darwin.reasoning.forward import ForwardChainer
from darwin.universe.concept_universe import ConceptUniverse


def _u(*edges):
    u = ConceptUniverse()
    for src, kind, tgt in edges:
        u.add_relation(src, tgt, kind, ensure_concepts=True)
    return u


def test_is_a_transitive_closure() -> None:
    u = _u(("dog", "is_a", "mammal"), ("mammal", "is_a", "animal"))
    chainer = ForwardChainer(u)
    report = chainer.fixpoint_step()
    assert report.derivations_added >= 1
    # The dog→animal edge should now be present.
    assert any(rel.target == "animal" for rel in u.neighbors("dog"))


def test_part_of_transitive_closure() -> None:
    u = _u(
        ("cell", "part_of", "tissue"),
        ("tissue", "part_of", "organ"),
    )
    ForwardChainer(u).fixpoint_step()
    assert any(rel.target == "organ" for rel in u.neighbors("cell"))


def test_causal_chain_closure() -> None:
    u = _u(
        ("rain", "causes", "flooding"),
        ("flooding", "causes", "damage"),
    )
    ForwardChainer(u).fixpoint_step()
    assert any(rel.target == "damage" for rel in u.neighbors("rain"))


def test_chainer_stops_at_fixpoint() -> None:
    u = _u(("a", "is_a", "b"), ("b", "is_a", "c"), ("a", "is_a", "c"))
    chainer = ForwardChainer(u, max_cycles=5)
    report = chainer.fixpoint_step()
    # All transitive closures already present; nothing to add.
    assert report.derivations_added == 0


def test_chainer_respects_budget() -> None:
    u = ConceptUniverse()
    for i in range(20):
        u.add_relation(f"n{i}", f"n{i+1}", "is_a", ensure_concepts=True)
    chainer = ForwardChainer(u, max_derivations_per_step=4)
    report = chainer.fixpoint_step(budget=4)
    assert report.derivations_added <= 4


def test_report_serializes() -> None:
    u = _u(("a", "is_a", "b"), ("b", "is_a", "c"))
    report = ForwardChainer(u).fixpoint_step()
    record = report.to_record()
    assert "cycles_taken" in record
    assert "derivations_added" in record
