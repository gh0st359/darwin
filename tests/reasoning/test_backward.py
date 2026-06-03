"""Tests for BackwardChainer (goal-directed proof search)."""

from __future__ import annotations

from darwin.reasoning.backward import BackwardChainer, ProofStep, ProofTree
from darwin.universe.concept_universe import ConceptUniverse


def _u(*edges):
    u = ConceptUniverse()
    for src, kind, tgt in edges:
        u.add_relation(src, tgt, kind, ensure_concepts=True)
    return u


def test_one_hop_is_a_proof() -> None:
    u = _u(("dog", "is_a", "mammal"))
    proof = BackwardChainer(u).prove("dog", "mammal", kind="is_a")
    assert proof is not None
    assert proof.length() == 1
    assert proof.chain[0].source == "dog"
    assert proof.chain[0].target == "mammal"


def test_multi_hop_chain_returns_full_proof() -> None:
    u = _u(("dog", "is_a", "mammal"), ("mammal", "is_a", "animal"))
    proof = BackwardChainer(u).prove("dog", "animal")
    assert proof is not None
    assert proof.length() == 2
    assert [s.target for s in proof.chain] == ["mammal", "animal"]


def test_no_proof_returns_none() -> None:
    u = _u(("dog", "is_a", "mammal"))
    assert BackwardChainer(u).prove("dog", "fish") is None


def test_self_target_returns_none() -> None:
    u = _u(("dog", "is_a", "mammal"))
    assert BackwardChainer(u).prove("dog", "dog") is None


def test_proof_confidence_decays_per_hop() -> None:
    u = ConceptUniverse()
    u.add_relation("a", "b", "is_a", weight=0.9, ensure_concepts=True)
    u.add_relation("b", "c", "is_a", weight=0.9, ensure_concepts=True)
    proof = BackwardChainer(u).prove("a", "c")
    assert proof is not None
    assert proof.confidence < 0.9


def test_max_depth_bounds_search() -> None:
    u = ConceptUniverse()
    for i in range(20):
        u.add_relation(f"n{i}", f"n{i+1}", "is_a", ensure_concepts=True)
    chainer = BackwardChainer(u, max_depth=3)
    assert chainer.prove("n0", "n10") is None


def test_proof_tree_serializes() -> None:
    u = _u(("a", "is_a", "b"))
    proof = BackwardChainer(u).prove("a", "b")
    record = proof.to_record()
    assert record["length"] == 1
    assert len(record["chain"]) == 1
