"""Tests for ResolutionProver (ground clause-form theorem proving)."""

from __future__ import annotations

from darwin.reasoning.resolution import Clause, Literal, ResolutionProver
from darwin.universe.concept_universe import ConceptUniverse


def test_direct_edge_proves_in_one_step() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    prover = ResolutionProver(u, max_depth=3)
    goal = Literal(source="dog", kind="is_a", target="mammal")
    proof = prover.prove(goal)
    assert proof is not None
    assert proof.depth >= 1


def test_unprovable_returns_none() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    prover = ResolutionProver(u, max_depth=3)
    goal = Literal(source="dog", kind="is_a", target="fish")
    assert prover.prove(goal) is None


def test_literal_negation_is_involutive() -> None:
    lit = Literal(source="a", kind="is_a", target="b")
    assert lit.negate().negated is True
    assert lit.negate().negate() == lit


def test_empty_clause_is_contradiction() -> None:
    assert Clause(literals=()).is_empty() is True
    assert Clause(literals=(Literal("a", "is_a", "b"),)).is_empty() is False


def test_max_depth_bounds_search() -> None:
    u = ConceptUniverse()
    u.add_relation("a", "b", "is_a", ensure_concepts=True)
    prover = ResolutionProver(u, max_depth=1)
    goal = Literal(source="a", kind="is_a", target="b")
    proof = prover.prove(goal)
    assert proof is not None
    assert proof.depth <= 1


def test_proof_serializes() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    prover = ResolutionProver(u)
    proof = prover.prove(Literal("dog", "is_a", "mammal"))
    assert proof is not None
    record = proof.to_record()
    assert "goal" in record
    assert "depth" in record
    assert "step_count" in record
