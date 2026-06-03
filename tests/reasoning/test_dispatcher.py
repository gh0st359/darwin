"""Tests for ReasoningDispatcher (route a query to the right reasoner)."""

from __future__ import annotations

from darwin.reasoning.dispatcher import ReasoningDispatcher
from darwin.universe.concept_universe import ConceptUniverse


def test_kind_check_via_backward() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    dispatcher = ReasoningDispatcher(universe=u)
    result = dispatcher.try_resolve("is a dog a mammal?")
    assert result is not None
    assert result.reasoner in ("backward", "defeasible", "resolution")
    assert result.succeeded() or "no proof" in result.answer


def test_kind_check_unprovable_returns_no_proof_message() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    dispatcher = ReasoningDispatcher(universe=u)
    result = dispatcher.try_resolve("is a dog a fish?")
    assert result is not None
    assert "no proof" in result.answer


def test_part_of_check() -> None:
    u = ConceptUniverse()
    u.add_relation("cell", "tissue", "part_of", ensure_concepts=True)
    dispatcher = ReasoningDispatcher(universe=u)
    result = dispatcher.try_resolve("is a cell part of a tissue?")
    assert result is not None
    assert "part of" in result.answer


def test_causal_check_dispatches_to_backward() -> None:
    u = ConceptUniverse()
    u.add_relation("rain", "flooding", "causes", ensure_concepts=True)
    dispatcher = ReasoningDispatcher(universe=u)
    result = dispatcher.try_resolve("does rain cause flooding?")
    assert result is not None
    assert "causes" in result.answer or "flooding" in result.answer


def test_probability_query_dispatches_to_bayesian() -> None:
    u = ConceptUniverse()
    u.add_concept("rain")
    dispatcher = ReasoningDispatcher(universe=u)
    dispatcher.bayesian.set_prior("rain", 0.8)
    result = dispatcher.try_resolve("how likely that rain")
    assert result is not None
    assert result.reasoner == "bayesian"
    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0


def test_unmatched_query_returns_none() -> None:
    u = ConceptUniverse()
    dispatcher = ReasoningDispatcher(universe=u)
    assert dispatcher.try_resolve("what is the meaning of life?") is None


def test_empty_message_returns_none() -> None:
    u = ConceptUniverse()
    dispatcher = ReasoningDispatcher(universe=u)
    assert dispatcher.try_resolve("") is None


def test_dispatch_result_serializes() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    dispatcher = ReasoningDispatcher(universe=u)
    result = dispatcher.try_resolve("is a dog a mammal?")
    assert result is not None
    record = result.to_record()
    assert "reasoner" in record
    assert "answer" in record
    assert "succeeded" in record
