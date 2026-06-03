"""Tests for BeliefNetwork (probabilistic propagation)."""

from __future__ import annotations

from darwin.reasoning.bayesian import BeliefNetwork
from darwin.universe.concept_universe import ConceptUniverse


def test_uninformative_query_returns_one_half() -> None:
    u = ConceptUniverse()
    net = BeliefNetwork(u)
    assert net.query("unknown") == 0.5


def test_prior_is_used_when_no_evidence() -> None:
    u = ConceptUniverse()
    u.add_concept("alpha")
    net = BeliefNetwork(u)
    net.set_prior("alpha", 0.8)
    net.propagate(steps=2)
    assert net.query("alpha") > 0.5


def test_evidence_pins_posterior() -> None:
    u = ConceptUniverse()
    u.add_concept("alpha")
    net = BeliefNetwork(u)
    net.set_evidence("alpha", 0.95)
    net.propagate(steps=3)
    assert abs(net.query("alpha") - 0.95) < 0.01


def test_belief_propagates_along_positive_edge() -> None:
    u = ConceptUniverse()
    u.add_relation("a", "b", "causes", weight=0.9, ensure_concepts=True)
    net = BeliefNetwork(u)
    net.set_evidence("a", 0.9)
    net.propagate(steps=3)
    # B's belief should rise above its uninformative prior.
    assert net.query("b") > 0.5


def test_propagation_converges_or_hits_step_cap() -> None:
    u = ConceptUniverse()
    u.add_relation("a", "b", "is_a", weight=0.5, ensure_concepts=True)
    net = BeliefNetwork(u)
    report = net.propagate(steps=10)
    assert report.steps_taken <= 10


def test_summary_reports_bounds() -> None:
    u = ConceptUniverse()
    u.add_concept("a")
    u.add_concept("b")
    net = BeliefNetwork(u)
    net.set_prior("a", 0.9)
    net.propagate(steps=1)
    s = net.summary()
    assert s["nodes"] == 2
    assert s["highest_posterior"] >= s["lowest_posterior"]
