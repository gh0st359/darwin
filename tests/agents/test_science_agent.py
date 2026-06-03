"""Tests for ScienceAgent."""

from __future__ import annotations

from types import SimpleNamespace

from darwin.agents.science_agent import ScienceAgent, ScienceProblem
from darwin.universe.concept_universe import ConceptUniverse


def _runtime_with_universe(u: ConceptUniverse) -> SimpleNamespace:
    return SimpleNamespace(
        universe=u, cortical_mesh=None,
        forward_chainer=None, belief_network=None,
    )


def test_picks_choice_with_more_graph_evidence() -> None:
    u = ConceptUniverse()
    u.add_relation("photon", "light", "is_a", ensure_concepts=True)
    u.add_relation("light", "electromagnetic", "is_a", ensure_concepts=True)
    agent = ScienceAgent(runtime=_runtime_with_universe(u))
    problem = ScienceProblem(
        question="What kind of phenomenon is a photon?",
        choices=["electromagnetic", "gravitational", "nuclear"],
    )
    sol = agent.solve(problem)
    assert sol.answer == "electromagnetic"


def test_no_choices_fails_gracefully() -> None:
    u = ConceptUniverse()
    agent = ScienceAgent(runtime=_runtime_with_universe(u))
    problem = ScienceProblem(question="anything?", choices=[])
    sol = agent.solve(problem)
    assert sol.succeeded is False


def test_string_input_handled_gracefully() -> None:
    u = ConceptUniverse()
    agent = ScienceAgent(runtime=_runtime_with_universe(u))
    sol = agent.solve("What is light?")
    # No choices, so it shouldn't succeed.
    assert sol.succeeded is False


def test_scoring_records_per_choice_scores() -> None:
    u = ConceptUniverse()
    u.add_relation("water", "liquid", "is_a", ensure_concepts=True)
    agent = ScienceAgent(runtime=_runtime_with_universe(u))
    problem = ScienceProblem(
        question="What is water?",
        choices=["liquid", "solid"],
    )
    sol = agent.solve(problem)
    assert "scores" in sol.extras
    assert "liquid" in sol.extras["scores"]
    assert "solid" in sol.extras["scores"]


def test_no_universe_returns_no_success() -> None:
    agent = ScienceAgent(runtime=None)
    sol = agent.solve(ScienceProblem(question="?", choices=["a", "b"]))
    assert sol.succeeded is False


def test_solution_serializes() -> None:
    u = ConceptUniverse()
    u.add_relation("photon", "light", "is_a", ensure_concepts=True)
    agent = ScienceAgent(runtime=_runtime_with_universe(u))
    sol = agent.solve(ScienceProblem(question="What is photon?", choices=["light", "mass"]))
    record = sol.to_record()
    assert record["agent"] == "science"
