"""Tests for ResearchAgent."""

from __future__ import annotations

from darwin.agents.research_agent import ResearchAgent, ResearchProblem


def test_fallback_extracts_relevant_sentence() -> None:
    agent = ResearchAgent(runtime=None)
    passage = (
        "Photosynthesis is the process by which plants make food. "
        "It uses sunlight, water, and carbon dioxide. "
        "The product is glucose."
    )
    problem = ResearchProblem(
        passage=passage,
        question="What is photosynthesis?",
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert "photosynthesis" in sol.answer.lower()


def test_no_match_fails_gracefully() -> None:
    agent = ResearchAgent(runtime=None)
    problem = ResearchProblem(
        passage="Cats are mammals.",
        question="What is the capital of France?",
    )
    sol = agent.solve(problem)
    assert sol.succeeded is False or sol.confidence < 0.5


def test_records_steps() -> None:
    agent = ResearchAgent(runtime=None)
    problem = ResearchProblem(
        passage="A is B.",
        question="What is A?",
    )
    sol = agent.solve(problem)
    assert len(sol.steps) >= 1


def test_wrong_problem_type_fails() -> None:
    agent = ResearchAgent(runtime=None)
    sol = agent.solve("just a string")
    assert sol.succeeded is False


def test_solution_serializes() -> None:
    agent = ResearchAgent(runtime=None)
    sol = agent.solve(
        ResearchProblem(passage="x is y", question="What is x?"),
    )
    record = sol.to_record()
    assert record["agent"] == "research"
