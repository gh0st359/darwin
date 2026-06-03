"""Tests for PlanningAgent."""

from __future__ import annotations

from darwin.agents.planning_agent import PlanningAgent, PlanningProblem


def test_identity_pattern() -> None:
    agent = PlanningAgent()
    problem = PlanningProblem(
        examples=[([[1, 2], [3, 4]], [[1, 2], [3, 4]])],
        test_input=[[5, 6], [7, 8]],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert sol.extras["primitive"] == "identity"
    assert sol.extras["grid"] == [[5, 6], [7, 8]]


def test_horizontal_flip() -> None:
    agent = PlanningAgent()
    problem = PlanningProblem(
        examples=[
            ([[1, 2, 3]], [[3, 2, 1]]),
            ([[4, 5, 6]], [[6, 5, 4]]),
        ],
        test_input=[[7, 8, 9]],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert sol.extras["primitive"] == "flip_h"
    assert sol.extras["grid"] == [[9, 8, 7]]


def test_vertical_flip() -> None:
    agent = PlanningAgent()
    problem = PlanningProblem(
        examples=[([[1], [2], [3]], [[3], [2], [1]])],
        test_input=[[4], [5], [6]],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert sol.extras["primitive"] == "flip_v"


def test_rotate_90() -> None:
    agent = PlanningAgent()
    problem = PlanningProblem(
        examples=[([[1, 2], [3, 4]], [[3, 1], [4, 2]])],
        test_input=[[5, 6], [7, 8]],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert sol.extras["primitive"] == "rotate_90"


def test_color_swap() -> None:
    agent = PlanningAgent()
    problem = PlanningProblem(
        examples=[
            ([[1, 2, 1]], [[2, 1, 2]]),
            ([[2, 2, 1]], [[1, 1, 2]]),
        ],
        test_input=[[1, 1, 2]],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert sol.extras["primitive"] == "color_swap"


def test_no_match_returns_identity_with_low_confidence() -> None:
    agent = PlanningAgent()
    problem = PlanningProblem(
        examples=[([[1, 2]], [[7, 9]])],  # arbitrary unmapping
        test_input=[[1, 2]],
    )
    sol = agent.solve(problem)
    assert sol.confidence < 0.5


def test_missing_test_input_fails() -> None:
    agent = PlanningAgent()
    sol = agent.solve(PlanningProblem(examples=[]))
    assert sol.succeeded is False
