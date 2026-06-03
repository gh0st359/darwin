"""Tests for MathAgent."""

from __future__ import annotations

from darwin.agents.math_agent import MathAgent, MathProblem


def test_simple_addition() -> None:
    agent = MathAgent()
    sol = agent.solve("What is 3 + 4?")
    assert sol.succeeded
    assert sol.answer == "7"


def test_subtraction() -> None:
    agent = MathAgent()
    sol = agent.solve("What is 10 - 7?")
    assert sol.succeeded
    assert sol.answer == "3"


def test_parenthesised_expression() -> None:
    agent = MathAgent()
    sol = agent.solve("Compute 2 * (3 + 4).")
    assert sol.succeeded
    assert sol.answer == "14"


def test_variable_substitution() -> None:
    agent = MathAgent()
    sol = agent.solve("If a=3 and b=4, what is a+b?")
    assert sol.succeeded
    assert sol.answer == "7"


def test_division_returns_fraction() -> None:
    agent = MathAgent()
    sol = agent.solve("What is 1 / 3?")
    assert sol.succeeded
    # Answer should be either the fraction or its decimal representation.
    assert "/" in sol.answer or sol.answer.startswith("0.")


def test_exponent() -> None:
    agent = MathAgent()
    sol = agent.solve("What is 2 ^ 10?")
    assert sol.succeeded
    assert sol.answer == "1024"


def test_no_expression_returns_failure() -> None:
    agent = MathAgent()
    sol = agent.solve("Hello there.")
    assert sol.succeeded is False
    assert sol.answer == ""


def test_solution_carries_value() -> None:
    agent = MathAgent()
    sol = agent.solve(MathProblem(prompt="What is 5 + 5?"))
    assert sol.succeeded
    assert sol.extras["value_numerator"] == 10
    assert sol.extras["value_denominator"] == 1
