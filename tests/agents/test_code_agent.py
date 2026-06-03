"""Tests for CodeAgent."""

from __future__ import annotations

from darwin.agents.code_agent import CodeAgent, CodeProblem


def test_sum_problem_solved() -> None:
    agent = CodeAgent()
    problem = CodeProblem(
        prompt="Write a function that returns the sum of a list of numbers.",
        function_name="solve",
        examples=[([1, 2, 3], 6), ([10, 20], 30), ([], 0)],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert "sum" in sol.answer


def test_length_problem_solved() -> None:
    agent = CodeAgent()
    problem = CodeProblem(
        prompt="Return the count of items in the list.",
        function_name="solve",
        examples=[([1, 2, 3], 3), ([], 0), (["a"], 1)],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert "len" in sol.answer


def test_reverse_problem_solved() -> None:
    agent = CodeAgent()
    problem = CodeProblem(
        prompt="Return the list reversed.",
        function_name="solve",
        examples=[([1, 2, 3], [3, 2, 1]), ([], [])],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert "reversed" in sol.answer


def test_filter_even_problem_solved() -> None:
    agent = CodeAgent()
    problem = CodeProblem(
        prompt="Return only the even numbers from the input.",
        function_name="solve",
        examples=[([1, 2, 3, 4], [2, 4]), ([1, 3, 5], [])],
    )
    sol = agent.solve(problem)
    assert sol.succeeded
    assert "% 2" in sol.answer


def test_no_examples_returns_best_guess() -> None:
    agent = CodeAgent()
    problem = CodeProblem(
        prompt="Sum the numbers.",
        function_name="solve",
        examples=[],
    )
    sol = agent.solve(problem)
    # With no examples to check against, succeeded should still be True
    # (vacuous match) and code returns a sum implementation.
    assert "sum" in sol.answer


def test_string_problem_normalises() -> None:
    agent = CodeAgent()
    sol = agent.solve("write a sum function")
    # String input becomes a CodeProblem with no examples; agent still
    # produces a candidate.
    assert sol.answer != ""


def test_solution_serializes() -> None:
    agent = CodeAgent()
    problem = CodeProblem(
        prompt="Return the sum.",
        function_name="solve",
        examples=[([1, 2], 3)],
    )
    sol = agent.solve(problem)
    record = sol.to_record()
    assert record["agent"] == "code"
    assert "elapsed_ms" in record
