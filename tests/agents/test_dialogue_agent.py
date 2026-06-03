"""Tests for DialogueAgent."""

from __future__ import annotations

import re

from darwin.agents.dialogue_agent import DialogueAgent, DialogueProblem


def test_soft_reply_to_question() -> None:
    agent = DialogueAgent(runtime=None)
    sol = agent.solve("What time is it?")
    assert sol.succeeded
    assert sol.answer != ""


def test_soft_reply_to_statement() -> None:
    agent = DialogueAgent(runtime=None)
    sol = agent.solve("I had a long day.")
    assert sol.succeeded
    assert sol.answer != ""


def test_empty_message_handled() -> None:
    agent = DialogueAgent(runtime=None)
    sol = agent.solve("")
    assert sol.succeeded


def test_reply_never_leaks_json() -> None:
    agent = DialogueAgent(runtime=None)
    for msg in ["hi", "what's up?", "tell me a story", ""]:
        sol = agent.solve(msg)
        assert "{" not in sol.answer
        assert "}" not in sol.answer
        assert not re.search(r'"\w+":', sol.answer)


def test_dialogue_problem_with_history() -> None:
    agent = DialogueAgent(runtime=None)
    problem = DialogueProblem(message="And then?", history=["Once upon a time."])
    sol = agent.solve(problem)
    assert sol.succeeded


def test_solution_records_agent_name() -> None:
    agent = DialogueAgent(runtime=None)
    sol = agent.solve("hello")
    assert sol.agent == "dialogue"
    assert sol.to_record()["agent"] == "dialogue"
