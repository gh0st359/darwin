"""Tests for AgentRegistry."""

from __future__ import annotations

from darwin.agents.registry import AgentRegistry


def test_registry_instantiates_six_agents() -> None:
    reg = AgentRegistry()
    assert reg.code is not None
    assert reg.math is not None
    assert reg.science is not None
    assert reg.planning is not None
    assert reg.research is not None
    assert reg.dialogue is not None
    assert len(reg.all()) == 6


def test_registry_summary() -> None:
    reg = AgentRegistry()
    s = reg.summary()
    assert s["count"] == 6
    assert set(s["agents"]) == {
        "code", "math", "science", "planning", "research", "dialogue",
    }


def test_registry_carries_runtime_through_to_agents() -> None:
    sentinel = object()
    reg = AgentRegistry(runtime=sentinel)
    for agent in reg.all():
        assert agent.runtime is sentinel


def test_math_agent_solves_via_registry() -> None:
    reg = AgentRegistry()
    sol = reg.math.solve("What is 2 + 2?")
    assert sol.succeeded
    assert sol.answer == "4"
