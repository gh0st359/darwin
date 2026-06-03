"""Tests for the multi-process integration."""

from __future__ import annotations

from darwin.agents.registry import AgentRegistry
from darwin.scale.multiprocess import _agent_loop, agent_subsystem_specs


def test_specs_built_for_six_agents() -> None:
    reg = AgentRegistry()
    specs = agent_subsystem_specs(reg)
    assert len(specs) == 6
    names = {s.name for s in specs}
    assert names == {
        "agent_code", "agent_math", "agent_science",
        "agent_planning", "agent_research", "agent_dialogue",
    }


def test_specs_target_agent_loop_entrypoint() -> None:
    reg = AgentRegistry()
    specs = agent_subsystem_specs(reg)
    for spec in specs:
        assert spec.entrypoint == "darwin.scale.multiprocess:_agent_loop"
        assert "agent_solve" in spec.topics


def test_specs_resolve_to_callable() -> None:
    reg = AgentRegistry()
    specs = agent_subsystem_specs(reg)
    for spec in specs:
        callable_obj = spec.resolve()
        assert callable(callable_obj)


def test_no_registry_yields_empty() -> None:
    assert agent_subsystem_specs(None) == []


def test_agent_loop_stub_returns_cleanly() -> None:
    # The stub should exit without error.
    _agent_loop(agent_name="code")
    _agent_loop(agent_name="math")
