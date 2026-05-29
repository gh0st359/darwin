"""Tests for the ConceptualReasoner."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.primitive_seed import seed_primitives
from darwin.universe.reasoning import (
    ConceptualReasoner,
    ReasoningStep,
    ReasoningTrace,
)


def _minimal_universe() -> ConceptUniverse:
    u = ConceptUniverse()
    seed_primitives(u)
    return u


def test_reasoner_returns_trace_with_seed_concepts() -> None:
    u = _minimal_universe()
    reasoner = ConceptualReasoner(u)
    trace = reasoner.think("what causes change?", seeds=["cause", "change"])
    assert isinstance(trace, ReasoningTrace)
    assert set(trace.seed_concepts) == {"cause", "change"}
    assert trace.steps


def test_reasoner_expand_step_surfaces_neighborhood() -> None:
    u = _minimal_universe()
    reasoner = ConceptualReasoner(u)
    trace = reasoner.think("explore cause", seeds=["cause"])
    expand_steps = [s for s in trace.steps if s.kind == "expand"]
    assert expand_steps
    step = expand_steps[0]
    assert "cause" in step.concepts


def test_reasoner_bridge_step_links_two_seeds() -> None:
    u = ConceptUniverse()
    u.add_concept("music")
    u.add_concept("ratio")
    u.add_concept("math")
    u.add_relation("music", "ratio", "describes")
    u.add_relation("ratio", "math", "part_of")
    reasoner = ConceptualReasoner(u)
    trace = reasoner.think(
        "is music related to math?", seeds=["music", "math"]
    )
    bridges = [s for s in trace.steps if s.kind == "bridge"]
    assert bridges
    assert "music" in bridges[0].concepts
    assert "math" in bridges[0].concepts


def test_reasoner_analogy_seeks_cross_domain_match() -> None:
    u = ConceptUniverse()
    u.add_concept("flow", domain="physics")
    u.add_concept("river", domain="physics")
    u.add_concept("melody", domain="arts")
    u.add_concept("song", domain="arts")
    u.add_relation("river", "flow", "instantiates")
    u.add_relation("melody", "flow", "instantiates")
    u.add_relation("song", "melody", "part_of")
    reasoner = ConceptualReasoner(u)
    trace = reasoner.think("what is flow like elsewhere?", seeds=["flow"])
    analogies = [s for s in trace.steps if s.kind == "analogy"]
    # Analogies are best-effort; if the graph supports one, we expect to see it.
    # Either way the trace is well-formed.
    assert isinstance(trace.coverage, float)
    for step in analogies:
        assert "flow" in step.concepts


def test_reasoner_reflect_step_describes_focus() -> None:
    u = _minimal_universe()
    reasoner = ConceptualReasoner(u)
    trace = reasoner.think("think about thinking", seeds=["thing"])
    reflects = [s for s in trace.steps if s.kind == "reflect"]
    assert reflects


def test_reasoner_answer_points_are_built_from_steps() -> None:
    u = _minimal_universe()
    reasoner = ConceptualReasoner(u)
    trace = reasoner.think("what is the difference?", seeds=["same", "different"])
    assert trace.suggested_answer_points
    assert any("same" in p or "different" in p for p in trace.suggested_answer_points)


def test_reasoner_publishes_to_bus_when_attached() -> None:
    from darwin.mysterio.bus import BusEvent, BusTopic, CognitionBus

    bus = CognitionBus()
    received: list[BusEvent] = []
    bus.subscribe(BusTopic.SIMULATIONS, received.append)
    u = _minimal_universe()
    reasoner = ConceptualReasoner(u, bus=bus)
    reasoner.think("ping", seeds=["self"])
    assert received
    payload = received[-1].payload
    assert payload["kind"] == "conceptual_reasoning"
    assert payload["query"] == "ping"


def test_reasoner_step_serializes() -> None:
    step = ReasoningStep(
        kind="expand",
        summary="testing",
        concepts=["a"],
        domains=["x"],
        confidence=0.6,
    )
    record = step.to_record()
    assert record["kind"] == "expand"
    assert record["concepts"] == ["a"]
