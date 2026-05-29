"""Tests for the primitive seed — Darwin's only hardcoded content."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.primitive_seed import (
    primitive_names,
    seed_primitives,
)


def test_primitive_seed_loads_meta_vocabulary() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    summary = u.summary()
    # The seed is intentionally small — meta-vocabulary only.
    assert 20 <= summary["concepts"] <= 60
    # And it must include the structural backbone Darwin reasons with.
    for name in [
        "thing", "change", "cause", "effect", "self", "model",
        "same", "different", "true", "false", "infer", "compose",
        "more", "less", "question", "answer",
    ]:
        assert u.has(name), f"missing primitive: {name}"


def test_primitive_seed_does_not_introduce_domain_facts() -> None:
    """The seed must not hardcode encyclopedic knowledge like 'gravity is_a force'."""

    u = ConceptUniverse()
    seed_primitives(u)
    for forbidden in [
        "gravity", "music", "math", "physics", "cell", "atom",
        "dna", "harmony", "consciousness", "neural_network",
        "calculus", "metaphor", "art",
    ]:
        assert not u.has(forbidden), (
            f"primitive seed leaked domain concept: {forbidden!r}"
        )


def test_primitive_seed_is_idempotent() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    first = u.summary()
    seed_primitives(u)
    second = u.summary()
    assert first["concepts"] == second["concepts"]
    assert first["relations"] == second["relations"]


def test_primitive_names_matches_seed_concepts() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    for name in primitive_names():
        assert u.has(name)


def test_inference_operators_are_present_as_primitives() -> None:
    """Generalize / specialize / compose / decompose / infer / contradict are
    the gears of conceptual derivation. They must be primitives."""

    u = ConceptUniverse()
    seed_primitives(u)
    for op in ["generalize", "specialize", "compose", "decompose", "infer", "contradict"]:
        assert u.has(op)
