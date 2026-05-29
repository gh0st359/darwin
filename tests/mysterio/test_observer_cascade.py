"""Tests for the recursive theory-of-mind cascade."""

from __future__ import annotations

from darwin.mysterio.observer_cascade import CascadeLevel, ObserverCascade
from darwin.mysterio.observer_modeler import ObserverWorld


def test_cascade_constructs_to_configured_depth() -> None:
    world = ObserverWorld()
    cascade = ObserverCascade(world, max_depth=4)
    assert cascade.max_depth == 4
    assert len(cascade.levels) == 4
    assert all(isinstance(level, CascadeLevel) for level in cascade.levels)


def test_cascade_propagates_from_base_with_damping() -> None:
    world = ObserverWorld()
    world.note_command("/divergence")
    world.note_command("/rollback abc")
    cascade = ObserverCascade(world, max_depth=4)
    cascade.step()
    levels = cascade.levels
    # Level 0 mirrors the live ObserverWorld operator.
    op = world.operator()
    assert levels[0].entity.attention_level == op.attention_level
    # Deeper levels damp toward zero relative to level 0.
    assert levels[3].entity.attention_level <= levels[2].entity.attention_level
    assert levels[2].entity.attention_level <= levels[1].entity.attention_level
    assert levels[1].entity.attention_level <= levels[0].entity.attention_level


def test_cascade_grow_extends_depth() -> None:
    cascade = ObserverCascade(ObserverWorld(), max_depth=4)
    cascade.grow(by=2)
    assert cascade.max_depth == 6
    assert len(cascade.levels) == 6


def test_belief_at_returns_clamped_level() -> None:
    cascade = ObserverCascade(ObserverWorld(), max_depth=4)
    assert cascade.belief_at(0).depth == 0
    assert cascade.belief_at(99).depth == 3
    assert cascade.belief_at(-5).depth == 0


def test_snapshot_contains_every_level() -> None:
    cascade = ObserverCascade(ObserverWorld(), max_depth=4)
    snap = cascade.snapshot()
    assert snap["max_depth"] == 4
    assert len(snap["levels"]) == 4
    assert snap["levels"][3]["depth"] == 3
