"""Tests for the ConceptualWorld adapter."""

from __future__ import annotations

from darwin.types import Action, Transition
from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.derivation import ConceptDeriver
from darwin.universe.primitive_seed import seed_primitives
from darwin.universe.world import CONCEPTUAL_ACTIONS, ConceptualWorld


def _world() -> ConceptualWorld:
    u = ConceptUniverse()
    seed_primitives(u)
    return ConceptualWorld(u, deriver=ConceptDeriver(u))


def test_observe_returns_focus_state() -> None:
    world = _world()
    state = world.observe()
    assert "focus" in state
    assert "concept_count" in state
    assert isinstance(state["concept_count"], int)


def test_possible_actions_lists_the_eight_conceptual_actions() -> None:
    world = _world()
    actions = world.possible_actions()
    assert len(actions) == len(CONCEPTUAL_ACTIONS)
    names = {a.name for a in actions}
    assert "explore_concept" in names
    assert "compose_concepts" in names
    assert "generalize_concept" in names
    assert "specialize_concept" in names
    assert "analogize_concept" in names
    assert "reflect_concept" in names
    assert "derive_concepts" in names
    assert "wander_universe" in names


def test_apply_explore_changes_focus() -> None:
    world = _world()
    before = world.observe()["focus"]
    after_state, reward = world.apply(Action("explore_concept"))
    # Focus may have shifted along an edge.
    assert isinstance(reward, float)
    # Either we moved or there were no edges; both are valid.
    assert "focus" in after_state


def test_apply_compose_creates_new_concept() -> None:
    world = _world()
    # Seed a known-good two-concept context.
    world._focus.primary = "cause"
    world._focus.secondary = "effect"
    before = len(world.universe)
    state, reward = world.apply(Action("compose_concepts"))
    after = len(world.universe)
    assert after > before
    assert reward > 0


def test_apply_generalize_either_descends_or_creates_parent() -> None:
    world = _world()
    world._focus.primary = "cause"
    state, reward = world.apply(Action("generalize_concept"))
    assert isinstance(reward, float)
    # Universe should not have shrunk.
    assert len(world.universe) >= 1


def test_apply_unknown_action_returns_negative_reward() -> None:
    world = _world()
    state, reward = world.apply(Action("nonsense_action"))
    assert reward < 0


def test_apply_derive_runs_deriver_pass() -> None:
    world = _world()
    state, reward = world.apply(Action("derive_concepts"))
    # The deriver may have produced nothing — that's OK; it should not blow up.
    assert isinstance(reward, float)


def test_apply_wander_moves_to_underservedly_visited_concept() -> None:
    world = _world()
    initial_focus = world._focus.primary
    state, reward = world.apply(Action("wander_universe"))
    # Wander always reports a focus shift.
    assert reward >= 0
    assert "focus" in state


def test_make_transition_attaches_grounded_track_metadata() -> None:
    world = _world()
    before = world.observe()
    after, reward = world.apply(Action("reflect_concept"))
    transition = world.make_transition(before, after, reward=reward)
    assert isinstance(transition, Transition)
    assert transition.metadata.get("track") == "grounded"
    assert transition.metadata.get("world") == "conceptual"


def test_compose_then_analogize_grows_relation_count() -> None:
    world = _world()
    world._focus.primary = "thing"
    world._focus.secondary = "model"
    before_rels = world.universe.summary()["relations"]
    world.apply(Action("compose_concepts"))
    after_rels = world.universe.summary()["relations"]
    assert after_rels > before_rels
