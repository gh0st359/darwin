"""Tests for the generative meta-proposer."""

from __future__ import annotations

from collections import Counter

from darwin.agent import Darwin
from darwin.mysterio.meta_proposer import MetaProposer, MetaProposerContext
from darwin.mysterio.safety import MutationKind
from darwin.types import Action, Transition


def _seed() -> Darwin:
    darwin = Darwin(actions=[Action("flip_switch"), Action("open_curtains")])
    for index in range(8):
        darwin.learn(
            Transition(
                before={"switch_on": False, "room_bright": False, "daylight": True},
                action="flip_switch",
                after={"switch_on": True, "room_bright": True, "daylight": True},
                reward=1.0,
                t=index,
            )
        )
    return darwin


def test_default_strategies_are_registered() -> None:
    mp = MetaProposer()
    names = set(mp.strategies())
    assert "variable_driven" in names
    assert "hypothesis_driven" in names
    assert "loop_starvation" in names
    assert "consolidation" in names
    assert "gate_evolution" in names


def test_propose_returns_typed_proposals_with_specs() -> None:
    darwin = _seed()
    # Force some prediction failures so variable_driven has material.
    darwin.self_model.prediction_failures["flip_switch:room_bright"] = 4
    darwin.self_model.prediction_failures["open_curtains:room_bright"] = 2

    mp = MetaProposer()
    ctx = MetaProposerContext(
        darwin=darwin,
        runtime=None,
        recent_outcomes=[],
        last_simulation=None,
        last_uncertainty_scan=None,
    )
    proposals = mp.propose(ctx)
    # At least the variable_driven and consolidation strategies should fire.
    assert len(proposals) >= 1
    for proposal in proposals:
        spec = getattr(proposal, "spec", None)
        assert spec is not None
        assert isinstance(spec.kind, MutationKind)
        assert spec.touches  # must declare at least one touch
        assert spec.introspection_signature


def test_dedup_via_signature() -> None:
    darwin = _seed()
    darwin.self_model.prediction_failures["flip_switch:room_bright"] = 4
    mp = MetaProposer()
    ctx = MetaProposerContext(
        darwin=darwin,
        runtime=None,
        recent_outcomes=[],
        last_simulation=None,
        last_uncertainty_scan=None,
    )
    first_round = mp.propose(ctx)
    second_round = mp.propose(ctx)
    # Same conditions produce the same signatures; second round drops dupes.
    first_signatures = {p.spec.introspection_signature for p in first_round if p.spec}
    second_signatures = {p.spec.introspection_signature for p in second_round if p.spec}
    overlap = first_signatures & second_signatures
    # Any overlap should have been deduped (so overlap is empty).
    assert overlap == set()


def test_mutation_kind_diversity_across_strategies() -> None:
    """Across the v6 default strategies, multiple MutationKinds should be reached
    once the substrate is rich enough."""
    darwin = _seed()
    darwin.self_model.prediction_failures["flip_switch:room_bright"] = 4

    import time as _time

    now = _time.time()
    class _FakeRuntime:
        loop_intervals = {"experiment": 2.0, "simulation": 3.0}
        _loop_state = {
            "experiment": {"last_time": now - 1000.0},  # very stale → starved
            "simulation": {"last_time": now},  # fresh
        }
        last_simulation = None
        last_uncertainty_scan = None

        class _MG:
            current = None
            history: list = []

        meta_gate = _MG()

    mp = MetaProposer()
    ctx = MetaProposerContext(
        darwin=darwin,
        runtime=_FakeRuntime(),
        recent_outcomes=[],
        last_simulation=None,
        last_uncertainty_scan=None,
    )
    proposals = mp.propose(ctx)
    kinds = Counter(p.spec.kind for p in proposals if p.spec)
    # At minimum RULE (variable_driven / consolidation) and KERNEL (loop_starvation)
    # should appear given the starved-loop fixture.
    assert MutationKind.RULE in kinds
    assert MutationKind.KERNEL in kinds
