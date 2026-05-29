"""Tests for the grounded/interior track partition.

The v7 invariant: writes to the interior track must not affect the grounded
substrate. We assert this by running the same grounded transitions through a
control darwin (no interior writes) and a treatment darwin (1000 interior
transitions interleaved with the same grounded stream) and comparing the
resulting causal-model state.
"""

from __future__ import annotations

import hashlib
import json

from darwin.agent import Darwin
from darwin.mysterio.tracks import (
    GROUNDED_TRACK,
    INTERIOR_TRACK,
    PRIVATE_SELF_TRACK,
    PUBLIC_TRACK,
    TrackRegistry,
    TrackedSubstrate,
    track_of,
)
from darwin.types import Action, Transition


def _grounded_transitions() -> list[Transition]:
    return [
        Transition(
            before={"light": False, "switch": False},
            action="flip",
            after={"light": True, "switch": True},
            reward=1.0,
            t=i,
        )
        for i in range(40)
    ]


def _interior_transitions(start_t: int = 10_000_000) -> list[Transition]:
    return [
        Transition(
            before={"uncertainty": 0.5, "attention": 0.2},
            action="probe",
            after={"uncertainty": 0.3, "attention": 0.4},
            reward=0.1,
            t=start_t + i,
            metadata={"track": INTERIOR_TRACK, "mode": "interior_simulation"},
        )
        for i in range(1000)
    ]


def _causal_signature(darwin: Darwin) -> str:
    beliefs = darwin.causal_model.beliefs(limit=10000)
    payload = json.dumps(
        sorted(
            [
                {
                    "action": getattr(b, "action", ""),
                    "variable": getattr(b, "variable", ""),
                    "effect": getattr(b, "effect", ""),
                    "confidence": round(float(getattr(b, "confidence", 0.0)), 6),
                    "samples": int(getattr(b, "samples", 0)),
                }
                for b in beliefs
            ],
            key=lambda d: (d["action"], d["variable"], d["effect"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_legacy_track_aliases_resolve_to_new_names() -> None:
    assert PUBLIC_TRACK == GROUNDED_TRACK
    assert PRIVATE_SELF_TRACK == INTERIOR_TRACK


def test_track_of_handles_default_and_legacy_strings() -> None:
    bare = Transition(before={}, action="x", after={}, reward=0.0, t=0)
    assert track_of(bare) == GROUNDED_TRACK
    legacy_public = Transition(
        before={}, action="x", after={}, reward=0.0, t=1, metadata={"track": "public"}
    )
    assert track_of(legacy_public) == GROUNDED_TRACK
    legacy_private = Transition(
        before={},
        action="x",
        after={},
        reward=0.0,
        t=2,
        metadata={"track": "private_self"},
    )
    assert track_of(legacy_private) == INTERIOR_TRACK
    explicit_interior = Transition(
        before={},
        action="x",
        after={},
        reward=0.0,
        t=3,
        metadata={"track": INTERIOR_TRACK},
    )
    assert track_of(explicit_interior) == INTERIOR_TRACK


def test_interior_writes_do_not_touch_grounded_substrate() -> None:
    """Control vs treatment: identical grounded stream, treatment additionally
    receives 1000 interior transitions. The grounded causal-model signature
    must be byte-equal across the two runs."""

    actions = [Action("flip", cost=0.0, description="flip the switch")]

    control = Darwin(actions=list(actions), seed=7)
    for transition in _grounded_transitions():
        control.learn(transition, persist=False)

    treatment = Darwin(actions=list(actions), seed=7)
    interior = _interior_transitions()
    grounded_iter = iter(_grounded_transitions())
    # Interleave: every grounded transition, then a burst of 25 interior writes.
    while True:
        try:
            t = next(grounded_iter)
        except StopIteration:
            break
        treatment.learn(t, persist=False)
        for _ in range(25):
            if interior:
                treatment.learn(interior.pop(0), persist=False)
    # Drain any leftover interior transitions.
    for t in interior:
        treatment.learn(t, persist=False)

    assert _causal_signature(control) == _causal_signature(treatment)


def test_interior_substrate_accumulates_beliefs_independently() -> None:
    darwin = Darwin(
        actions=[Action("probe", cost=0.0, description="probe internal state")],
        seed=11,
    )
    for transition in _interior_transitions(start_t=0):
        darwin.learn(transition, persist=False)
    interior = darwin.tracks.get(INTERIOR_TRACK)
    assert interior.learned_count == 1000
    # The grounded substrate must have received zero learning signals.
    assert darwin.causal_model.total_observations() == 0


def test_track_registry_lifecycle() -> None:
    registry = TrackRegistry()
    assert not registry.has("interior")
    substrate = registry.get("interior")
    assert isinstance(substrate, TrackedSubstrate)
    assert registry.has("interior")
    assert registry.names() == ["interior"]
    summary = registry.summaries()[0]
    assert summary["track"] == "interior"
