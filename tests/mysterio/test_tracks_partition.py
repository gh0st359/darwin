"""Track partition is absolute: private cognition never pollutes public belief."""

from __future__ import annotations

from darwin.agent import Darwin
from darwin.mysterio.tracks import PRIVATE_SELF_TRACK, PUBLIC_TRACK, track_of
from darwin.types import Action, Transition


def _public_fingerprint(darwin: Darwin) -> tuple:
    causal = darwin.causal_model
    beliefs = tuple(
        (b.action, b.variable, b.effect, round(float(b.confidence), 6), int(b.samples))
        for b in causal.beliefs(limit=1000)
    )
    return (
        causal.total_observations(),
        causal.min_samples,
        len(darwin.memory.episodes),
        beliefs,
    )


def test_track_of_defaults_public() -> None:
    t = Transition(before={}, action="a", after={}, reward=0.0, t=0)
    assert track_of(t) == PUBLIC_TRACK
    t2 = Transition(
        before={}, action="a", after={}, reward=0.0, t=0,
        metadata={"track": "private_self"},
    )
    assert track_of(t2) == "private_self"


def test_private_learning_does_not_touch_public_models() -> None:
    darwin = Darwin(actions=[Action("flip")])
    for i in range(10):
        darwin.learn(
            Transition(before={"on": False}, action="flip", after={"on": True}, reward=1.0, t=i)
        )
    baseline = _public_fingerprint(darwin)

    for i in range(1000):
        darwin.learn(
            Transition(
                before={"phantom": i % 3},
                action="flip",
                after={"phantom": (i + 1) % 3, "imagined": True},
                reward=float(i % 2),
                t=100000 + i,
                metadata={"track": PRIVATE_SELF_TRACK},
            )
        )

    after = _public_fingerprint(darwin)
    assert after == baseline, "private cognition polluted the public substrate"
    assert darwin.tracks.get(PRIVATE_SELF_TRACK).learned_count == 1000


def test_control_equivalence() -> None:
    """Public state byte-identical to a control run with no private loops."""
    def build(public_only: bool) -> Darwin:
        d = Darwin(actions=[Action("go")])
        for i in range(20):
            d.learn(
                Transition(
                    before={"s": i % 2},
                    action="go",
                    after={"s": (i + 1) % 2},
                    reward=1.0,
                    t=i,
                )
            )
            if not public_only:
                d.learn(
                    Transition(
                        before={"z": True}, action="go", after={"z": False}, reward=0.0,
                        t=50000 + i, metadata={"track": PRIVATE_SELF_TRACK},
                    )
                )
        return d

    control = build(public_only=True)
    with_private = build(public_only=False)
    assert _public_fingerprint(control) == _public_fingerprint(with_private)


def test_multiple_private_tracks_isolated() -> None:
    darwin = Darwin(actions=[Action("a")])
    darwin.learn(Transition(before={}, action="a", after={}, reward=0.0, t=0, metadata={"track": "dream"}))
    darwin.learn(Transition(before={}, action="a", after={}, reward=0.0, t=1, metadata={"track": "private_self"}))
    assert set(darwin.tracks.names()) == {"dream", "private_self"}
    assert darwin.tracks.get("dream").learned_count == 1
    assert darwin.tracks.get("private_self").learned_count == 1
