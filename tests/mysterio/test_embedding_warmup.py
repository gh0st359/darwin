"""Tests for self-trained causal embeddings.

The embedding space is Darwin's own vocabulary — no pretrained weights. Tokens
that co-occur in transitions should drift toward each other; unrelated tokens
should not.
"""

from __future__ import annotations

import math

from darwin.mysterio.embeddings import (
    CausalEmbeddingSpace,
    cosine,
    tokens_for_transition,
)


def test_tokens_for_transition_canonical_order() -> None:
    toks = tokens_for_transition(
        before={"b": 1, "a": 0},
        action="flip",
        after={"a": 1, "b": 1},
    )
    assert toks[0] == "act:flip"
    pre = [t for t in toks if t.startswith("pre:")]
    post = [t for t in toks if t.startswith("post:")]
    assert pre == ["pre:a=0", "pre:b=1"]
    assert post == ["post:a=1", "post:b=1"]


def test_seed_initialization_is_deterministic() -> None:
    a = CausalEmbeddingSpace(seed=42)
    b = CausalEmbeddingSpace(seed=42)
    assert a.embed("act:open_curtains") == b.embed("act:open_curtains")


def test_observe_trains_and_grows_vocab() -> None:
    space = CausalEmbeddingSpace(dim=16, seed=7)
    assert space.vocab_size() == 0
    for _ in range(20):
        space.observe(
            before={"room_bright": False, "switch_on": False},
            action="flip_switch",
            after={"room_bright": True, "switch_on": True},
        )
    # 1 action + 2 pre tokens + 2 post tokens = 5
    assert space.vocab_size() == 5
    assert space.stats()["train_steps"] > 0


def test_co_occurring_tokens_drift_closer_than_unrelated() -> None:
    space = CausalEmbeddingSpace(dim=24, seed=11, learning_rate=0.2, negatives=3)
    # Two distinct transition families that never co-occur.
    for _ in range(120):
        space.observe(
            before={"room_bright": False},
            action="open_curtains",
            after={"room_bright": True},
        )
        space.observe(
            before={"fuse_intact": True},
            action="check_fuse",
            after={"fuse_intact": True},
        )

    sim_related = cosine(
        space.embed("act:open_curtains"),
        space.embed("post:room_bright=True"),
    )
    sim_unrelated = cosine(
        space.embed("act:open_curtains"),
        space.embed("post:fuse_intact=True"),
    )
    assert sim_related > sim_unrelated


def test_nearest_returns_top_k() -> None:
    space = CausalEmbeddingSpace(dim=16, seed=13, learning_rate=0.15)
    for _ in range(60):
        space.observe(
            before={"x": 0},
            action="advance",
            after={"x": 1},
        )
    near = space.nearest("act:advance", k=2)
    assert len(near) == 2
    # Each entry is (token, cosine_similarity).
    for token, score in near:
        assert isinstance(token, str)
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6


def test_checkpoint_hash_changes_when_state_changes() -> None:
    space = CausalEmbeddingSpace(dim=8, seed=3)
    h0 = space.checkpoint_hash()
    space.observe(before={"a": 0}, action="x", after={"a": 1})
    h1 = space.checkpoint_hash()
    assert h0 != h1


def test_no_external_weights_ever_required() -> None:
    """Pure-python init must work with zero optional deps importable."""

    space = CausalEmbeddingSpace()
    # Backend label is one of the two declared modes; pure-python is reference.
    assert space.backend in {"python", "torch"}
    vec = space.embed("act:reflect")
    # Vector has the configured dimensionality and finite magnitude.
    assert len(vec) == space.dim
    assert all(math.isfinite(x) for x in vec)
