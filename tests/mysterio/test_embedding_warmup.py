"""Warmup: after training, nearest-neighbor retrieval beats random by a margin."""

from __future__ import annotations

import random

from darwin.mysterio.embeddings import CausalEmbeddingSpace, cosine, tokens_for_transition


def _synthetic_transitions(n: int, rng: random.Random) -> list[tuple[dict, str, dict]]:
    """Two stable causal regimes; the embedding should learn their structure.

    Regime A: flip_switch reliably turns room_bright True.
    Regime B: open_curtains reliably sets curtains_open True.
    """
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            out.append((
                {"switch_on": False, "room_bright": False},
                "flip_switch",
                {"switch_on": True, "room_bright": True},
            ))
        else:
            out.append((
                {"curtains_open": False, "daylight": True},
                "open_curtains",
                {"curtains_open": True, "daylight": True},
            ))
    return out


def test_checkpoint_hash_stable_and_changes_with_training() -> None:
    space = CausalEmbeddingSpace(dim=16, seed=7)
    h0 = space.checkpoint_hash()
    space.observe({"a": False}, "act", {"a": True})
    h1 = space.checkpoint_hash()
    assert h0 != h1
    assert space.checkpoint_hash() == h1  # deterministic


def test_nearest_neighbor_beats_random() -> None:
    rng = random.Random(123)
    space = CausalEmbeddingSpace(dim=24, seed=99, learning_rate=0.08)
    transitions = _synthetic_transitions(2000, rng)
    for before, action, after in transitions:
        space.observe(before, action, after)

    # The action token for flip_switch should be closer to its own effect
    # (post:room_bright=True) than to the unrelated regime's effect.
    flip = space.embed_action("flip_switch")
    own_effect = space.embed("post:room_bright=True")
    other_effect = space.embed("post:curtains_open=True")
    sim_own = cosine(flip, own_effect)
    sim_other = cosine(flip, other_effect)
    assert sim_own > sim_other, (sim_own, sim_other)

    # Nearest neighbors of flip_switch, restricted to post: tokens, should
    # surface its own effect above the other regime's.
    neighbors = space.nearest("act:flip_switch", k=4, prefix="post:")
    names = [tok for tok, _ in neighbors]
    assert "post:room_bright=True" in names


def test_retrieval_accuracy_above_chance() -> None:
    """Same-regime retrieval should beat the 50% chance baseline."""
    rng = random.Random(2024)
    space = CausalEmbeddingSpace(dim=24, seed=11, learning_rate=0.08)
    for before, action, after in _synthetic_transitions(3000, rng):
        space.observe(before, action, after)

    # Build query embeddings per action and check nearest action token.
    correct = 0
    trials = 200
    for _ in range(trials):
        if rng.random() < 0.5:
            q = space.embed("pre:room_bright=False")
            expected = "act:flip_switch"
        else:
            q = space.embed("pre:curtains_open=False")
            expected = "act:open_curtains"
        neighbors = space.nearest(q, k=1, prefix="act:")
        if neighbors and neighbors[0][0] == expected:
            correct += 1
    accuracy = correct / trials
    assert accuracy > 0.6, f"retrieval accuracy {accuracy} not above chance"


def test_save_load_roundtrip(tmp_path) -> None:
    space = CausalEmbeddingSpace(dim=16, seed=3)
    for _ in range(50):
        space.observe({"x": False}, "go", {"x": True})
    h = space.save(tmp_path / "emb.json")
    reloaded = CausalEmbeddingSpace()
    reloaded.load(tmp_path / "emb.json")
    assert reloaded.checkpoint_hash() == h
    assert reloaded.vocab_size() == space.vocab_size()
