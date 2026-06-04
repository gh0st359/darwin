"""LearnedCausalSpace — SGNS learning, back-compat surface, persistence."""

from __future__ import annotations

import math

from darwin.neural.learned_space import LearnedCausalSpace


def test_default_dim_is_128():
    s = LearnedCausalSpace()
    assert s.dim == 128


def test_observe_transition_back_compat():
    s = LearnedCausalSpace(dim=16)

    class T:
        before = {"x": 1}
        action = "step"
        after = {"y": 2}

    s.observe_transition(T())
    # Three tokens were registered: act:step, pre:x=1, post:y=2.
    assert s.vocab_size() == 3
    assert s._train_steps > 0


def test_train_tokens_grows_vocab_and_steps():
    s = LearnedCausalSpace(dim=16)
    s.train_tokens(["alpha", "beta", "gamma"])
    assert s.vocab_size() == 3
    assert s._train_steps > 0


def test_train_tokens_no_op_on_short_input():
    s = LearnedCausalSpace(dim=8)
    s.train_tokens([])
    s.train_tokens(["solo"])
    assert s._train_steps == 0


def test_loss_decreases_over_repeated_co_occurrence():
    s = LearnedCausalSpace(
        dim=16,
        learning_rate=0.1,
        negatives=2,
        seed=42,
        subsample_threshold=1.0,  # disable subsampling for stability
    )
    pair = ["apple", "fruit"]
    # Train many iterations on the same pair; loss EWMA should drop.
    for _ in range(10):
        s.train_tokens(pair)
    early = s.light_stats()["loss_ewma"]
    for _ in range(200):
        s.train_tokens(pair)
    late = s.light_stats()["loss_ewma"]
    assert late < early


def test_related_tokens_become_nearer_than_unrelated():
    s = LearnedCausalSpace(
        dim=24, learning_rate=0.1, negatives=3, seed=11,
        subsample_threshold=1.0,
    )
    # Train "cat" and "dog" together; train "rocket" alone with another word.
    for _ in range(150):
        s.train_tokens(["cat", "dog", "pet"])
        s.train_tokens(["rocket", "space", "launch"])
    sim_pet = _cos(s.embed("cat"), s.embed("dog"))
    sim_mix = _cos(s.embed("cat"), s.embed("rocket"))
    assert sim_pet > sim_mix


def test_checkpoint_hash_is_stable_across_recomputation():
    s = LearnedCausalSpace(dim=8, seed=99)
    s.train_tokens(["a", "b", "c", "d"])
    h1 = s.checkpoint_hash()
    h2 = s.checkpoint_hash()
    assert h1 == h2 and len(h1) == 64


def test_checkpoint_hash_changes_after_more_training():
    s = LearnedCausalSpace(dim=8, seed=99)
    s.train_tokens(["a", "b", "c"])
    h1 = s.checkpoint_hash()
    for _ in range(20):
        s.train_tokens(["a", "b", "c"])
    assert s.checkpoint_hash() != h1


def test_save_and_load_round_trip(tmp_path):
    s = LearnedCausalSpace(dim=12, seed=1)
    for _ in range(5):
        s.train_tokens(["x", "y", "z"])
    path = tmp_path / "state.json"
    h = s.save(path)
    s2 = LearnedCausalSpace(dim=12, seed=1)
    s2.load(path)
    assert s2.vocab_size() == s.vocab_size()
    assert s2.checkpoint_hash() == h


def test_lr_decays_over_horizon():
    s = LearnedCausalSpace(
        dim=4, learning_rate=0.1, min_learning_rate=0.001,
        decay_horizon=10, seed=5,
    )
    lr0 = s._current_lr()
    for _ in range(50):
        s.train_tokens(["p", "q", "r"])
    lr1 = s._current_lr()
    assert lr1 < lr0


def test_nearest_returns_other_tokens_with_finite_scores():
    s = LearnedCausalSpace(dim=8, seed=3)
    s.train_tokens(["one", "two", "three", "four"])
    nearest = s.nearest("one", k=2)
    assert len(nearest) == 2
    for tok, score in nearest:
        assert tok != "one"
        assert math.isfinite(score)


def test_env_var_overrides_dim(monkeypatch):
    monkeypatch.setenv("DARWIN_NEURAL_DIM", "64")
    s = LearnedCausalSpace()
    assert s.dim == 64


def test_stats_includes_loss_and_lr():
    s = LearnedCausalSpace(dim=8)
    s.train_tokens(["foo", "bar"])
    st = s.stats()
    for key in ("backend", "dim", "vocab_size", "loss_ewma", "lr", "checkpoint_hash"):
        assert key in st


def _cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return s / (na * nb)
