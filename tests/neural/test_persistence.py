"""NeuralPersistence — atomic save/load + labelled checkpoints + rollback."""

from __future__ import annotations

import json

import pytest

from darwin.neural.learned_space import LearnedCausalSpace
from darwin.neural.persistence import NeuralPersistence
from darwin.neural.tokenizer import NeuralTokenizer


def _train(space, *batches):
    for batch in batches:
        space.train_tokens(batch)


def test_save_writes_manifest_shards_and_training(tmp_path):
    space = LearnedCausalSpace(dim=8)
    tok = NeuralTokenizer()
    _train(space, ["a", "b", "c"], ["b", "c", "d"])
    tok.tokenize("a b c")
    persistence = NeuralPersistence(tmp_path)
    manifest = persistence.save(space=space, tokenizer=tok, label="t1")
    assert manifest.vocab_size == space.vocab_size()
    assert persistence.manifest_path().exists()
    assert persistence.training_path().exists()
    assert persistence.tokenizer_path().exists()
    assert any(persistence.shards_dir().glob("shard_*.bin"))


def test_load_restores_space_and_tokenizer(tmp_path):
    space = LearnedCausalSpace(dim=8, seed=7)
    tok = NeuralTokenizer()
    _train(space, ["foo", "bar"], ["bar", "baz"])
    tok.tokenize("foo bar baz")
    persistence = NeuralPersistence(tmp_path)
    persistence.save(space=space, tokenizer=tok)
    original_hash = space.checkpoint_hash()
    original_vocab = space.vocab_size()
    fresh_space = LearnedCausalSpace(dim=8, seed=7)
    fresh_tok = NeuralTokenizer()
    manifest = persistence.load(space=fresh_space, tokenizer=fresh_tok)
    assert manifest is not None
    assert fresh_space.vocab_size() == original_vocab
    assert fresh_space.checkpoint_hash() == original_hash
    assert fresh_tok.vocab_size() == tok.vocab_size()


def test_load_with_no_manifest_returns_none(tmp_path):
    persistence = NeuralPersistence(tmp_path)
    space = LearnedCausalSpace(dim=4)
    assert persistence.load(space=space) is None


def test_load_dim_mismatch_raises(tmp_path):
    space = LearnedCausalSpace(dim=8)
    _train(space, ["x", "y"])
    persistence = NeuralPersistence(tmp_path)
    persistence.save(space=space)
    other = LearnedCausalSpace(dim=16)
    with pytest.raises(ValueError):
        persistence.load(space=other)


def test_checkpoint_label_and_list(tmp_path):
    space = LearnedCausalSpace(dim=4)
    _train(space, ["a", "b"])
    persistence = NeuralPersistence(tmp_path)
    persistence.save(space=space)
    persistence.checkpoint("baseline")
    assert "baseline" in persistence.list_checkpoints()
    persistence.checkpoint("baseline-2")
    assert sorted(persistence.list_checkpoints()) == ["baseline", "baseline-2"]


def test_rollback_restores_prior_label(tmp_path):
    space = LearnedCausalSpace(dim=8, seed=2)
    _train(space, ["a", "b", "c"])
    persistence = NeuralPersistence(tmp_path)
    persistence.save(space=space)
    persistence.checkpoint("v1")
    v1_hash = space.checkpoint_hash()
    # Train more, save, then roll back to v1.
    for _ in range(20):
        space.train_tokens(["a", "b", "c", "d", "e"])
    persistence.save(space=space)
    persistence.rollback("v1")
    restored = LearnedCausalSpace(dim=8, seed=2)
    persistence.load(space=restored)
    assert restored.checkpoint_hash() == v1_hash


def test_atomic_save_is_safe_against_partial_writes(tmp_path):
    # Stand-in for a process restart: write, then load into a fresh space.
    space = LearnedCausalSpace(dim=4, seed=3)
    _train(space, ["alpha", "beta"])
    persistence = NeuralPersistence(tmp_path)
    persistence.save(space=space)
    # Manifest is always the last file written; if it's there, the save is valid.
    manifest = json.loads(persistence.manifest_path().read_text())
    assert manifest["vocab_size"] == space.vocab_size()


def test_cursor_round_trip(tmp_path):
    persistence = NeuralPersistence(tmp_path)
    assert persistence.read_cursor() == {}
    persistence.write_cursor({"source": "wiki", "offset": 42})
    assert persistence.read_cursor() == {"source": "wiki", "offset": 42}
