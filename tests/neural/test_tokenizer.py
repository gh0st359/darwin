"""Tokenizer determinism, merge growth, persistence."""

from __future__ import annotations

from darwin.neural.tokenizer import NeuralTokenizer, split_words


def test_split_words_lowercase_and_strip_punctuation():
    assert split_words("The Quick brown fox, jumped!") == [
        "the", "quick", "brown", "fox", "jumped",
    ]


def test_split_words_keeps_intra_word_hyphen_apostrophe_underscore():
    assert split_words("self-modifying don't can_do") == [
        "self-modifying", "don't", "can_do",
    ]


def test_tokenize_is_deterministic_without_merges():
    tok = NeuralTokenizer()
    a = tok.tokenize("Darwin learns from text.")
    b = tok.tokenize("Darwin learns from text.")
    assert a == b == ["darwin", "learns", "from", "text"]


def test_token_counts_grow_with_chunks():
    tok = NeuralTokenizer()
    tok.tokenize("a b c")
    tok.tokenize("a c d")
    assert tok.token_counts["a"] == 2
    assert tok.token_counts["d"] == 1
    assert tok.vocab_size() == 4


def test_learn_merges_picks_most_frequent_pair():
    tok = NeuralTokenizer()
    corpus = ["new york is large"] * 3 + ["new haven is small"]
    added = tok.learn_merges(corpus, max_merges=1)
    assert added == 1
    assert tok.merges[0] == ("new", "york")


def test_apply_merges_after_learning():
    tok = NeuralTokenizer()
    tok.learn_merges(["new york new york new york new york"], max_merges=1)
    out = tok.tokenize("new york is great")
    assert out[0] == "newyork"
    assert "is" in out
    assert "great" in out


def test_persistence_round_trip(tmp_path):
    tok = NeuralTokenizer()
    tok.learn_merges(["a b a b a b"], max_merges=1)
    tok.tokenize("a b c")
    path = tmp_path / "tok.json"
    tok.save(path)
    restored = NeuralTokenizer.load(path)
    assert restored.merges == tok.merges
    assert restored.token_counts == tok.token_counts
    assert restored.vocab_size() == tok.vocab_size()
