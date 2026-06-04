"""CooccurrenceWindow — pair emission."""

from __future__ import annotations

from darwin.neural.cooccurrence import CooccurrenceWindow


def test_first_token_emits_no_pairs():
    w = CooccurrenceWindow(window=3)
    assert w.push("a") == []


def test_second_token_emits_one_pair_distance_one():
    w = CooccurrenceWindow(window=3)
    w.push("a")
    pairs = w.push("b")
    assert len(pairs) == 1
    assert pairs[0].center == "b"
    assert pairs[0].context == "a"
    assert pairs[0].distance == 1


def test_window_bounds_emission():
    w = CooccurrenceWindow(window=2)
    for t in ["a", "b", "c", "d"]:
        last = w.push(t)
    # On push("d"), buffer was [b, c] → pairs (d,c,1), (d,b,2)
    assert [(p.context, p.distance) for p in last] == [("c", 1), ("b", 2)]


def test_push_stream_iterates_all_pairs():
    w = CooccurrenceWindow(window=2)
    pairs = list(w.push_stream(["a", "b", "c", "d"]))
    # a → 0 pairs; b → (b,a,1); c → (c,b,1),(c,a,2); d → (d,c,1),(d,b,2)
    assert len(pairs) == 5


def test_reset_clears_buffer():
    w = CooccurrenceWindow(window=3)
    w.push("a")
    w.push("b")
    assert w.size() == 2
    w.reset()
    assert w.size() == 0
    assert w.push("c") == []
