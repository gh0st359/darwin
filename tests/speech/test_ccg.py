"""Tests for the CCG combinator substrate."""

from __future__ import annotations

from darwin.speech.ccg import (
    CCGCategory,
    CCGSign,
    N,
    NP,
    S,
    backward_apply,
    combine,
    forward_apply,
    forward_compose,
    parse_category,
)


def test_atomic_category_string() -> None:
    assert str(N) == "N"
    assert str(NP) == "NP"


def test_functional_forward_string() -> None:
    cat = CCGCategory.forward(NP, N)
    assert str(cat) == "NP/N"


def test_functional_backward_string() -> None:
    cat = CCGCategory.backward(S, NP)
    assert str(cat) == "S\\NP"


def test_parse_atomic_round_trip() -> None:
    assert str(parse_category("N")) == "N"
    assert str(parse_category("NP")) == "NP"


def test_parse_functional_round_trip() -> None:
    assert str(parse_category("NP/N")) == "NP/N"
    assert str(parse_category("S\\NP")) == "S\\NP"


def test_parse_nested_round_trip() -> None:
    # S\NP/NP — copula needing an NP on each side.
    cat = parse_category("S\\NP/NP")
    assert str(cat) == "S\\NP/NP"


def test_forward_apply_combines_determiner_and_noun() -> None:
    the = CCGSign(category=CCGCategory.forward(NP, N), surface="the")
    cat = CCGSign(category=N, surface="cat")
    out = forward_apply(the, cat)
    assert out is not None
    assert out.surface == "the cat"
    assert str(out.category) == "NP"


def test_forward_apply_mismatch_returns_none() -> None:
    the = CCGSign(category=CCGCategory.forward(NP, N), surface="the")
    sentence = CCGSign(category=S, surface="it rains")
    assert forward_apply(the, sentence) is None


def test_backward_apply_combines_subject_and_predicate() -> None:
    subj = CCGSign(category=NP, surface="the cat")
    pred = CCGSign(
        category=CCGCategory.backward(S, NP),
        surface="sleeps",
    )
    out = backward_apply(subj, pred)
    assert out is not None
    assert out.surface == "the cat sleeps"
    assert str(out.category) == "S"


def test_forward_compose_chains_two_slashes() -> None:
    a = CCGSign(category=CCGCategory.forward(S, NP), surface="loves")
    b = CCGSign(category=CCGCategory.forward(NP, N), surface="the")
    out = forward_compose(a, b)
    assert out is not None
    assert str(out.category) == "S/N"


def test_combine_tries_combinators_in_order() -> None:
    the = CCGSign(category=CCGCategory.forward(NP, N), surface="the")
    cat = CCGSign(category=N, surface="cat")
    out = combine(the, cat)
    assert out is not None
    assert out.surface == "the cat"
