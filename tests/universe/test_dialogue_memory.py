"""Tests for DialogueMemory."""

from __future__ import annotations

from darwin.universe.dialogue_memory import DialogueMemory


def test_record_appends_turn() -> None:
    m = DialogueMemory()
    m.record(user_text="hi", darwin_text="hello", grounded_concepts=["self"])
    assert len(m) == 1


def test_last_mention_returns_most_recent_turn_for_concept() -> None:
    m = DialogueMemory()
    m.record(user_text="a", darwin_text="a", grounded_concepts=["x"])
    m.record(user_text="b", darwin_text="b", grounded_concepts=["y"])
    m.record(user_text="c", darwin_text="c", grounded_concepts=["x", "z"])
    turn = m.last_mention("x")
    assert turn is not None
    assert turn.user_text == "c"


def test_thread_for_returns_every_turn_with_concept() -> None:
    m = DialogueMemory()
    m.record(user_text="1", darwin_text="1", grounded_concepts=["alpha"])
    m.record(user_text="2", darwin_text="2", grounded_concepts=["beta"])
    m.record(user_text="3", darwin_text="3", grounded_concepts=["alpha", "gamma"])
    thread = m.thread_for("alpha")
    assert len(thread) == 2
    texts = [t.user_text for t in thread]
    assert texts == ["1", "3"]


def test_recent_concepts_counts_across_window() -> None:
    m = DialogueMemory()
    m.record(user_text="1", darwin_text="1", grounded_concepts=["a", "b"])
    m.record(user_text="2", darwin_text="2", grounded_concepts=["a", "c"])
    m.record(user_text="3", darwin_text="3", grounded_concepts=["a"])
    rc = m.recent_concepts(3)
    assert rc[0] == "a"  # most frequent


def test_capacity_evicts_oldest() -> None:
    m = DialogueMemory(capacity=3)
    for i in range(5):
        m.record(user_text=f"u{i}", darwin_text=f"d{i}", grounded_concepts=[f"c{i}"])
    assert len(m) == 3
    earliest = m.latest(3)[0]
    assert earliest.user_text == "u2"


def test_evicted_turns_clear_concept_index() -> None:
    m = DialogueMemory(capacity=2)
    m.record(user_text="1", darwin_text="1", grounded_concepts=["alpha"])
    m.record(user_text="2", darwin_text="2", grounded_concepts=["beta"])
    m.record(user_text="3", darwin_text="3", grounded_concepts=["gamma"])
    assert m.last_mention("alpha") is None
    assert m.last_mention("beta") is not None
    assert m.last_mention("gamma") is not None


def test_contradicts_prior_requires_overlap_and_contradicting_kind() -> None:
    m = DialogueMemory()
    m.record(
        user_text="prior",
        darwin_text="X is a Y",
        grounded_concepts=["x", "y"],
        inferences_used=["is_a_chain"],
    )
    # Same concepts, but the current turn used a contradiction operator.
    prior = m.contradicts_prior(
        claim_concepts=["x", "y"],
        inferences_used=["contradiction"],
    )
    assert prior is not None
    assert prior.user_text == "prior"


def test_contradicts_prior_returns_none_when_no_overlap() -> None:
    m = DialogueMemory()
    m.record(
        user_text="prior",
        darwin_text="r",
        grounded_concepts=["alpha"],
        inferences_used=["is_a_chain"],
    )
    prior = m.contradicts_prior(
        claim_concepts=["bravo"],
        inferences_used=["contradiction"],
    )
    assert prior is None


def test_summary_reports_state() -> None:
    m = DialogueMemory()
    m.record(
        user_text="u",
        darwin_text="d",
        grounded_concepts=["a", "b"],
        question_kind="definition",
    )
    summary = m.summary()
    assert summary["turns"] == 1
    assert summary["tracked_concepts"] == 2
    assert summary["question_kinds"]["definition"] == 1
