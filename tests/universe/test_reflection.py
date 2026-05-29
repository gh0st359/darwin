"""Tests for ReflectiveDialogue."""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.universe.reflection import (
    Reflection,
    is_reflective_prompt,
    reflect_on_last_reply,
)


def test_is_reflective_prompt_matches_why_did_you() -> None:
    assert is_reflective_prompt("Why did you say that?")
    assert is_reflective_prompt("How did you arrive at that earlier?")
    assert is_reflective_prompt("Explain your last reply.")


def test_is_reflective_prompt_matches_self_thinking() -> None:
    assert is_reflective_prompt("What are you thinking about?")
    assert is_reflective_prompt("What's on your mind?")


def test_is_reflective_prompt_skips_neutral_text() -> None:
    assert not is_reflective_prompt("Tell me about gravity.")
    assert not is_reflective_prompt("Hello.")


@dataclass
class _FakeInference:
    operator: str = "is_a_chain"
    claim: str = "dog is a animal"
    confidence: float = 0.9
    chain: list = field(default_factory=lambda: [
        {"source": "dog", "kind": "is_a", "target": "mammal"},
        {"source": "mammal", "kind": "is_a", "target": "animal"},
    ])


@dataclass
class _FakeTurn:
    user_text: str = "Is a dog an animal?"
    darwin_text: str = "Yes, in my universe dog is a animal."
    grounded_concepts: list = field(default_factory=lambda: ["dog", "animal"])
    inferences_used: list = field(default_factory=lambda: ["is_a_chain"])
    question_kind: str = "kind_check"


def test_reflect_walks_back_through_inference_chain() -> None:
    r = reflect_on_last_reply(
        user_text="Why did you say that?",
        last_turn=_FakeTurn(),
        last_inferences=[_FakeInference()],
        last_rendered_answer=None,
        last_synthesis=None,
    )
    assert r.kind == "why_last_reply"
    assert "dog" in r.text
    assert "mammal" in r.text or "animal" in r.text
    assert r.chain_walked


def test_reflect_handles_no_prior_turn() -> None:
    r = reflect_on_last_reply(
        user_text="Why did you say that?",
        last_turn=None,
        last_inferences=[],
        last_rendered_answer=None,
        last_synthesis=None,
    )
    assert r.kind == "no_match"


def test_self_thinking_returns_dialogue_anchored_answer() -> None:
    r = reflect_on_last_reply(
        user_text="What are you thinking about?",
        last_turn=None,
        last_inferences=[],
        last_rendered_answer=None,
        last_synthesis=None,
        dialogue_summary={"most_discussed": ["physics", "music", "math"]},
    )
    assert r.kind == "self_thinking"
    assert "physics" in r.text or "music" in r.text or "math" in r.text


def test_self_thinking_surfaces_top_hypothesis() -> None:
    class _H:
        confidence = 0.9

        def as_question(self) -> str:
            return "Is X a kind of Y?"

    r = reflect_on_last_reply(
        user_text="What's on your mind?",
        last_turn=None,
        last_inferences=[],
        last_rendered_answer=None,
        last_synthesis=None,
        dialogue_summary={"most_discussed": []},
        last_hypotheses=[_H()],
    )
    assert "Is X a kind of Y" in r.text


def test_reflection_serializes() -> None:
    r = Reflection(text="t", kind="why_last_reply", chain_walked=["a"])
    record = r.to_record()
    assert record["text"] == "t"
    assert record["chain_walked"] == ["a"]
