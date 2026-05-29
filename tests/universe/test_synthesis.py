"""Tests for AnswerSynthesizer + self-introspection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _FakeInference:
    operator: str = "is_a_chain"
    claim: str = "x is a y"
    confidence: float = 0.9
    chain: list = field(default_factory=lambda: [
        {"source": "x", "target": "y", "kind": "is_a"}
    ])


def test_synthesize_combines_multiple_inferences_with_discourse_markers() -> None:
    from darwin.universe.synthesis import synthesize

    inferences = [
        _FakeInference(operator="is_a_chain", claim="a is a b"),
        _FakeInference(operator="causal_chain", claim="b causes c",
                       chain=[{"source": "b", "target": "c", "kind": "causes"}]),
        _FakeInference(operator="shortest_path", claim="a is connected to d",
                       chain=[{"source": "a", "target": "d", "kind": "related_to"}]),
    ]
    answer = synthesize(
        question_kind="relation",
        grounded_concepts=["a", "b", "c", "d"],
        inferences=inferences,
    )
    assert answer.style == "synthesis"
    assert "Also" in answer.text or "In addition" in answer.text or "Moreover" in answer.text
    assert len(answer.sentences) == 3


def test_synthesize_priority_starts_with_is_a_chain() -> None:
    from darwin.universe.synthesis import synthesize

    inferences = [
        _FakeInference(operator="shortest_path", claim="a is connected to b"),
        _FakeInference(operator="is_a_chain", claim="a is a b"),
    ]
    answer = synthesize(
        question_kind="kind_check",
        grounded_concepts=["a", "b"],
        inferences=inferences,
    )
    # is_a_chain should lead.
    assert answer.sentences[0].lower().startswith(("yes", "a is"))


def test_synthesize_includes_contradictions_first() -> None:
    from darwin.universe.synthesis import synthesize

    @dataclass
    class _C:
        a: str = "hot"
        b: str = "cold"
        reason: str = "opposes"
        chain: list = field(default_factory=list)

    answer = synthesize(
        question_kind="contradiction",
        grounded_concepts=["hot", "cold"],
        contradictions=[_C()],
        inferences=[_FakeInference()],
    )
    assert "contradiction" in answer.text.lower()


def test_synthesize_empty_inputs_returns_empty_text() -> None:
    from darwin.universe.synthesis import synthesize

    answer = synthesize(
        question_kind="unknown",
        grounded_concepts=[],
        inferences=[],
        contradictions=[],
    )
    # No inferences = no synthesized body.
    assert answer.text == ""


def test_self_introspection_includes_universe_stats() -> None:
    from darwin.universe.synthesis import synthesize_self_introspection

    answer = synthesize_self_introspection(
        grounded_concepts=["self", "model"],
        universe_summary={"concepts": 42, "relations": 80, "domains": 5},
        inferences_count=3,
    )
    assert "42 concept" in answer.text
    assert "80 relation" in answer.text
    assert "5 domain" in answer.text


def test_self_introspection_honest_when_no_inferences() -> None:
    from darwin.universe.synthesis import synthesize_self_introspection

    answer = synthesize_self_introspection(
        grounded_concepts=["something_obscure"],
        universe_summary={"concepts": 30, "relations": 50, "domains": 3},
        inferences_count=0,
    )
    assert "don't have a strong" in answer.text.lower() or "thin" in answer.text.lower()


def test_synthesized_answer_serializes() -> None:
    from darwin.universe.synthesis import SynthesizedAnswer

    a = SynthesizedAnswer(text="hello", sentences=["hello"], confidence=0.6)
    record = a.to_record()
    assert record["text"] == "hello"
    assert record["confidence"] == 0.6
