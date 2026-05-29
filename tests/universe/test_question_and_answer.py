"""Tests for question understanding + answer rendering."""

from __future__ import annotations

from dataclasses import dataclass

from darwin.universe.answer import (
    RenderedAnswer,
    build_answer,
    render_chain,
    render_contradiction,
    render_definition,
    render_inference,
)
from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.inference import Contradiction, Inference, InferenceEngine
from darwin.universe.question import analyze_question


# -- question analysis -------------------------------------------------------


def test_definition_question_detected() -> None:
    a = analyze_question("What is consciousness?", ["consciousness"])
    assert a.kind == "definition"
    assert a.primary_concepts == ["consciousness"]
    assert a.is_question


def test_kind_check_detected() -> None:
    a = analyze_question("Is a photon a particle?", ["photon", "particle"])
    assert a.kind in ("kind_check", "definition")  # both surface kinds match
    assert a.is_question


def test_causal_why_detected() -> None:
    a = analyze_question("Why does gravity cause acceleration?", ["gravity", "acceleration"])
    assert a.kind == "causal_why"


def test_relation_question_detected() -> None:
    a = analyze_question("How does music relate to math?", ["music", "math"])
    assert a.kind in ("relation", "causal_how")


def test_compare_question_detected() -> None:
    a = analyze_question("Compare cause and effect.", ["cause", "effect"])
    assert a.kind == "compare"


def test_opinion_question_detected() -> None:
    a = analyze_question("What do you think about free will?", ["free_will"])
    assert a.kind == "opinion"


def test_greeting_detected() -> None:
    a = analyze_question("Hello there", [])
    assert a.kind == "greeting"
    assert not a.is_question


def test_unknown_when_no_cues() -> None:
    a = analyze_question("just talking here", ["talking"])
    assert a.kind in ("unknown", "definition")


# -- chain rendering ---------------------------------------------------------


def test_render_chain_one_step() -> None:
    chain = [{"source": "dog", "target": "mammal", "kind": "is_a"}]
    s = render_chain(chain)
    assert "dog is a mammal" in s


def test_render_chain_multi_step() -> None:
    chain = [
        {"source": "dog", "target": "mammal", "kind": "is_a"},
        {"source": "mammal", "target": "animal", "kind": "is_a"},
    ]
    s = render_chain(chain)
    assert "dog is a mammal" in s
    assert "which is a animal" in s


# -- inference rendering ----------------------------------------------------


def test_render_is_a_chain_one_hop_is_direct() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    inf = InferenceEngine(u).is_a_chain("dog", "mammal")
    text = render_inference(inf)
    assert text.startswith("Yes")
    assert "dog is a mammal" in text


def test_render_causal_chain_includes_chain_when_multi_hop() -> None:
    u = ConceptUniverse()
    u.add_relation("rain", "wetness", "causes", ensure_concepts=True)
    u.add_relation("wetness", "slipperiness", "causes", ensure_concepts=True)
    u.add_relation("slipperiness", "falls", "causes", ensure_concepts=True)
    inf = InferenceEngine(u).causal_chain("rain", "falls")
    text = render_inference(inf)
    assert "rain" in text and "falls" in text
    assert "3 step" in text


def test_render_contradiction_states_reason() -> None:
    c = Contradiction(a="hot", b="cold", reason="explicit opposition edge")
    text = render_contradiction(c)
    assert "contradiction" in text.lower()
    assert "hot" in text and "cold" in text


def test_render_definition_includes_domain_and_definition() -> None:
    u = ConceptUniverse()
    concept = u.add_concept("x", domain="alpha", definition="a sample thing")
    text = render_definition(concept)
    assert "alpha" in text
    assert "sample thing" in text


# -- build_answer composes prose -------------------------------------------


def test_build_answer_uses_inferences_for_kind_questions() -> None:
    u = ConceptUniverse()
    u.add_relation("dog", "mammal", "is_a", ensure_concepts=True)
    inf = InferenceEngine(u).is_a_chain("dog", "mammal")
    answer = build_answer(
        question_kind="kind_check",
        grounded_concepts=["dog", "mammal"],
        inferences=[inf],
    )
    assert "dog is a mammal" in answer.text
    assert "is_a_chain" in answer.used_inferences


def test_build_answer_surfaces_contradictions_first() -> None:
    c = Contradiction(a="hot", b="cold", reason="opposes")
    answer = build_answer(
        question_kind="contradiction",
        grounded_concepts=["hot", "cold"],
        contradictions=[c],
        inferences=[],
    )
    assert answer.text.startswith("I see a contradiction")
    assert "contradiction" in answer.used_inferences


def test_build_answer_falls_back_to_curiosity_when_nothing_derivable() -> None:
    answer = build_answer(
        question_kind="definition",
        grounded_concepts=["whorzplatz"],
        inferences=[],
        curiosity_questions=["How does whorzplatz relate to anything?"],
    )
    assert "whorzplatz" in answer.text or "don't have a confident" in answer.text
    assert answer.style == "concede_uncertainty"


def test_build_answer_final_fallback_when_no_signal() -> None:
    answer = build_answer(
        question_kind="unknown",
        grounded_concepts=[],
        inferences=[],
    )
    assert answer.text
    assert answer.style == "concede_uncertainty"


def test_rendered_answer_serializes() -> None:
    answer = RenderedAnswer(
        text="hi", style="neutral", points=["hi"], grounded_concepts=["a"],
    )
    record = answer.to_record()
    assert record["text"] == "hi"
    assert record["grounded_concepts"] == ["a"]
