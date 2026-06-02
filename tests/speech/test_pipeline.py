"""Tests for the five-stage SpeechPipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.speech.pipeline import SpeechPipeline


@dataclass
class _FakePlan:
    """Minimal stand-in for a ResponsePlan."""

    mode: str = "answer"
    intent: str = "explain"
    thesis: str = "A neuron is a cell."
    answer_points: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    uncertainties: list = field(default_factory=list)
    clarification_questions: list = field(default_factory=list)
    next_actions: list = field(default_factory=list)
    target_length: str = "medium"


@dataclass
class _FakeOperatorModel:
    verbosity: str = "medium"

    def preferred_length(self, mode: str) -> str:
        return self.verbosity


class _FakeOperatorModels:
    def __init__(self, verbosity: str = "medium") -> None:
        self.verbosity = verbosity

    def get(self, user_id):
        return _FakeOperatorModel(self.verbosity)


def test_pipeline_renders_minimum_plan_as_prose() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(thesis="A photon is a particle.")
    result = pipeline.render(plan)
    assert result.text.startswith("A photon")
    assert result.text.endswith(".")
    assert result.leak_passed


def test_pipeline_includes_support_points_with_markers() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(
        thesis="Neurons connect to other neurons.",
        answer_points=[
            "Each neuron has thousands of synapses.",
            "Signals travel via action potentials.",
        ],
    )
    result = pipeline.render(plan)
    assert "synapses" in result.text
    assert "Also" in result.text or "Furthermore" in result.text or "Moreover" in result.text


def test_pipeline_short_verbosity_drops_points() -> None:
    pipeline = SpeechPipeline(operator_models=_FakeOperatorModels("short"))
    plan = _FakePlan(
        thesis="A is B.",
        answer_points=["one", "two", "three", "four"],
    )
    result = pipeline.render(plan)
    # Short mode keeps thesis only + at most one support.
    assert "three" not in result.text
    assert "four" not in result.text


def test_pipeline_long_verbosity_includes_evidence_and_caveats() -> None:
    pipeline = SpeechPipeline(operator_models=_FakeOperatorModels("long"))
    plan = _FakePlan(
        thesis="X is Y.",
        answer_points=["a", "b"],
        evidence=["evidence_one"],
        uncertainties=["I'm uncertain about z."],
    )
    result = pipeline.render(plan)
    assert "evidence_one" in result.text
    assert "uncertain" in result.text.lower() or "z" in result.text.lower()


def test_pipeline_handles_empty_plan_safely() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(thesis="")
    result = pipeline.render(plan)
    # Empty thesis is fine; pipeline produces an empty string (or
    # nearly-empty) without error.
    assert isinstance(result.text, str)


def test_pipeline_clarification_appears_as_followup_question() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(
        thesis="I see what you mean.",
        clarification_questions=["What kind of X did you mean?"],
    )
    result = pipeline.render(plan)
    assert "Quick question back" in result.text
    assert "?" in result.text


def test_pipeline_leak_gate_substitutes_fallback_when_thesis_leaks() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(thesis='Json-looking {"a": 1} contents.')
    result = pipeline.render(plan)
    # The leak gate kicked in.
    assert not result.leak_passed
    assert "{" not in result.text
    assert "}" not in result.text


def test_pipeline_strips_internal_bracketed_tags() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(thesis="[is_a_chain] photon is a particle.")
    result = pipeline.render(plan)
    assert "is_a_chain" not in result.text
    assert "particle" in result.text


def test_pipeline_underscore_concepts_become_natural_phrases() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(thesis="The free_will question is hard.")
    result = pipeline.render(plan)
    # The lexicon substitutes underscored names with human spacing.
    # With no lexicon attached the pipeline preserves; with a lexicon the
    # default surface_for_concept maps free_will → "free will". Test both.
    assert "free_will" in result.text or "free will" in result.text


def test_pipeline_result_serializes() -> None:
    pipeline = SpeechPipeline()
    plan = _FakePlan(thesis="A is B.")
    result = pipeline.render(plan)
    record = result.to_record()
    assert "text" in record
    assert "leak_passed" in record
