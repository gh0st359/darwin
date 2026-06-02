"""Tests for the SpeechDLM adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.speech import SpeechDLM, SpeechPipeline


@dataclass
class _FakePlan:
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
class _FakeFrame:
    user_id: str = ""


@dataclass
class _FakeTrace:
    user_text: str = ""


def test_dlm_render_returns_dlm_render_result_shape() -> None:
    pipeline = SpeechPipeline()
    dlm = SpeechDLM(pipeline)
    plan = _FakePlan(thesis="The dog is a mammal.")
    result = dlm.render(plan, _FakeFrame(), _FakeTrace())
    assert hasattr(result, "text")
    assert hasattr(result, "renderer")
    assert hasattr(result, "valid")
    assert result.renderer == "speech"


def test_dlm_render_name_matches_protocol() -> None:
    pipeline = SpeechPipeline()
    dlm = SpeechDLM(pipeline)
    assert dlm.name == "speech"


def test_dlm_render_records_validation_notes_on_leak_fallback() -> None:
    pipeline = SpeechPipeline()
    dlm = SpeechDLM(pipeline)
    plan = _FakePlan(thesis='Curly leak {"x": 1} here.')
    result = dlm.render(plan, _FakeFrame(), _FakeTrace())
    assert result.validation_notes
    assert any("leak" in n.lower() for n in result.validation_notes)


def test_dlm_render_clean_input_passes_with_no_notes() -> None:
    pipeline = SpeechPipeline()
    dlm = SpeechDLM(pipeline)
    plan = _FakePlan(thesis="A bird is an animal.")
    result = dlm.render(plan, _FakeFrame(), _FakeTrace())
    assert result.valid
    assert result.validation_notes == []
