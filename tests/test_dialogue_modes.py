"""Tests for the v6.5 ResponsePlan modes added to DiscoursePlanner."""

from __future__ import annotations

from darwin.discourse import DiscoursePlanner
from darwin.semantics import SemanticFrame


def _frame(text: str = "tell me more") -> SemanticFrame:
    return SemanticFrame(
        source="user",
        original_text=text,
        normalized_text=text.lower(),
        tokens=text.lower().split(),
        speech_act="question",
        topic="general",
        intent="inquire",
        confidence=0.8,
        uncertainty=0.2,
    )


def test_dialogue_modes_constant_exposes_new_modes() -> None:
    expected = {
        "probe",
        "clarify_request",
        "offer_alternative",
        "revisit_prior_thread",
        "concede_uncertainty",
    }
    assert expected <= DiscoursePlanner.DIALOGUE_MODES


def test_concede_uncertainty_plan_yields_honest_non_answer() -> None:
    planner = DiscoursePlanner()
    plan = planner.concede_uncertainty_plan(
        _frame(), reason="the question targets an area I haven't observed"
    )
    assert plan.mode == "concede_uncertainty"
    assert plan.confidence < 0.5
    assert plan.target_length == "short"
    assert plan.uncertainties


def test_probe_plan_carries_clarification_question() -> None:
    planner = DiscoursePlanner()
    plan = planner.probe_plan(
        _frame(),
        question="Which sensor are you referring to?",
        because="multiple sensors share that name",
    )
    assert plan.mode == "probe"
    assert "Which sensor are you referring to?" in plan.clarification_questions
    assert plan.uncertainties == ["multiple sensors share that name"]


def test_offer_alternative_plan_states_thesis_as_alternative() -> None:
    planner = DiscoursePlanner()
    plan = planner.offer_alternative_plan(
        _frame(),
        alternative="we could instead inspect the divergence report directly",
        reason="that would skirt the need to confabulate a confidence",
    )
    assert plan.mode == "offer_alternative"
    assert "divergence report" in plan.thesis
    assert plan.confidence >= 0.5


def test_revisit_prior_thread_plan_summarizes_thread() -> None:
    planner = DiscoursePlanner()
    plan = planner.revisit_prior_thread_plan(
        _frame(),
        thread_summary="earlier we were investigating why the fuse keeps blowing",
    )
    assert plan.mode == "revisit_prior_thread"
    assert "fuse keeps blowing" in plan.thesis
    assert plan.target_length == "medium"


def test_dialogue_mode_plans_serialize_via_to_record() -> None:
    planner = DiscoursePlanner()
    plans = [
        planner.concede_uncertainty_plan(_frame(), reason="no data"),
        planner.probe_plan(_frame(), question="which one?"),
        planner.offer_alternative_plan(_frame(), alternative="try Y instead"),
        planner.revisit_prior_thread_plan(
            _frame(), thread_summary="continuing the earlier thread"
        ),
    ]
    for plan in plans:
        record = plan.to_record()
        assert "mode" in record
        assert record["mode"] == plan.mode
