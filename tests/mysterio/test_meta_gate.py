"""Tests for the self-modifiable accept gate."""

from __future__ import annotations

from darwin.mysterio.meta_gate import (
    GateInputs,
    GateSpec,
    MetaGate,
    default_gate_fn,
    make_gate_spec,
)


def test_default_gate_accepts_strict_improvement() -> None:
    gate = MetaGate()
    decision = gate.decide(
        GateInputs(improvement=0.05, baseline_error=0.2, candidate_error=0.15)
    )
    assert decision.accepted
    assert decision.score > 0


def test_default_gate_rejects_no_improvement() -> None:
    gate = MetaGate()
    decision = gate.decide(
        GateInputs(improvement=-0.01, baseline_error=0.2, candidate_error=0.21)
    )
    assert not decision.accepted


def test_continuity_bonus_can_swing_borderline_case() -> None:
    gate = MetaGate()
    # tiny improvement plus continuity bonus should accept
    decision = gate.decide(
        GateInputs(
            improvement=0.001,
            baseline_error=0.2,
            candidate_error=0.199,
            continuity_term=1.0,
        )
    )
    assert decision.accepted


def test_swap_records_history() -> None:
    gate = MetaGate()
    original_id = gate.current.gate_id
    alt = make_gate_spec(
        default_gate_fn(lambda_continuity=0.0, lambda_visibility=0.0),
        description="strict",
    )
    record = gate.swap(alt, shadow_outcomes=[], notes="test swap")
    assert gate.current.gate_id == alt.gate_id
    assert len(gate.history) == 1
    assert record.old_gate_id == original_id
    assert record.new_gate_id == alt.gate_id


def test_shadow_agreement_is_computed() -> None:
    gate = MetaGate()
    alt = make_gate_spec(
        default_gate_fn(lambda_continuity=0.0, lambda_visibility=0.0),
        description="strict",
    )
    inputs = GateInputs(improvement=0.05, baseline_error=0.2, candidate_error=0.15)
    shadow = [(inputs, True), (inputs, True)]
    record = gate.swap(alt, shadow_outcomes=shadow)
    assert record.shadow_sample_size == 2
    assert 0.0 <= record.shadow_agreement <= 1.0


def test_default_gate_spec_metadata() -> None:
    spec = GateSpec.default()
    assert spec.gate_id == "default-v6"
    assert "improvement" in spec.description
