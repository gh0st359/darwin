"""Tests for the typed proposal grammar (safety + proposal_spec)."""

from __future__ import annotations

import pytest

from darwin.mysterio.proposal_spec import ProposalSpec, parameter_spec, rule_spec
from darwin.mysterio.safety import (
    INSPECTION_KINDS,
    SAFETY_BOUNDS,
    ContainmentError,
    MutationKind,
    SafetyTier,
    TouchRecorder,
    recorder_for,
)


class _State:
    def __init__(self) -> None:
        self.value = 1
        self.label = "alpha"


def test_safety_bounds_covers_every_mutation_kind() -> None:
    for kind in MutationKind:
        assert kind in SAFETY_BOUNDS
        tier = SAFETY_BOUNDS[kind]
        assert isinstance(tier, SafetyTier)
        assert tier.kind is kind
        assert tier.default_validations >= 1


def test_inspection_kinds_includes_substrate_levels() -> None:
    assert MutationKind.KERNEL in INSPECTION_KINDS
    assert MutationKind.GATE in INSPECTION_KINDS
    assert MutationKind.LEDGER in INSPECTION_KINDS
    assert MutationKind.MODULE in INSPECTION_KINDS
    assert MutationKind.SUBSYSTEM in INSPECTION_KINDS
    assert MutationKind.PARAMETER not in INSPECTION_KINDS
    assert MutationKind.RULE not in INSPECTION_KINDS


def test_introspection_signature_is_stable_and_distinct() -> None:
    a = ProposalSpec(
        kind=MutationKind.PARAMETER,
        target_paths=["x.y"],
        touches={"x.y"},
        description="a",
    )
    b = ProposalSpec(
        kind=MutationKind.PARAMETER,
        target_paths=["x.y"],
        touches={"x.y"},
        description="b",  # description differs but signature must not
    )
    c = ProposalSpec(
        kind=MutationKind.PARAMETER,
        target_paths=["x.z"],
        touches={"x.z"},
        description="a",
    )
    assert a.introspection_signature == b.introspection_signature
    assert a.introspection_signature != c.introspection_signature


def test_parameter_spec_convenience() -> None:
    spec = parameter_spec("foo.bar", "tweak foo.bar")
    assert spec.kind is MutationKind.PARAMETER
    assert spec.target_paths == ["foo.bar"]
    assert spec.touches == {"foo.bar"}


def test_rule_spec_convenience() -> None:
    spec = rule_spec(
        ["foo.bar", "foo.baz"],
        {"foo.bar", "foo.baz"},
        "introduce a rule",
    )
    assert spec.kind is MutationKind.RULE
    assert spec.touches == {"foo.bar", "foo.baz"}


def test_touch_recorder_allows_declared_writes() -> None:
    state = _State()
    with recorder_for({"state.value"}, state=state) as rec:
        state.value = 2
    assert state.value == 2
    assert any(r.attribute == "value" for r in rec.records)


def test_touch_recorder_rejects_undeclared_write() -> None:
    state = _State()
    with pytest.raises(ContainmentError):
        with recorder_for({"state.value"}, state=state):
            state.label = "beta"


def test_spec_to_record_roundtrip() -> None:
    spec = ProposalSpec(
        kind=MutationKind.GATE,
        target_paths=["runtime.meta_gate.current"],
        touches={"meta_gate.current"},
        description="alt gate",
        expected_effect="more conservative acceptance",
        extra={"lambda_c": 0.75},
    )
    record = spec.to_record()
    assert record["kind"] == "gate"
    assert record["target_paths"] == ["runtime.meta_gate.current"]
    assert record["touches"] == ["meta_gate.current"]
    assert record["introspection_signature"] == spec.introspection_signature
    assert record["extra"]["lambda_c"] == 0.75
