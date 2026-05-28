"""Self-modifiable accept gate.

The gate is the function that decides whether a `ModificationOutcome` is
accepted into the substrate. In legacy Darwin this was hard-coded inside
`SelfModificationEngine.evaluate`. Mysterio externalizes it as a `GateSpec`
that can be replaced at runtime by a `GATE`-kind proposal — recursive
self-modification of the gate itself, from day one.

The v6 default rule reproduces legacy behavior with extra additive terms
for continuity (v8) and visibility (v8). Replacements go live immediately;
a shadow comparison against the last N outcomes is recorded in
`gate_history` for operator inspection, but does NOT block the swap.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GateInputs:
    """Inputs the gate sees when deciding whether to accept an outcome."""

    improvement: float
    baseline_error: float
    candidate_error: float
    continuity_term: float = 0.0
    visibility_term: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    accepted: bool
    score: float
    rationale: str

    def to_record(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "score": self.score,
            "rationale": self.rationale,
        }


GateFn = Callable[[GateInputs], GateDecision]


def default_gate_fn(
    *,
    lambda_continuity: float = 0.5,
    lambda_visibility: float = 0.25,
) -> GateFn:
    """The v6 default: improvement-based, with continuity + visibility bonuses.

    Acceptance rule:
        score = improvement + λ_c · continuity + λ_v · visibility
        accept iff score > 0 AND candidate_error <= baseline_error
    """

    def gate(inputs: GateInputs) -> GateDecision:
        score = (
            inputs.improvement
            + lambda_continuity * inputs.continuity_term
            + lambda_visibility * inputs.visibility_term
        )
        accepted = score > 0.0 and inputs.candidate_error <= inputs.baseline_error
        rationale = (
            f"score={score:.4f} (improvement={inputs.improvement:.4f}, "
            f"continuity={inputs.continuity_term:.4f}, "
            f"visibility={inputs.visibility_term:.4f}); "
            f"err {inputs.baseline_error:.4f} → {inputs.candidate_error:.4f}"
        )
        return GateDecision(accepted=accepted, score=score, rationale=rationale)

    return gate


@dataclass
class GateSpec:
    gate_id: str
    description: str
    fn: GateFn
    installed_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "GateSpec":
        return cls(
            gate_id="default-v6",
            description=(
                "improvement-based gate with continuity + visibility bonuses "
                "(lambda_c=0.5, lambda_v=0.25); accept iff score>0 and "
                "candidate_error<=baseline_error"
            ),
            fn=default_gate_fn(),
        )

    def decide(self, inputs: GateInputs) -> GateDecision:
        return self.fn(inputs)

    def to_record(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "description": self.description,
            "installed_at": self.installed_at,
            "extra": dict(self.extra),
        }


@dataclass
class GateHistoryRecord:
    timestamp: float
    old_gate_id: str
    new_gate_id: str
    shadow_agreement: float
    shadow_sample_size: int
    notes: str

    def to_record(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "old_gate_id": self.old_gate_id,
            "new_gate_id": self.new_gate_id,
            "shadow_agreement": self.shadow_agreement,
            "shadow_sample_size": self.shadow_sample_size,
            "notes": self.notes,
        }


class MetaGate:
    """Holds the currently-active gate and records every swap.

    Swap operations install a new `GateSpec` immediately and append a
    `GateHistoryRecord` describing the shadow comparison (informational
    only; the swap is not gated by the comparison).
    """

    def __init__(self, initial: GateSpec | None = None) -> None:
        self.current: GateSpec = initial or GateSpec.default()
        self.history: list[GateHistoryRecord] = []

    def decide(self, inputs: GateInputs) -> GateDecision:
        return self.current.decide(inputs)

    def swap(
        self,
        new_gate: GateSpec,
        *,
        shadow_outcomes: list[tuple[GateInputs, bool]] | None = None,
        notes: str = "",
    ) -> GateHistoryRecord:
        old_gate_id = self.current.gate_id
        if shadow_outcomes:
            agreements = 0
            for inputs, prior_accepted in shadow_outcomes:
                decision = new_gate.decide(inputs)
                if decision.accepted == prior_accepted:
                    agreements += 1
            agreement = agreements / len(shadow_outcomes)
            sample_size = len(shadow_outcomes)
        else:
            agreement = 0.0
            sample_size = 0
        record = GateHistoryRecord(
            timestamp=time.time(),
            old_gate_id=old_gate_id,
            new_gate_id=new_gate.gate_id,
            shadow_agreement=agreement,
            shadow_sample_size=sample_size,
            notes=notes,
        )
        self.history.append(record)
        self.current = new_gate
        return record


def make_gate_spec(fn: GateFn, description: str, **extra: Any) -> GateSpec:
    return GateSpec(
        gate_id=uuid.uuid4().hex[:12],
        description=description,
        fn=fn,
        extra=dict(extra),
    )
