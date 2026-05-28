"""Typed envelope describing what a self-modification proposal does.

A `ProposalSpec` is attached to a `ProposedModification` and read by
`SelfModificationEngine.evaluate` to (1) drive the `TouchRecorder` for
containment, (2) classify the proposal under `SAFETY_BOUNDS` for operator
review, (3) provide a stable hash the meta-proposer uses to dedupe and
recognize the same structural proposal across runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from darwin.mysterio.safety import SAFETY_BOUNDS, MutationKind


@dataclass
class ProposalSpec:
    kind: MutationKind
    target_paths: list[str]
    touches: set[str]
    description: str
    expected_effect: str = ""
    reversible: bool = True
    generated_code: str | None = None
    target_module_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MutationKind):
            self.kind = MutationKind(self.kind)
        self.target_paths = list(self.target_paths)
        self.touches = set(self.touches)

    @property
    def introspection_signature(self) -> str:
        payload = "|".join(
            [
                self.kind.value,
                ";".join(sorted(self.target_paths)),
                ";".join(sorted(self.touches)),
                self.target_module_path or "",
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @property
    def tier_notes(self) -> str:
        tier = SAFETY_BOUNDS.get(self.kind)
        return tier.notes if tier else ""

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_paths": list(self.target_paths),
            "touches": sorted(self.touches),
            "description": self.description,
            "expected_effect": self.expected_effect,
            "reversible": self.reversible,
            "introspection_signature": self.introspection_signature,
            "target_module_path": self.target_module_path,
            "has_generated_code": self.generated_code is not None,
            "extra": dict(self.extra),
        }


def parameter_spec(target_path: str, description: str, **kwargs: Any) -> ProposalSpec:
    """Convenience constructor for the common scalar-tweak case."""
    return ProposalSpec(
        kind=MutationKind.PARAMETER,
        target_paths=[target_path],
        touches={target_path},
        description=description,
        **kwargs,
    )


def rule_spec(
    target_paths: list[str],
    touches: set[str],
    description: str,
    **kwargs: Any,
) -> ProposalSpec:
    return ProposalSpec(
        kind=MutationKind.RULE,
        target_paths=list(target_paths),
        touches=set(touches),
        description=description,
        **kwargs,
    )
