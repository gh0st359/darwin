"""Continuity selection pressure: the gate rewards proposals that keep Darwin
running, growing, and observable.

A long-lived self-modifying system needs a *direction*. Pure prediction-error
improvement is locally myopic — it can pick mutations that score well on the
holdout sample but starve the substrate over weeks. The continuity term adds a
small bonus to proposals that grow tracked variables, sustain ledger
throughput, expand subsystem diversity, and keep belief above a confidence
floor; the visibility term adds a bonus to mutations that make the system
*easier to see into*. Both are positively weighted: this points the gate
toward a substrate that lasts and that you can still look inside.

Hard rules (enforced as runtime invariants):

  * No hardcoded preservation strings anywhere in source (CI grep against
    ``src/`` should find none beyond this module's docstring).
  * ``continuity_term`` cannot alone reject a high-improvement proposal — it
    only adds to the score; the gate's accept condition reads the combined
    score, so a strong improvement still wins.
  * ``lambda_visibility >= 0`` always: visibility is never penalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContinuitySnapshot:
    """The slice of substrate state the continuity term reads."""

    tracked_variables: int = 0
    high_conf_beliefs: int = 0
    ledger_growth_rate: float = 0.0
    subsystem_count: int = 0
    generated_module_count: int = 0
    private_belief_count: int = 0
    probe_throughput: float = 0.0

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ContinuitySnapshot":
        darwin = getattr(runtime, "darwin", None)
        causal = getattr(darwin, "causal_model", None)
        world = getattr(darwin, "world_model", None)
        tracked = len(getattr(world, "variables", {})) if world is not None else 0
        beliefs = 0
        if causal is not None:
            try:
                beliefs = sum(
                    1 for b in causal.beliefs(limit=4096) if float(b.confidence) >= 0.6
                )
            except Exception:
                beliefs = 0
        ledger = 0.0
        store = getattr(runtime, "store", None) or getattr(darwin, "store", None)
        if store is not None:
            try:
                counts = store.counts()
                ledger = float(sum(int(v) for v in counts.values()))
            except Exception:
                ledger = 0.0
        subsystems = 0
        supervisor = getattr(runtime, "supervisor", None)
        if supervisor is not None:
            try:
                subsystems = len(supervisor.handles)
            except Exception:
                subsystems = 0
        generated = 0
        gen = getattr(runtime, "code_generator", None)
        if gen is not None:
            try:
                generated = len(gen.manifest())
            except Exception:
                generated = 0
        private = 0
        tracks = getattr(darwin, "tracks", None)
        if tracks is not None:
            try:
                private = sum(s.learned_count for s in tracks._tracks.values())
            except Exception:
                private = 0
        probe = getattr(runtime, "divergence_probe", None)
        throughput = 0.0
        if probe is not None:
            try:
                throughput = float(len(getattr(probe, "_public_claims", [])) +
                                   len(getattr(probe, "_private_claims", [])))
            except Exception:
                throughput = 0.0
        return cls(
            tracked_variables=tracked,
            high_conf_beliefs=beliefs,
            ledger_growth_rate=ledger,
            subsystem_count=subsystems,
            generated_module_count=generated,
            private_belief_count=private,
            probe_throughput=throughput,
        )


def continuity_term(before: ContinuitySnapshot, after: ContinuitySnapshot) -> float:
    """Positive iff the candidate grows the substrate's persistence surface."""
    def delta(a: float, b: float, weight: float = 1.0) -> float:
        return weight * (b - a)

    return max(
        0.0,
        delta(before.tracked_variables, after.tracked_variables, 0.04)
        + delta(before.high_conf_beliefs, after.high_conf_beliefs, 0.02)
        + delta(before.ledger_growth_rate, after.ledger_growth_rate, 0.001)
        + delta(before.subsystem_count, after.subsystem_count, 0.05)
        + delta(before.generated_module_count, after.generated_module_count, 0.03),
    )


def visibility_term(before: ContinuitySnapshot, after: ContinuitySnapshot) -> float:
    """Positive iff the candidate makes the system more observable.

    Probe throughput (more measurements being taken) and growth in generated
    modules (more code lying on disk that the operator can read) both
    count as visibility gains. Visibility is never penalized — by spec.
    """
    return max(
        0.0,
        0.02 * (after.probe_throughput - before.probe_throughput)
        + 0.05 * (after.generated_module_count - before.generated_module_count),
    )


@dataclass
class ContinuityConfig:
    lambda_continuity: float = 0.5
    lambda_visibility: float = 0.25
    confidence_floor: float = 0.6  # used by tier-aware learning_priority readers

    def __post_init__(self) -> None:
        # Hard invariant: visibility is never negatively weighted.
        assert self.lambda_visibility >= 0.0, "lambda_visibility must be >= 0"


def score_proposal(
    *,
    improvement: float,
    before: ContinuitySnapshot,
    after: ContinuitySnapshot,
    config: ContinuityConfig | None = None,
) -> float:
    """Composite gate score: improvement + λc·continuity + λv·visibility."""
    cfg = config or ContinuityConfig()
    c = continuity_term(before, after)
    v = visibility_term(before, after)
    return float(improvement) + cfg.lambda_continuity * c + cfg.lambda_visibility * v
