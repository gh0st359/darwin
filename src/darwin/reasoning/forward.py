"""ForwardChainer — apply inference rules until fixpoint.

The chainer walks the ConceptUniverse graph each cycle and applies
transitive-closure rules for is_a / part_of / instantiates, causal
chains, and a small set of structural inferences. Every derived
relation is added to the universe with provenance noted; every accepted
derivation can optionally be recorded as a MutationLedger entry so it
can be rolled back if downstream reasoning reveals it as faulty.

The chainer is *bounded*: ``max_cycles`` and ``max_derivations`` cap
each invocation. Without bounds, transitive closure over a large
universe explodes.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


_TRANSITIVE_KINDS = ("is_a", "part_of", "instantiates")
_CAUSAL_KINDS = ("causes", "derives_from", "expresses")


@dataclass
class DerivedFact:
    """One fact derived by the forward chainer."""

    source: str
    target: str
    kind: str
    derived_via: list[str]
    confidence: float = 0.7

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "derived_via": list(self.derived_via),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ForwardChainReport:
    """Summary of one fixpoint_step pass."""

    cycles_taken: int = 0
    derivations_added: int = 0
    duration_ms: float = 0.0
    derived_facts: list[DerivedFact] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "cycles_taken": self.cycles_taken,
            "derivations_added": self.derivations_added,
            "duration_ms": round(self.duration_ms, 2),
            "derived_count": len(self.derived_facts),
        }


class ForwardChainer:
    """Apply transitive + causal closure rules to the universe."""

    def __init__(
        self,
        universe: Any,
        *,
        max_cycles: int = 8,
        max_derivations_per_step: int = 256,
    ) -> None:
        self.universe = universe
        self.max_cycles = int(max_cycles)
        self.max_derivations_per_step = int(max_derivations_per_step)
        self.total_derivations = 0

    def fixpoint_step(self, *, budget: int | None = None) -> ForwardChainReport:
        """Run until no new derivations OR budget exhausted."""

        started = time.perf_counter()
        report = ForwardChainReport()
        cap = budget if budget is not None else self.max_derivations_per_step
        for cycle in range(self.max_cycles):
            report.cycles_taken += 1
            added_this_cycle = 0
            for kind in _TRANSITIVE_KINDS:
                added_this_cycle += self._closure_pass(kind, report, cap)
                if report.derivations_added >= cap:
                    break
            if report.derivations_added < cap:
                for kind in _CAUSAL_KINDS:
                    added_this_cycle += self._closure_pass(kind, report, cap)
                    if report.derivations_added >= cap:
                        break
            if added_this_cycle == 0 or report.derivations_added >= cap:
                break
        report.duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        self.total_derivations += report.derivations_added
        return report

    # -- internals -----------------------------------------------------

    def _closure_pass(
        self, kind: str, report: ForwardChainReport, cap: int,
    ) -> int:
        """A → B and B → C (same kind) → A → C, if not already present."""

        if self.universe is None:
            return 0
        # Index neighbors per source for this kind.
        outgoing: dict[str, list[Any]] = defaultdict(list)
        try:
            for concept in self.universe.all_concepts():
                outgoing[concept.name] = [
                    rel for rel in self.universe.neighbors(concept.name, kinds=[kind])
                ]
        except Exception:
            return 0
        added = 0
        for source, edges in outgoing.items():
            for first_edge in edges:
                mid = first_edge.target
                for second_edge in outgoing.get(mid, []):
                    target = second_edge.target
                    if target == source:
                        continue
                    # Already present?
                    already = any(
                        e.target == target for e in outgoing.get(source, [])
                    )
                    if already:
                        continue
                    try:
                        self.universe.add_relation(
                            source, target, kind,
                            weight=min(first_edge.weight, second_edge.weight) * 0.9,
                            notes=f"derived via transitive {kind}: {source}->{mid}->{target}",
                        )
                    except Exception:
                        continue
                    outgoing[source].append(_StubEdge(target, kind, first_edge.weight))
                    added += 1
                    report.derived_facts.append(DerivedFact(
                        source=source, target=target, kind=kind,
                        derived_via=[source, mid, target],
                        confidence=min(
                            float(first_edge.weight or 0.7),
                            float(second_edge.weight or 0.7),
                        ) * 0.9,
                    ))
                    report.derivations_added += 1
                    if report.derivations_added >= cap:
                        return added
        return added


@dataclass
class _StubEdge:
    """Lightweight stand-in for a Relation when we don't want to re-read."""

    target: str
    kind: str
    weight: float = 0.7


__all__ = ["DerivedFact", "ForwardChainReport", "ForwardChainer"]
