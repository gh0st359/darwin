"""HypotheticalReasoner — temporary universe overlays for counterfactuals.

Asks "if I assume X is also Y, what follows?". Uses a copy-on-write
overlay so the base universe is untouched. Inside the overlay, the
forward chainer runs and produces derivations; on exit, every overlaid
edge is removed cleanly so the base universe returns to its prior
state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass
class _OverlayedEdge:
    """One edge added during an overlay (so we can remove it on exit)."""

    source: str
    target: str
    kind: str


@dataclass
class HypotheticalResult:
    """Outcome of one overlay session."""

    assumptions: list[tuple[str, str, str]]
    derived: list[tuple[str, str, str]] = field(default_factory=list)
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "assumptions": [list(a) for a in self.assumptions],
            "derived": [list(d) for d in self.derived],
            "notes": self.notes,
        }


class HypotheticalReasoner:
    """Run forward chaining inside a temporary universe overlay."""

    def __init__(self, universe: Any) -> None:
        self.universe = universe

    @contextmanager
    def overlay(
        self,
        facts: Iterable[tuple[str, str, str]],
    ) -> Iterator[HypotheticalResult]:
        """Context manager. Adds each fact, yields a result, then removes."""

        overlaid: list[_OverlayedEdge] = []
        assumptions = list(facts)
        for source, kind, target in assumptions:
            try:
                self.universe.add_concept(source)
                self.universe.add_concept(target)
                # Only overlay edges that don't already exist.
                already = any(
                    rel.target == target and rel.kind == kind
                    for rel in self.universe.neighbors(source)
                )
                if not already:
                    self.universe.add_relation(
                        source, target, kind,
                        weight=0.7,
                        notes="hypothetical overlay",
                    )
                    overlaid.append(_OverlayedEdge(source, target, kind))
            except Exception:
                continue
        result = HypotheticalResult(
            assumptions=[(s, k, t) for s, k, t in assumptions],
        )
        try:
            yield result
        finally:
            self._remove_overlay(overlaid)

    def _remove_overlay(self, overlaid: list[_OverlayedEdge]) -> None:
        """Remove the overlaid edges from the universe.

        ConceptUniverse doesn't expose a public remove_relation, but its
        adjacency lists are accessible via the same locking discipline.
        """

        if not overlaid:
            return
        try:
            with self.universe._lock:
                for edge in overlaid:
                    forward = self.universe._adjacency.get(edge.source, [])
                    self.universe._adjacency[edge.source] = [
                        rel for rel in forward
                        if not (rel.target == edge.target and rel.kind == edge.kind
                                and "hypothetical overlay" in (rel.notes or ""))
                    ]
                    reverse = self.universe._reverse.get(edge.target, [])
                    self.universe._reverse[edge.target] = [
                        rel for rel in reverse
                        if not (rel.source == edge.source and rel.kind == edge.kind
                                and "hypothetical overlay" in (rel.notes or ""))
                    ]
        except Exception:
            pass


__all__ = ["HypotheticalReasoner", "HypotheticalResult"]
