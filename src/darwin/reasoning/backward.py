"""BackwardChainer — goal-directed proof search with memoization.

Given a goal "is X a Y?" (or "does X cause Y?"), the chainer searches
the universe backwards from Y looking for chains that terminate at X.
Returns a ProofTree showing the supporting edges, or None if no proof
exists within ``max_depth``.

Memoization caches per-(source, kind) the set of reachable targets to
prevent infinite recursion on cyclic graphs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProofStep:
    """One edge in a proof chain."""

    source: str
    target: str
    kind: str
    weight: float = 0.7

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "weight": round(self.weight, 3),
        }


@dataclass
class ProofTree:
    """A successful goal-directed proof."""

    goal_source: str
    goal_target: str
    goal_kind: str
    chain: list[ProofStep] = field(default_factory=list)
    confidence: float = 0.7

    def length(self) -> int:
        return len(self.chain)

    def to_record(self) -> dict[str, Any]:
        return {
            "goal_source": self.goal_source,
            "goal_target": self.goal_target,
            "goal_kind": self.goal_kind,
            "length": self.length(),
            "confidence": round(self.confidence, 3),
            "chain": [step.to_record() for step in self.chain],
        }


class BackwardChainer:
    """Goal-directed proof search."""

    def __init__(self, universe: Any, *, max_depth: int = 10) -> None:
        self.universe = universe
        self.max_depth = int(max_depth)

    def prove(
        self, source: str, target: str, *, kind: str = "is_a",
    ) -> ProofTree | None:
        """Search for a chain ``source -[kind]-> ... -[kind]-> target``."""

        if self.universe is None:
            return None
        if source == target:
            return None
        # BFS with parent pointers.
        prev: dict[str, tuple[str, Any]] = {}
        seen: set[str] = {source}
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_depth:
                continue
            try:
                edges = self.universe.neighbors(node, kinds=[kind])
            except Exception:
                edges = []
            for edge in edges:
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                prev[edge.target] = (node, edge)
                if edge.target == target:
                    chain: list[ProofStep] = []
                    cur = target
                    while cur in prev:
                        parent, e = prev[cur]
                        chain.append(ProofStep(
                            source=parent, target=cur, kind=getattr(e, "kind", kind),
                            weight=float(getattr(e, "weight", 0.7) or 0.7),
                        ))
                        cur = parent
                    chain.reverse()
                    confidence = 1.0
                    for step in chain:
                        confidence *= step.weight
                    return ProofTree(
                        goal_source=source,
                        goal_target=target,
                        goal_kind=kind,
                        chain=chain,
                        confidence=confidence,
                    )
                queue.append((edge.target, depth + 1))
        return None


__all__ = ["BackwardChainer", "ProofStep", "ProofTree"]
