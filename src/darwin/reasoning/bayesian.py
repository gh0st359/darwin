"""BeliefNetwork — probabilistic belief propagation over the concept graph.

Uses the universe's typed edges as conditional dependency hints. Each
concept has a prior probability (default 0.5). Each edge carries a
conditional P(target | source) derived from the edge's weight (clamped
to [0.05, 0.95] so log-odds remain finite).

Provides:
  * ``set_prior(name, p)`` — operator-supplied evidence.
  * ``query(name)`` — returns the current posterior at that concept,
    computed by walking incoming edges and combining via noisy-OR.
  * ``propagate(steps=N)`` — one belief-propagation sweep.

This is a lightweight pure-Python implementation. The V-Scale phase can
swap in a torch-backed message-passing kernel without changing this
module's API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, low: float = 0.05, high: float = 0.95) -> float:
    return max(low, min(high, value))


@dataclass
class BeliefNode:
    """A concept's probabilistic state."""

    name: str
    prior: float = 0.5
    posterior: float = 0.5
    evidence_count: int = 0


@dataclass
class BeliefReport:
    """A snapshot of the belief network's state."""

    nodes: dict[str, float] = field(default_factory=dict)
    max_change: float = 0.0
    steps_taken: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "max_change": round(self.max_change, 4),
            "steps_taken": self.steps_taken,
        }


class BeliefNetwork:
    """Bayesian belief layer over a ConceptUniverse."""

    def __init__(self, universe: Any) -> None:
        self.universe = universe
        self._nodes: dict[str, BeliefNode] = {}
        self._evidence: dict[str, float] = {}

    def set_prior(self, name: str, p: float) -> None:
        node = self._get_or_create(name)
        node.prior = _clamp(p)
        node.posterior = node.prior

    def set_evidence(self, name: str, p: float) -> None:
        """Pin a node's posterior to p (hard evidence)."""

        self._evidence[name] = _clamp(p)
        node = self._get_or_create(name)
        node.posterior = self._evidence[name]
        node.evidence_count += 1

    def query(self, name: str) -> float:
        node = self._nodes.get(name)
        if node is None:
            return 0.5  # uninformative
        return node.posterior

    def propagate(self, *, steps: int = 4, damping: float = 0.5) -> BeliefReport:
        """Run ``steps`` belief-propagation sweeps over the universe."""

        report = BeliefReport()
        if self.universe is None:
            return report
        # Materialise all relations once.
        try:
            relations = self.universe.relations()
        except Exception:
            return report
        # Ensure every concept has a node.
        try:
            for concept in self.universe.all_concepts():
                self._get_or_create(concept.name)
        except Exception:
            return report
        for step in range(steps):
            report.steps_taken += 1
            new_posteriors: dict[str, float] = {}
            # Aggregate evidence into each target from its incoming edges.
            incoming_msgs: dict[str, list[tuple[float, float]]] = {}
            for rel in relations:
                source = self._nodes.get(rel.source)
                if source is None:
                    continue
                # Conditional weight: clamp the symbolic weight into a
                # probability range. Negative weights are inverse evidence.
                w = float(getattr(rel, "weight", 0.5) or 0.5)
                p_cond = _clamp(0.5 + 0.5 * w, 0.05, 0.95)
                incoming_msgs.setdefault(rel.target, []).append(
                    (source.posterior, p_cond)
                )
            # Compute noisy-OR style posterior for each node.
            for name, node in self._nodes.items():
                if name in self._evidence:
                    new_posteriors[name] = self._evidence[name]
                    continue
                msgs = incoming_msgs.get(name, [])
                if not msgs:
                    new_posteriors[name] = node.prior
                    continue
                # Noisy-OR: P(¬target) = prod((1 - p_source * p_cond))
                p_not = 1.0
                for src_p, cond in msgs:
                    p_not *= 1.0 - src_p * cond
                computed = 1.0 - p_not
                damped = damping * node.posterior + (1 - damping) * computed
                new_posteriors[name] = _clamp(damped)
            # Apply.
            max_change = 0.0
            for name, p in new_posteriors.items():
                node = self._nodes[name]
                delta = abs(node.posterior - p)
                max_change = max(max_change, delta)
                node.posterior = p
            report.max_change = max_change
            if max_change < 1e-4:
                break
        report.nodes = {n.name: n.posterior for n in self._nodes.values()}
        return report

    def _get_or_create(self, name: str) -> BeliefNode:
        node = self._nodes.get(name)
        if node is None:
            node = BeliefNode(name=name)
            self._nodes[name] = node
        return node

    def summary(self) -> dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "evidence_pinned": len(self._evidence),
            "highest_posterior": max(
                (n.posterior for n in self._nodes.values()), default=0.0,
            ),
            "lowest_posterior": min(
                (n.posterior for n in self._nodes.values()), default=0.0,
            ),
        }


__all__ = ["BeliefNetwork", "BeliefNode", "BeliefReport"]
