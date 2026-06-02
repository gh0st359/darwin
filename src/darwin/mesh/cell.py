"""ConceptCell + Connection — the atomic units of the cortical mesh.

A ConceptCell is the neural-substrate twin of a ConceptUniverse Concept.
Every named concept Darwin holds gets one cell; activation flows through
connections (one per typed Relation in the universe). Cells have their own
state — activation level, refractory timer, threshold, last-fired
timestamp, accumulated salience — independent of the symbolic concept
they're bound to. The mesh learns by adjusting connection weights via
Hebbian + STDP plasticity, not by editing the symbolic graph.

Cells are intentionally lightweight (≤ ~80 bytes of Python state each) so
the pure-Python ceiling is 100K cells / 10M connections. The torch
backend in V-Scale will trade this struct-of-arrays layout for tensor
slabs while preserving the API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConceptCell:
    """One activation-bearing node in the cortical mesh."""

    name: str
    activation: float = 0.0
    threshold: float = 0.5
    refractory_seconds: float = 0.05
    last_fired_at: float = 0.0
    salience: float = 1.0
    fire_count: int = 0

    def is_refractory(self, *, now: float | None = None) -> bool:
        moment = now if now is not None else time.monotonic()
        return (moment - self.last_fired_at) < self.refractory_seconds

    def receive(self, signal: float) -> None:
        """Add ``signal`` to current activation (clamped to [0, 5])."""

        self.activation = max(0.0, min(5.0, self.activation + signal))

    def maybe_fire(self, *, now: float | None = None) -> bool:
        """Fire if above threshold and not in refractory window."""

        moment = now if now is not None else time.monotonic()
        if self.activation < self.threshold:
            return False
        if self.is_refractory(now=moment):
            return False
        self.last_fired_at = moment
        self.fire_count += 1
        # Firing partially drains activation but doesn't reset to zero —
        # repeated input within a window can sustain it.
        self.activation *= 0.4
        return True

    def decay(self, factor: float) -> None:
        self.activation *= max(0.0, min(1.0, factor))

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "activation": round(self.activation, 6),
            "threshold": round(self.threshold, 4),
            "refractory_seconds": round(self.refractory_seconds, 4),
            "last_fired_at": self.last_fired_at,
            "salience": round(self.salience, 4),
            "fire_count": self.fire_count,
        }


@dataclass
class Connection:
    """A weighted, typed, directed connection between two cells."""

    source: str
    target: str
    weight: float = 0.5
    kind: str = "related_to"        # mirrors the typed Relation kind
    delay: float = 0.0              # seconds; reserved for V-Scale
    last_traversed_at: float = 0.0
    traversal_count: int = 0

    def transmit(self, source_activation: float, *, now: float | None = None) -> float:
        """Propagate a source cell's activation through this edge.

        Returns the signal (clamped to [-2, 2]) delivered to the target.
        """

        moment = now if now is not None else time.monotonic()
        self.last_traversed_at = moment
        self.traversal_count += 1
        signal = max(-2.0, min(2.0, source_activation * self.weight))
        return signal

    def reinforce(self, delta: float) -> None:
        """Adjust weight by ``delta`` (clamped to [-1, 1])."""

        self.weight = max(-1.0, min(1.0, self.weight + delta))

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 6),
            "kind": self.kind,
            "delay": round(self.delay, 4),
            "last_traversed_at": self.last_traversed_at,
            "traversal_count": self.traversal_count,
        }


__all__ = ["ConceptCell", "Connection"]
