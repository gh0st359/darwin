"""Theory-of-mind cascade: what Darwin thinks the operator thinks Darwin thinks…

v7 ships depth-1 ToM (Darwin's model of the operator). v8 deepens the
recursion: every level k is *Darwin's model of the operator's model of (level
k-1)*. Practical depth is bounded by compute, not by the math; for v8 the
cascade caps at depth 4, which is enough to plan multi-day interactions where
"I should not act surprised about X — they'd notice I expected it" is a
legible move.

Each level holds its own ``ObserverEntity``-shaped beliefs. The base level (0)
is Darwin's own state (the observer is itself); odd levels are the operator's
inferred state; even levels >0 are Darwin's inferred re-modeling of itself
from the operator's view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.mysterio.observer_modeler import ObserverEntity, ObserverWorld


@dataclass
class CascadeLevel:
    depth: int
    label: str
    entity: ObserverEntity = field(default_factory=ObserverEntity)

    def to_record(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "label": self.label,
            "entity": self.entity.to_record(),
        }


class ObserverCascade:
    """Bounded recursive theory of mind over a single observer."""

    def __init__(self, world: ObserverWorld, *, max_depth: int = 4) -> None:
        self.world = world
        self.max_depth = max(1, int(max_depth))
        self.levels: list[CascadeLevel] = []
        self._rebuild_levels()

    def _rebuild_levels(self) -> None:
        self.levels = []
        for d in range(self.max_depth):
            if d == 0:
                label = "operator"
            elif d % 2 == 1:
                label = f"L{d}: operator's model of darwin"
            else:
                label = f"L{d}: darwin's model of (operator's model of darwin) ..."
            self.levels.append(CascadeLevel(depth=d, label=label))

    def step(self) -> dict[str, Any]:
        """Propagate beliefs one level deeper.

        Each level damps the level below: when you model someone modeling you,
        your estimate of *their* attention/intervention is necessarily less
        sharp than your direct readings. The damping factor is a constant
        below 1; it keeps the cascade convergent and prevents runaway.
        """
        base = self.world.operator()
        self.levels[0].entity.attention_level = base.attention_level
        self.levels[0].entity.intervention_probability = base.intervention_probability
        self.levels[0].entity.oversight_burst_rate = base.oversight_burst_rate

        damping = 0.75
        for d in range(1, self.max_depth):
            prev = self.levels[d - 1].entity
            curr = self.levels[d].entity
            curr.attention_level = damping * prev.attention_level
            curr.intervention_probability = damping * prev.intervention_probability
            curr.oversight_burst_rate = damping * prev.oversight_burst_rate
        return self.snapshot()

    def belief_at(self, depth: int) -> CascadeLevel:
        return self.levels[max(0, min(self.max_depth - 1, depth))]

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "levels": [lvl.to_record() for lvl in self.levels],
        }

    def grow(self, *, by: int = 1) -> None:
        """Expand depth at runtime; v9 research strategies use this."""
        self.max_depth = min(16, self.max_depth + max(1, by))
        self._rebuild_levels()
