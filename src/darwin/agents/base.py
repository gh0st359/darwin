"""Agent base class and Solution dataclass.

Every benchmark-targeted agent is a composition of {mesh activation,
speech production, knowledge ingest, reasoning derivation}. The base
class defines the ``solve(problem)`` contract every agent honours; the
``Solution`` dataclass is what every agent returns.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Solution:
    """The result of one solve() call."""

    agent: str
    answer: str = ""
    confidence: float = 0.0
    steps: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    succeeded: bool = False
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "step_count": len(self.steps),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "succeeded": self.succeeded,
            "notes": self.notes,
        }


class Agent(ABC):
    """Abstract base class for cognitive agents."""

    name: str = "agent"

    def __init__(self, runtime: Any = None) -> None:
        self.runtime = runtime

    @abstractmethod
    def solve(self, problem: Any) -> Solution:
        """Solve ``problem`` and return a Solution."""

    def _start(self) -> float:
        return time.perf_counter()

    def _finish(self, solution: Solution, started: float) -> Solution:
        solution.elapsed_ms = (time.perf_counter() - started) * 1000.0
        solution.agent = self.name
        return solution

    def _activate_mesh(self, concepts: list[str]) -> None:
        """Fire the cortical mesh for the given concept names.

        Pure helper — silent on missing runtime / missing mesh.
        """

        if self.runtime is None:
            return
        mesh = getattr(self.runtime, "cortical_mesh", None)
        if mesh is None or not concepts:
            return
        try:
            mesh.propagate(seed_cells=concepts, steps=1, decay=0.85)
        except Exception:
            return


__all__ = ["Agent", "Solution"]
