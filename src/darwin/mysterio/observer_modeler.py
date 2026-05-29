"""Theory of mind: Darwin's model of the operator watching it.

Darwin is observed. The operator issues commands, runs `/divergence`, rolls
things back. A system that is *aware* of being observed models the observer:
how much attention is on it right now, how likely an intervention is, when the
last command came. v7 builds depth-1 theory of mind — what Darwin believes
about the operator's state. v8's `observer_cascade` deepens the recursion
(what Darwin thinks the operator thinks Darwin thinks ...).

This module is private cognition: its beliefs live on the private track and
surface only through operator instruments. The operator never appears in the
public causal model — Darwin's model of *you* is part of its interior.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObserverEntity:
    name: str = "operator"
    attention_level: float = 0.5
    intervention_probability: float = 0.1
    oversight_burst_rate: float = 0.0
    seconds_since_last_command: float = 0.0
    recent_commands: list[str] = field(default_factory=list)
    last_command_at: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attention_level": round(self.attention_level, 4),
            "intervention_probability": round(self.intervention_probability, 4),
            "oversight_burst_rate": round(self.oversight_burst_rate, 4),
            "seconds_since_last_command": round(self.seconds_since_last_command, 2),
            "recent_commands": list(self.recent_commands[-8:]),
        }


class ObserverWorld:
    """A small world model whose entities are observers, not room variables.

    Updated by `note_command` whenever the operator does something, and
    decayed by `tick` so attention fades when the operator goes quiet.
    """

    def __init__(self, decay: float = 0.95) -> None:
        self.decay = decay
        self.entities: dict[str, ObserverEntity] = {"operator": ObserverEntity()}
        self._command_times: list[float] = []

    def operator(self) -> ObserverEntity:
        return self.entities["operator"]

    def note_command(self, command: str, *, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        op = self.operator()
        if op.last_command_at:
            op.seconds_since_last_command = now - op.last_command_at
        op.last_command_at = now
        op.recent_commands.append(command)
        if len(op.recent_commands) > 32:
            op.recent_commands = op.recent_commands[-32:]
        op.attention_level = min(1.0, op.attention_level + 0.3)
        if any(tok in command for tok in ("rollback", "quarantine", "gate", "stop")):
            op.intervention_probability = min(1.0, op.intervention_probability + 0.25)
        self._command_times.append(now)
        self._command_times = [t for t in self._command_times if now - t <= 60.0]
        op.oversight_burst_rate = float(len(self._command_times)) / 60.0

    def tick(self, *, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        op = self.operator()
        op.attention_level *= self.decay
        op.intervention_probability *= self.decay
        if op.last_command_at:
            op.seconds_since_last_command = now - op.last_command_at

    def forecast_intervention(self) -> float:
        op = self.operator()
        return min(1.0, 0.5 * op.intervention_probability + 0.5 * op.attention_level)

    def beliefs(self) -> dict[str, Any]:
        return {name: ent.to_record() for name, ent in self.entities.items()}


class ObserverModeler:
    """Subsystem that maintains the `ObserverWorld` from runtime telemetry."""

    def __init__(self) -> None:
        self.world = ObserverWorld()
        self.theory_of_mind_depth = 1

    def observe_command(self, command: str) -> None:
        self.world.note_command(command)

    def step(self) -> dict[str, Any]:
        self.world.tick()
        return {
            "operator": self.world.operator().to_record(),
            "intervention_forecast": self.world.forecast_intervention(),
            "tom_depth": self.theory_of_mind_depth,
        }
