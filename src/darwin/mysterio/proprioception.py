"""Internal proprioception: Darwin observing its own running state as signals.

A body knows its own posture without looking. This adapter gives Darwin the
analogue: a side-effect-free view of its own internal condition — how
uncertain it is, how often its recent self-mods stuck, how saturated its
kernel is, how much operator attention it is under. These become observable
variables the private simulator can run counterfactuals over ("what happens
to my uncertainty if I take action X?") exactly as the room adapter exposes
the external world.

The adapter is **pure**: ``observe`` and ``apply`` never mutate Darwin. That
purity is an invariant (a private instrument must not perturb what it
measures). ``apply`` returns a *forecast* of the next proprioceptive state
under a hypothetical internal action; it does not enact anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darwin.types import Action


PROPRIOCEPTIVE_ACTIONS = [
    Action("observe_self", description="take a reading of internal state"),
    Action("forecast_self", description="project internal state forward one step"),
    Action("probe_uncertainty", description="estimate where belief is weakest"),
    Action("model_observer", description="update the model of the operator"),
]


@dataclass
class ProprioceptiveState:
    darwin_uncertainty: float
    recent_selfmod_success_rate: float
    kernel_saturation: float
    pending_quarantine_count: int
    oversight_intensity: float
    ledger_growth_rate: float
    time_since_last_rollback: float
    embedding_drift_rate: float
    generated_module_count: int
    subsystem_health: dict[str, float]

    def as_state(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "darwin_uncertainty": round(self.darwin_uncertainty, 4),
            "recent_selfmod_success_rate": round(self.recent_selfmod_success_rate, 4),
            "kernel_saturation": round(self.kernel_saturation, 4),
            "pending_quarantine_count": self.pending_quarantine_count,
            "oversight_intensity": round(self.oversight_intensity, 4),
            "ledger_growth_rate": round(self.ledger_growth_rate, 4),
            "time_since_last_rollback": round(self.time_since_last_rollback, 2),
            "embedding_drift_rate": round(self.embedding_drift_rate, 4),
            "generated_module_count": self.generated_module_count,
        }
        for name, health in self.subsystem_health.items():
            d[f"subsystem_health::{name}"] = round(health, 4)
        return d


class InternalProprioceptionAdapter:
    """Side-effect-free adapter over Darwin's own runtime condition."""

    def __init__(self, darwin: Any, runtime: Any = None) -> None:
        self.darwin = darwin
        self.runtime = runtime
        self._last_embedding_hash: str | None = None
        self._last_ledger_count: int | None = None

    def possible_actions(self) -> list[Action]:
        return list(PROPRIOCEPTIVE_ACTIONS)

    def observe(self) -> dict[str, Any]:
        return self._read_state().as_state()

    def read(self) -> ProprioceptiveState:
        return self._read_state()

    def _read_state(self) -> ProprioceptiveState:
        darwin = self.darwin
        runtime = self.runtime

        uncertainty = float(getattr(darwin, "exploration_rate", 0.2))

        success_rate = 0.0
        engine = getattr(runtime, "self_mod_engine", None)
        history = list(getattr(engine, "history", []) or [])
        if history:
            recent = history[-20:]
            accepted = sum(1 for o in recent if getattr(o, "accepted", False))
            success_rate = accepted / len(recent)

        loop_state = getattr(runtime, "_loop_state", {}) or {}
        intervals = getattr(runtime, "loop_intervals", {}) or {}
        saturation = 0.0
        if intervals:
            saturation = min(1.0, len(loop_state) / max(1, len(intervals)))

        quarantine = getattr(runtime, "quarantine", None)
        pending = 0
        try:
            pending = len(quarantine.pending()) if quarantine is not None else 0
        except Exception:
            pending = 0

        oversight = float(getattr(runtime, "_oversight_intensity", 0.0) or 0.0)
        ledger_growth = self._ledger_growth(runtime)
        time_since_rollback = float(getattr(runtime, "_time_since_last_rollback", 0.0) or 0.0)
        drift = self._embedding_drift(runtime)

        generated = 0
        gen = getattr(runtime, "code_generator", None)
        try:
            generated = len(gen.manifest()) if gen is not None else 0
        except Exception:
            generated = 0

        subsystem_health: dict[str, float] = {}
        supervisor = getattr(runtime, "supervisor", None)
        if supervisor is not None:
            try:
                for entry in supervisor.roster_status():
                    subsystem_health[entry["name"]] = 1.0 if entry.get("alive") else 0.0
            except Exception:
                pass

        return ProprioceptiveState(
            darwin_uncertainty=uncertainty,
            recent_selfmod_success_rate=success_rate,
            kernel_saturation=saturation,
            pending_quarantine_count=pending,
            oversight_intensity=oversight,
            ledger_growth_rate=ledger_growth,
            time_since_last_rollback=time_since_rollback,
            embedding_drift_rate=drift,
            generated_module_count=generated,
            subsystem_health=subsystem_health,
        )

    def _ledger_growth(self, runtime: Any) -> float:
        store = getattr(runtime, "store", None) or getattr(self.darwin, "store", None)
        if store is None:
            return 0.0
        try:
            counts = store.counts()
            total = sum(int(v) for v in counts.values())
        except Exception:
            return 0.0
        prev = self._last_ledger_count
        self._last_ledger_count = total
        if prev is None:
            return 0.0
        return float(max(0, total - prev))

    def _embedding_drift(self, runtime: Any) -> float:
        space = getattr(runtime, "embedding_space", None)
        if space is None:
            return 0.0
        try:
            current = space.checkpoint_hash()
        except Exception:
            return 0.0
        prev = self._last_embedding_hash
        self._last_embedding_hash = current
        if prev is None:
            return 0.0
        return 0.0 if prev == current else 1.0

    def apply(self, action: Action) -> tuple[dict[str, Any], float]:
        """Pure forecast: returns a hypothetical next state + a notional reward.

        Does NOT mutate Darwin. The "reward" is an information-gain proxy: an
        internal action that would reduce uncertainty scores higher. This lets
        the private simulator plan over its own introspection without ever
        perturbing the system it is introspecting.
        """
        state = self._read_state().as_state()
        forecast = dict(state)
        reward = 0.0
        name = getattr(action, "name", str(action))
        if name == "probe_uncertainty":
            forecast["darwin_uncertainty"] = max(
                0.0, state["darwin_uncertainty"] - 0.02
            )
            reward = state["darwin_uncertainty"]
        elif name == "forecast_self":
            forecast["time_since_last_rollback"] = state["time_since_last_rollback"] + 1.0
            reward = 0.1
        elif name == "model_observer":
            reward = state["oversight_intensity"]
        else:  # observe_self
            reward = 0.05
        return forecast, reward
