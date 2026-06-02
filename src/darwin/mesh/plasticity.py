"""Hebbian + spike-timing-dependent plasticity rules.

Plasticity is *how the mesh learns from its own firings*. Two complementary
rules:

  * **Hebbian** — `Δw = η · pre · post`. When both pre- and post-synaptic
    cells are active, the connecting weight strengthens. Pure
    correlation, no timing.
  * **STDP** — spike-timing-dependent plasticity. When the pre-synaptic
    cell fires *before* the post-synaptic cell within τ_+, potentiation
    fires (Δw > 0). When the post fires before the pre within τ_−,
    depression fires (Δw < 0). This is what gives the mesh temporal
    causality: "A causes B" patterns reinforce; "B happened then A" weakens.

The :class:`PlasticityController` composes both rules and applies them
over the recent-firings ring of a :class:`CorticalMesh` per cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from darwin.mesh.mesh import CorticalMesh, FiringEvent


@dataclass
class HebbianRule:
    """Δw = η · pre_activation · post_activation."""

    learning_rate: float = 0.01
    max_weight: float = 1.0
    min_weight: float = -1.0

    def apply(self, *, pre_activation: float, post_activation: float) -> float:
        return self.learning_rate * pre_activation * post_activation


@dataclass
class STDPRule:
    """Spike-timing-dependent plasticity.

    Potentiation when pre fires before post within τ_plus.
    Depression when post fires before pre within τ_minus.
    """

    a_plus: float = 0.02           # potentiation magnitude
    a_minus: float = 0.018         # depression magnitude
    tau_plus_seconds: float = 0.02
    tau_minus_seconds: float = 0.02

    def apply(self, *, pre_time: float, post_time: float) -> float:
        dt = post_time - pre_time
        if dt > 0:
            # Pre then post → potentiation.
            return self.a_plus * math.exp(-dt / max(1e-6, self.tau_plus_seconds))
        if dt < 0:
            # Post then pre → depression.
            return -self.a_minus * math.exp(dt / max(1e-6, self.tau_minus_seconds))
        return 0.0


@dataclass
class PlasticityReport:
    """Summary of one ``apply_cycle``."""

    hebbian_updates: int
    stdp_updates: int
    total_delta_magnitude: float
    affected_pairs: list[tuple[str, str]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "hebbian_updates": self.hebbian_updates,
            "stdp_updates": self.stdp_updates,
            "total_delta_magnitude": round(self.total_delta_magnitude, 6),
            "affected_pairs": [list(p) for p in self.affected_pairs[:32]],
        }


class PlasticityController:
    """Compose Hebbian + STDP and apply over a mesh's recent firings.

    apply_cycle reads the mesh's recent_firings ring and updates the
    weights of every connection whose source fired close in time to its
    target. The Hebbian rule fires on every (pre, post) co-firing; the
    STDP rule fires when both endpoints fired within either the
    potentiation or depression window.
    """

    def __init__(
        self,
        *,
        hebbian: HebbianRule | None = None,
        stdp: STDPRule | None = None,
    ) -> None:
        self.hebbian = hebbian or HebbianRule()
        self.stdp = stdp or STDPRule()

    def apply_cycle(
        self,
        mesh: CorticalMesh,
        *,
        window_seconds: float | None = None,
    ) -> PlasticityReport:
        """One plasticity pass over the mesh's recent firings."""

        firings = mesh.recent_firings
        if len(firings) < 2:
            return PlasticityReport(
                hebbian_updates=0,
                stdp_updates=0,
                total_delta_magnitude=0.0,
            )
        if window_seconds is None:
            window_seconds = max(
                self.stdp.tau_plus_seconds,
                self.stdp.tau_minus_seconds,
            ) * 4.0
        # Build per-cell last-fire timestamps and activations from the ring.
        by_cell: dict[str, list[FiringEvent]] = {}
        for ev in firings:
            by_cell.setdefault(ev.cell_name, []).append(ev)

        hebbian_updates = 0
        stdp_updates = 0
        total_delta = 0.0
        pairs: list[tuple[str, str]] = []

        # Walk every outgoing connection in the mesh, checking whether
        # source and target both fired within the window.
        for source_name, events in by_cell.items():
            for conn in mesh.outgoing(source_name):
                target_events = by_cell.get(conn.target)
                if not target_events:
                    continue
                # Use the most recent pre/post fires for an O(1) update.
                pre_event = events[-1]
                post_event = target_events[-1]
                dt = abs(post_event.at - pre_event.at)
                if dt > window_seconds:
                    continue
                # Hebbian.
                delta_h = self.hebbian.apply(
                    pre_activation=pre_event.activation_before_fire,
                    post_activation=post_event.activation_before_fire,
                )
                delta_h = max(
                    self.hebbian.min_weight - conn.weight,
                    min(self.hebbian.max_weight - conn.weight, delta_h),
                )
                if delta_h != 0.0:
                    conn.reinforce(delta_h)
                    hebbian_updates += 1
                    total_delta += abs(delta_h)
                # STDP.
                delta_s = self.stdp.apply(
                    pre_time=pre_event.at, post_time=post_event.at,
                )
                delta_s = max(
                    self.hebbian.min_weight - conn.weight,
                    min(self.hebbian.max_weight - conn.weight, delta_s),
                )
                if delta_s != 0.0:
                    conn.reinforce(delta_s)
                    stdp_updates += 1
                    total_delta += abs(delta_s)
                if delta_h != 0.0 or delta_s != 0.0:
                    pairs.append((source_name, conn.target))
        return PlasticityReport(
            hebbian_updates=hebbian_updates,
            stdp_updates=stdp_updates,
            total_delta_magnitude=total_delta,
            affected_pairs=pairs,
        )


__all__ = [
    "HebbianRule",
    "PlasticityController",
    "PlasticityReport",
    "STDPRule",
]
