"""CorticalMesh — the activation substrate over Darwin's concept graph.

The mesh holds ``ConceptCell`` instances keyed by concept-name and an
adjacency map of typed ``Connection`` edges. ``propagate`` seeds the
named cells with activation, then iterates a bounded number of steps in
which every firing cell transmits along its outgoing connections and
delivers signal to its neighbors. A bounded ring buffer of recent firings
is the substrate-level analogue of working memory.

The mesh is thread-safe under a single reentrant lock. Cells and
connections can be added at any time. Removal is supported but
discouraged at runtime; rollback of a learned mistake should go through
the meta-gate, not by surgical deletion.

Pure-Python ceiling: 100K cells / 10M connections. The V-Scale torch
backend will trade the per-cell dict-of-dataclasses representation for
contiguous tensor slabs while preserving this API.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.mesh.cell import ConceptCell, Connection


@dataclass
class FiringEvent:
    """One cell firing during a propagation step."""

    cell_name: str
    step: int
    activation_before_fire: float
    at: float = field(default_factory=time.monotonic)

    def to_record(self) -> dict[str, Any]:
        return {
            "cell_name": self.cell_name,
            "step": self.step,
            "activation_before_fire": round(self.activation_before_fire, 6),
            "at": self.at,
        }


@dataclass
class PropagationResult:
    """Summary of one ``propagate`` invocation."""

    seeds: list[str]
    steps_taken: int
    firings: list[FiringEvent]
    final_activation_total: float

    def to_record(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "steps_taken": self.steps_taken,
            "firings": [f.to_record() for f in self.firings],
            "final_activation_total": round(self.final_activation_total, 4),
            "firing_count": len(self.firings),
        }


class CorticalMesh:
    """The cell store, adjacency, and propagation engine."""

    def __init__(self, *, recent_firings_capacity: int = 4096) -> None:
        self._lock = threading.RLock()
        self._cells: dict[str, ConceptCell] = {}
        self._outgoing: dict[str, list[Connection]] = defaultdict(list)
        self._incoming: dict[str, list[Connection]] = defaultdict(list)
        self._recent_firings: deque[FiringEvent] = deque(
            maxlen=int(recent_firings_capacity)
        )
        self._propagation_count = 0
        self._created_at = time.time()

    # -- structural mutation ---------------------------------------------

    def add_cell(self, name: str, **kwargs: Any) -> ConceptCell:
        with self._lock:
            existing = self._cells.get(name)
            if existing is not None:
                # Enrich tunables without resetting state.
                if "threshold" in kwargs:
                    existing.threshold = float(kwargs["threshold"])
                if "refractory_seconds" in kwargs:
                    existing.refractory_seconds = float(kwargs["refractory_seconds"])
                if "salience" in kwargs:
                    existing.salience = max(existing.salience, float(kwargs["salience"]))
                return existing
            cell = ConceptCell(name=name, **kwargs)
            self._cells[name] = cell
            return cell

    def connect(
        self,
        source: str,
        target: str,
        *,
        weight: float = 0.5,
        kind: str = "related_to",
        delay: float = 0.0,
    ) -> Connection:
        with self._lock:
            # Auto-instantiate endpoints so callers don't need to
            # pre-register both before adding an edge.
            self.add_cell(source)
            self.add_cell(target)
            # If a same-kind edge between these endpoints already exists,
            # reinforce instead of duplicating.
            for existing in self._outgoing[source]:
                if existing.target == target and existing.kind == kind:
                    existing.reinforce(weight - existing.weight)
                    return existing
            conn = Connection(
                source=source, target=target, weight=float(weight),
                kind=kind, delay=float(delay),
            )
            self._outgoing[source].append(conn)
            self._incoming[target].append(conn)
            return conn

    # -- reads ------------------------------------------------------------

    def cell(self, name: str) -> ConceptCell | None:
        return self._cells.get(name)

    def has(self, name: str) -> bool:
        return name in self._cells

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        return len(self._cells)

    def all_cells(self) -> list[ConceptCell]:
        with self._lock:
            return list(self._cells.values())

    def outgoing(self, name: str) -> list[Connection]:
        with self._lock:
            return list(self._outgoing.get(name, ()))

    def incoming(self, name: str) -> list[Connection]:
        with self._lock:
            return list(self._incoming.get(name, ()))

    def neighbors(self, name: str) -> list[str]:
        with self._lock:
            return [rel.target for rel in self._outgoing.get(name, ())]

    @property
    def recent_firings(self) -> list[FiringEvent]:
        """Snapshot of the recent firing ring (oldest → newest)."""

        with self._lock:
            return list(self._recent_firings)

    # -- propagation ------------------------------------------------------

    def activate(self, names: Iterable[str], magnitude: float = 1.0) -> None:
        """Inject activation into the named cells (auto-creating them)."""

        with self._lock:
            for name in names:
                cell = self.add_cell(name)
                cell.receive(float(magnitude))

    def propagate(
        self,
        seed_cells: Iterable[str] | None = None,
        *,
        steps: int = 3,
        decay: float = 0.6,
        seed_magnitude: float = 1.0,
        now: float | None = None,
    ) -> PropagationResult:
        """Iterate activation through the mesh.

        On step 0, seed cells receive ``seed_magnitude``. On every
        subsequent step, every cell that fires (activation ≥ threshold
        and not refractory) transmits along its outgoing connections;
        every non-firing cell decays by ``decay`` factor.
        """

        moment = now if now is not None else time.monotonic()
        seeds_list = list(seed_cells or ())
        firings: list[FiringEvent] = []
        with self._lock:
            self._propagation_count += 1
            for name in seeds_list:
                cell = self.add_cell(name)
                cell.receive(float(seed_magnitude))
            steps_taken = 0
            for step in range(int(max(0, steps))):
                steps_taken += 1
                fired_this_step: list[ConceptCell] = []
                # Snapshot cells before mutating; iteration over the dict
                # value view is otherwise unsafe under add_cell calls.
                for cell in list(self._cells.values()):
                    activation_before = cell.activation
                    if cell.maybe_fire(now=moment):
                        firings.append(FiringEvent(
                            cell_name=cell.name,
                            step=step,
                            activation_before_fire=activation_before,
                            at=moment,
                        ))
                        self._recent_firings.append(firings[-1])
                        fired_this_step.append(cell)
                    else:
                        cell.decay(decay)
                # Propagate from firers to their downstream targets.
                for source_cell in fired_this_step:
                    for conn in self._outgoing.get(source_cell.name, ()):
                        signal = conn.transmit(source_cell.activation + activation_before * 0.0, now=moment)
                        # transmit() needs the pre-fire activation as the basis.
                        # We re-compute it cleanly from the captured value
                        # by deriving signal off the pre-fire level.
                        target = self._cells.get(conn.target)
                        if target is None:
                            continue
                        # Use the fired cell's activation_before_fire stored in the
                        # firing event for fidelity.
                        for f in reversed(firings):
                            if f.cell_name == source_cell.name and f.step == step:
                                pre = f.activation_before_fire
                                signal = max(-2.0, min(2.0, pre * conn.weight))
                                break
                        target.receive(signal)
            final_total = sum(c.activation for c in self._cells.values())
            return PropagationResult(
                seeds=list(seeds_list),
                steps_taken=steps_taken,
                firings=firings,
                final_activation_total=final_total,
            )

    # -- introspection ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            n_conns = sum(len(adj) for adj in self._outgoing.values())
            kinds: dict[str, int] = defaultdict(int)
            for adj in self._outgoing.values():
                for conn in adj:
                    kinds[conn.kind] += 1
            return {
                "cells": len(self._cells),
                "connections": n_conns,
                "kinds": dict(kinds),
                "recent_firings": len(self._recent_firings),
                "propagation_count": self._propagation_count,
                "age_seconds": time.time() - self._created_at,
            }


__all__ = ["CorticalMesh", "FiringEvent", "PropagationResult"]
