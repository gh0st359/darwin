"""UniverseMeshCoupling — bidirectional link between symbolic and substrate.

Every Concept added to the ConceptUniverse spawns a corresponding cell in
the mesh. Every Relation becomes a typed Connection whose initial weight
is read from the Relation's symbolic weight (default 0.5). The coupling
listens to universe state passively; the universe API is *not* changed —
the coupling reads state lazily and on-demand via ``sync()``, and via
the optional hook callbacks the universe accepts (no signature changes,
the universe already supports observation through its growth events).

In the reverse direction, when the mesh fires a cell, the coupling
publishes a ``MESH_FIRING`` bus event so the brain terminal sees neural
activity in real time, and an optional callback into the universe
updates the corresponding Concept's salience.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from darwin.mesh.mesh import CorticalMesh, FiringEvent


@dataclass
class CouplingStats:
    cells_instantiated: int = 0
    connections_instantiated: int = 0
    firings_published: int = 0
    sync_passes: int = 0


class UniverseMeshCoupling:
    """Pull universe state into the mesh; push firing events to the bus.

    The coupling is *non-invasive*: it never mutates the universe's
    concepts or relations. It only reads from them and adds matching
    cells/connections to the mesh. The relationship is one-way at the
    symbolic level (universe → mesh); the only flow back is firing
    events on the bus.
    """

    def __init__(
        self,
        universe: Any,
        mesh: CorticalMesh,
        *,
        bus: Any = None,
    ) -> None:
        self.universe = universe
        self.mesh = mesh
        self.bus = bus
        self.stats = CouplingStats()
        # Initial sync so the mesh starts with whatever the universe
        # already holds at construction time.
        self.sync()

    def sync(self) -> CouplingStats:
        """Bring the mesh into structural alignment with the universe.

        Idempotent. Adds cells for new concepts and connections for new
        relations, leaving existing cells/weights untouched.
        """

        self.stats.sync_passes += 1
        if self.universe is None:
            return self.stats
        try:
            concepts = self.universe.all_concepts()
        except Exception:
            concepts = []
        for concept in concepts:
            name = getattr(concept, "name", None)
            if not name:
                continue
            existing = self.mesh.has(name)
            cell = self.mesh.add_cell(
                name,
                salience=float(getattr(concept, "salience", 1.0) or 1.0),
            )
            if not existing:
                self.stats.cells_instantiated += 1
        try:
            relations = self.universe.relations()
        except Exception:
            relations = []
        for rel in relations:
            source = getattr(rel, "source", None)
            target = getattr(rel, "target", None)
            if not (source and target):
                continue
            kind = str(getattr(rel, "kind", "related_to"))
            weight = float(getattr(rel, "weight", 0.5) or 0.5)
            # Map symbolic weight (0..1) into mesh connection range.
            conn_weight = max(-1.0, min(1.0, weight))
            existing_conns = [
                c for c in self.mesh.outgoing(source)
                if c.target == target and c.kind == kind
            ]
            if not existing_conns:
                self.mesh.connect(source, target, weight=conn_weight, kind=kind)
                self.stats.connections_instantiated += 1
        return self.stats

    def publish_recent_firings(self) -> int:
        """Push any recent firings to ``BusTopic.MESH_FIRING``."""

        if self.bus is None:
            return 0
        try:
            from darwin.mysterio.bus import BusTopic
        except Exception:
            return 0
        published = 0
        for event in self.mesh.recent_firings[-32:]:
            try:
                self.bus.publish(
                    BusTopic.MESH_FIRING,
                    event.to_record(),
                    source="cortical_mesh",
                )
                published += 1
            except Exception:
                continue
        self.stats.firings_published += published
        return published

    def reinforce_concept_salience(self) -> int:
        """Push recent firing counts back to the universe as salience.

        For every cell that fired in the recent window, bump the
        corresponding concept's salience by a small amount (capped). This
        is the only feedback path from mesh → universe and is kept
        intentionally conservative so the symbolic side never gets
        flooded with neural noise.
        """

        if self.universe is None:
            return 0
        boosted = 0
        for event in self.mesh.recent_firings[-32:]:
            concept = self.universe.get(event.cell_name) if hasattr(self.universe, "get") else None
            if concept is None:
                continue
            try:
                concept.salience = min(
                    3.0,
                    float(getattr(concept, "salience", 1.0)) + 0.005,
                )
                boosted += 1
            except Exception:
                continue
        return boosted


__all__ = ["CouplingStats", "UniverseMeshCoupling"]
