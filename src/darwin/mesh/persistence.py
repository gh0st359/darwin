"""MeshPersistence — atomic JSON save/load for the cortical mesh.

Serializes every cell and every connection to a single JSON file routed
through ``darwin.paths.data_dir()`` so the test isolation fixture
automatically sandboxes per-test mesh state. Writes are debounced: a
save only fires when the number of new propagations since the last save
exceeds a threshold, OR an explicit ``save()`` call is made.

Backward-compatible: missing fields read as defaults. Loading does NOT
clear an existing mesh — it merges, with the file's cell state
overriding only when the in-memory cell is at default activation.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darwin.mesh.cell import ConceptCell, Connection
from darwin.mesh.mesh import CorticalMesh


@dataclass
class MeshPersistenceState:
    last_saved_at: float = 0.0
    last_propagation_count_saved: int = 0
    save_count: int = 0
    load_count: int = 0


class MeshPersistence:
    """JSON-backed persistence for the cortical mesh."""

    def __init__(
        self,
        path: str | Path,
        *,
        save_every_n_propagations: int = 10,
    ) -> None:
        self.path = Path(path)
        self.save_every_n_propagations = int(save_every_n_propagations)
        self.state = MeshPersistenceState()

    # -- save ------------------------------------------------------------

    def maybe_save(self, mesh: CorticalMesh) -> bool:
        """Save the mesh if enough propagations have elapsed since last save."""

        delta = mesh._propagation_count - self.state.last_propagation_count_saved
        if delta < self.save_every_n_propagations:
            return False
        return self.save(mesh)

    def save(self, mesh: CorticalMesh) -> bool:
        """Atomically serialize the mesh to disk."""

        payload = self._encode(mesh)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix="mesh_", suffix=".json", dir=str(self.path.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(tmp_path, self.path)
            self.state.save_count += 1
            self.state.last_saved_at = time.time()
            self.state.last_propagation_count_saved = mesh._propagation_count
            return True
        except OSError:
            return False

    # -- load ------------------------------------------------------------

    def load_into(self, mesh: CorticalMesh) -> int:
        """Read the on-disk mesh into ``mesh``. Returns the number of cells loaded."""

        if not self.path.exists():
            return 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(payload, dict):
            return 0
        cells_loaded = 0
        for record in payload.get("cells", []) or []:
            try:
                name = record["name"]
                mesh.add_cell(
                    name,
                    threshold=float(record.get("threshold", 0.5)),
                    refractory_seconds=float(record.get("refractory_seconds", 0.05)),
                    salience=float(record.get("salience", 1.0)),
                )
                cell = mesh.cell(name)
                if cell is not None:
                    # Only restore prior activation if the cell is at rest.
                    if cell.activation == 0.0:
                        cell.activation = float(record.get("activation", 0.0))
                    cell.last_fired_at = float(record.get("last_fired_at", 0.0))
                    cell.fire_count = int(record.get("fire_count", 0))
                cells_loaded += 1
            except (KeyError, TypeError, ValueError):
                continue
        for record in payload.get("connections", []) or []:
            try:
                mesh.connect(
                    record["source"],
                    record["target"],
                    weight=float(record.get("weight", 0.5)),
                    kind=record.get("kind", "related_to"),
                    delay=float(record.get("delay", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
        self.state.load_count += 1
        return cells_loaded

    # -- helpers ---------------------------------------------------------

    def _encode(self, mesh: CorticalMesh) -> dict[str, Any]:
        cells = [cell.to_record() for cell in mesh.all_cells()]
        connections: list[dict[str, Any]] = []
        for cell in mesh.all_cells():
            for conn in mesh.outgoing(cell.name):
                connections.append(conn.to_record())
        # Deterministic ordering for diff-stability.
        cells.sort(key=lambda r: r["name"])
        connections.sort(key=lambda r: (r["source"], r["kind"], r["target"]))
        return {
            "version": 1,
            "saved_at": time.time(),
            "cells": cells,
            "connections": connections,
            "summary": mesh.summary(),
        }


def default_mesh_path() -> Path:
    """Convenience: default location for the mesh JSON next to the universe."""

    from darwin.paths import data_dir

    return data_dir() / "darwin_mesh.json"


__all__ = ["MeshPersistence", "MeshPersistenceState", "default_mesh_path"]
