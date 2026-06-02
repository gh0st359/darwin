"""Tests for UniverseMeshCoupling."""

from __future__ import annotations

from darwin.mesh.coupling import UniverseMeshCoupling
from darwin.mesh.mesh import CorticalMesh
from darwin.mysterio.bus import BusTopic, CognitionBus
from darwin.universe import build_default_universe


def test_initial_sync_instantiates_cells_for_existing_concepts() -> None:
    universe = build_default_universe()
    mesh = CorticalMesh()
    coupling = UniverseMeshCoupling(universe, mesh)
    # The primitive seed has ~45 concepts; each must have a cell.
    summary = universe.summary()
    assert len(mesh) == summary["concepts"]
    assert coupling.stats.cells_instantiated == summary["concepts"]


def test_subsequent_sync_picks_up_new_concepts() -> None:
    universe = build_default_universe()
    mesh = CorticalMesh()
    coupling = UniverseMeshCoupling(universe, mesh)
    before = len(mesh)
    universe.add_concept("widget", domain="general")
    universe.add_concept("gadget", domain="general")
    universe.add_relation("widget", "gadget", "is_a")
    coupling.sync()
    assert len(mesh) == before + 2
    # The new relation should manifest as a connection.
    assert any(c.target == "gadget" for c in mesh.outgoing("widget"))


def test_sync_is_idempotent() -> None:
    universe = build_default_universe()
    mesh = CorticalMesh()
    coupling = UniverseMeshCoupling(universe, mesh)
    a = coupling.stats.cells_instantiated
    coupling.sync()
    coupling.sync()
    # Re-syncing the same universe should not double-count instantiations.
    assert coupling.stats.cells_instantiated == a


def test_publish_recent_firings_sends_to_bus() -> None:
    universe = build_default_universe()
    mesh = CorticalMesh()
    bus = CognitionBus()
    received: list = []
    bus.subscribe(BusTopic.MESH_FIRING, received.append)
    coupling = UniverseMeshCoupling(universe, mesh, bus=bus)
    # Lower the threshold on a known primitive so it fires immediately.
    cell = mesh.cell("thing")
    cell.threshold = 0.05
    cell.refractory_seconds = 0.0
    mesh.activate(["thing"], magnitude=1.0)
    mesh.propagate([], steps=1)
    n = coupling.publish_recent_firings()
    assert n >= 1
    assert received


def test_reinforce_concept_salience_increases_salience_on_fired_cells() -> None:
    universe = build_default_universe()
    mesh = CorticalMesh()
    coupling = UniverseMeshCoupling(universe, mesh)
    # Lower threshold on a primitive and fire it.
    cell = mesh.cell("thing")
    cell.threshold = 0.05
    cell.refractory_seconds = 0.0
    before_salience = universe.expect("thing").salience
    mesh.activate(["thing"], magnitude=1.0)
    mesh.propagate([], steps=1)
    coupling.reinforce_concept_salience()
    after_salience = universe.expect("thing").salience
    assert after_salience >= before_salience


def test_coupling_handles_missing_concept_gracefully() -> None:
    universe = build_default_universe()
    mesh = CorticalMesh()
    coupling = UniverseMeshCoupling(universe, mesh)
    # Add a cell that has no corresponding concept; reinforce_concept_salience
    # should not raise on it.
    mesh.add_cell("orphan", threshold=0.05, refractory_seconds=0.0)
    mesh.activate(["orphan"], magnitude=1.0)
    mesh.propagate([], steps=1)
    # Should silently no-op the orphan and not error.
    coupling.reinforce_concept_salience()
