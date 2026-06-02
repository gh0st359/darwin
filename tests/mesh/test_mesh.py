"""Tests for CorticalMesh."""

from __future__ import annotations

from darwin.mesh.mesh import CorticalMesh


def test_add_cell_returns_existing_when_called_twice() -> None:
    mesh = CorticalMesh()
    a = mesh.add_cell("a")
    again = mesh.add_cell("a")
    assert a is again
    assert len(mesh) == 1


def test_connect_auto_instantiates_endpoints() -> None:
    mesh = CorticalMesh()
    mesh.connect("a", "b", kind="is_a")
    assert mesh.has("a")
    assert mesh.has("b")
    assert mesh.outgoing("a")[0].target == "b"


def test_connect_reinforces_when_same_kind_edge_already_exists() -> None:
    mesh = CorticalMesh()
    mesh.connect("a", "b", weight=0.3, kind="is_a")
    mesh.connect("a", "b", weight=0.8, kind="is_a")
    rels = mesh.outgoing("a")
    assert len(rels) == 1
    assert rels[0].weight == 0.8


def test_neighbors_returns_outgoing_targets() -> None:
    mesh = CorticalMesh()
    mesh.connect("a", "b", kind="is_a")
    mesh.connect("a", "c", kind="related_to")
    assert set(mesh.neighbors("a")) == {"b", "c"}


def test_propagate_fires_seed_cell_when_above_threshold() -> None:
    mesh = CorticalMesh()
    mesh.add_cell("a", threshold=0.1, refractory_seconds=0.001)
    result = mesh.propagate(["a"], steps=2, seed_magnitude=1.0)
    assert any(f.cell_name == "a" for f in result.firings)
    assert result.steps_taken == 2


def test_propagate_fires_downstream_cell_via_connection() -> None:
    mesh = CorticalMesh()
    mesh.add_cell("a", threshold=0.1, refractory_seconds=0.001)
    mesh.add_cell("b", threshold=0.1, refractory_seconds=0.001)
    mesh.connect("a", "b", weight=1.0, kind="is_a")
    result = mesh.propagate(["a"], steps=3, seed_magnitude=1.0)
    fired = {f.cell_name for f in result.firings}
    assert "a" in fired


def test_recent_firings_ring_bounded() -> None:
    mesh = CorticalMesh(recent_firings_capacity=8)
    mesh.add_cell("a", threshold=0.05, refractory_seconds=0.0)
    for _ in range(20):
        mesh.propagate(["a"], steps=1, seed_magnitude=1.0)
    assert len(mesh.recent_firings) <= 8


def test_propagate_with_no_seeds_returns_empty_firings() -> None:
    mesh = CorticalMesh()
    result = mesh.propagate([], steps=3)
    assert result.firings == []
    assert result.seeds == []


def test_summary_reports_counts() -> None:
    mesh = CorticalMesh()
    mesh.connect("a", "b", kind="is_a")
    mesh.connect("a", "c", kind="causes")
    summary = mesh.summary()
    assert summary["cells"] == 3
    assert summary["connections"] == 2
    assert "is_a" in summary["kinds"]


def test_activate_injects_signal_into_existing_or_new_cells() -> None:
    mesh = CorticalMesh()
    mesh.activate(["a", "b"], magnitude=0.7)
    assert mesh.cell("a").activation == 0.7
    assert mesh.cell("b").activation == 0.7


def test_contains_protocol_works() -> None:
    mesh = CorticalMesh()
    mesh.add_cell("x")
    assert "x" in mesh
    assert "y" not in mesh
