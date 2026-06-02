"""Tests for MeshPersistence."""

from __future__ import annotations

from pathlib import Path

from darwin.mesh.mesh import CorticalMesh
from darwin.mesh.persistence import MeshPersistence, default_mesh_path


def test_save_then_load_preserves_cells_and_connections(tmp_path: Path) -> None:
    mesh = CorticalMesh()
    mesh.add_cell("a", threshold=0.4, salience=1.8)
    mesh.connect("a", "b", weight=0.7, kind="is_a")
    persistence = MeshPersistence(tmp_path / "mesh.json")
    persistence.save(mesh)

    reloaded = CorticalMesh()
    n = persistence.load_into(reloaded)
    assert n >= 2
    assert reloaded.has("a")
    assert reloaded.cell("a").threshold == 0.4
    assert reloaded.cell("a").salience == 1.8
    rels = reloaded.outgoing("a")
    assert any(r.target == "b" and r.weight == 0.7 for r in rels)


def test_load_nonexistent_returns_zero(tmp_path: Path) -> None:
    mesh = CorticalMesh()
    persistence = MeshPersistence(tmp_path / "missing.json")
    n = persistence.load_into(mesh)
    assert n == 0


def test_save_is_atomic_via_tempfile_rename(tmp_path: Path) -> None:
    mesh = CorticalMesh()
    mesh.add_cell("a")
    persistence = MeshPersistence(tmp_path / "m.json")
    persistence.save(mesh)
    # No leftover temp files in the parent directory.
    leftovers = [
        p.name for p in tmp_path.iterdir()
        if p.name.startswith("mesh_") and p.name.endswith(".json")
        and p.name != "m.json"
    ]
    assert leftovers == []


def test_maybe_save_respects_propagation_threshold(tmp_path: Path) -> None:
    mesh = CorticalMesh()
    mesh.add_cell("a")
    persistence = MeshPersistence(tmp_path / "m.json", save_every_n_propagations=10)
    assert persistence.maybe_save(mesh) is False
    mesh._propagation_count = 12
    assert persistence.maybe_save(mesh) is True


def test_default_mesh_path_uses_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    assert default_mesh_path() == tmp_path / "darwin_mesh.json"


def test_malformed_file_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not json at all {")
    mesh = CorticalMesh()
    n = MeshPersistence(path).load_into(mesh)
    assert n == 0
