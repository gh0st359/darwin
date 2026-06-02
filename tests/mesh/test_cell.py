"""Tests for ConceptCell + Connection."""

from __future__ import annotations

from darwin.mesh.cell import ConceptCell, Connection


def test_cell_receive_clamps_to_range() -> None:
    cell = ConceptCell(name="x")
    cell.receive(10.0)
    assert cell.activation <= 5.0
    cell.receive(-100.0)
    assert cell.activation >= 0.0


def test_cell_fires_above_threshold_outside_refractory() -> None:
    cell = ConceptCell(name="x", threshold=0.3, refractory_seconds=0.01)
    cell.receive(1.0)
    assert cell.maybe_fire(now=10.0) is True
    assert cell.fire_count == 1


def test_cell_does_not_fire_below_threshold() -> None:
    cell = ConceptCell(name="x", threshold=0.8)
    cell.receive(0.5)
    assert cell.maybe_fire(now=10.0) is False
    assert cell.fire_count == 0


def test_cell_refractory_prevents_immediate_refire() -> None:
    cell = ConceptCell(name="x", threshold=0.3, refractory_seconds=0.05)
    cell.receive(1.0)
    cell.maybe_fire(now=10.0)
    cell.receive(1.0)
    # 0.01s later — inside the refractory window.
    assert cell.maybe_fire(now=10.01) is False


def test_cell_decay_reduces_activation() -> None:
    cell = ConceptCell(name="x")
    cell.receive(1.0)
    cell.decay(0.5)
    assert cell.activation == 0.5


def test_connection_transmit_scales_signal_by_weight() -> None:
    conn = Connection(source="a", target="b", weight=0.5)
    signal = conn.transmit(2.0, now=10.0)
    assert signal == 1.0
    assert conn.traversal_count == 1


def test_connection_reinforce_clamps_to_unit_range() -> None:
    conn = Connection(source="a", target="b", weight=0.5)
    conn.reinforce(0.7)
    assert conn.weight == pytest_approx(1.0)
    conn.reinforce(-5.0)
    assert conn.weight == pytest_approx(-1.0)


def test_cell_to_record_is_serializable() -> None:
    cell = ConceptCell(name="x", threshold=0.42, salience=1.5)
    record = cell.to_record()
    assert record["name"] == "x"
    assert record["threshold"] == 0.42
    assert record["salience"] == 1.5


# Local helper because we don't want to import pytest's approx for one use.
def pytest_approx(target: float, *, tol: float = 1e-6) -> object:
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return abs(float(other) - float(target)) <= tol

        def __repr__(self) -> str:
            return f"~{target}"
    return _Approx()
