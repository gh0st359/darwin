"""Tests for the optional torch mesh propagator."""

from __future__ import annotations

import pytest

from darwin.scale.torch_backend import TorchMeshPropagator, torch_available


def test_propagator_reports_availability() -> None:
    p = TorchMeshPropagator()
    assert p.available() is torch_available()


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_torch_propagator_runs() -> None:
    p = TorchMeshPropagator(decay=0.5, threshold=0.5)
    cells = {"a": 1.0, "b": 0.0, "c": 0.0}
    connections = [("a", "b", 1.0), ("b", "c", 1.0)]
    result = p.propagate(cells, connections, steps=3, decay=0.5)
    # After propagation, c should have received some activation.
    assert result["c"] > 0.0


@pytest.mark.skipif(torch_available(), reason="torch available")
def test_torch_propagator_raises_without_torch() -> None:
    p = TorchMeshPropagator()
    with pytest.raises(RuntimeError):
        p.propagate({"a": 1.0}, [], steps=1)
