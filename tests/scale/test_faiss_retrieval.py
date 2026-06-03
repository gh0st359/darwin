"""Tests for the optional FAISS vector index."""

from __future__ import annotations

import pytest

from darwin.scale.faiss_retrieval import FAISSVectorIndex, faiss_available


def test_availability_check() -> None:
    assert isinstance(faiss_available(), bool)


@pytest.mark.skipif(not faiss_available(), reason="faiss not installed")
def test_faiss_index_add_and_search() -> None:
    idx = FAISSVectorIndex(dim=4)
    idx.add("a", [1.0, 0.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0, 0.0])
    idx.add("c", [0.0, 0.0, 1.0, 0.0])
    results = idx.search([1.0, 0.0, 0.0, 0.0], k=1)
    assert results[0][0] == "a"
    assert idx.size() == 3


@pytest.mark.skipif(faiss_available(), reason="faiss available")
def test_faiss_index_raises_without_faiss() -> None:
    with pytest.raises(RuntimeError):
        FAISSVectorIndex(dim=4)
