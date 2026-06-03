"""FAISSVectorIndex — optional FAISS-backed nearest-neighbour index.

When ``faiss`` is importable, this index backs ``CausalEmbeddingSpace``
lookups. Otherwise it raises on construction so callers can fall back to
the pure-Python cosine sweep.
"""

from __future__ import annotations

from typing import Any


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


class FAISSVectorIndex:
    """A flat L2 FAISS index that mirrors the pure-Python lookup surface."""

    def __init__(self, dim: int) -> None:
        if not faiss_available():
            raise RuntimeError("faiss not importable; install faiss-cpu")
        import faiss
        import numpy as np
        self.dim = int(dim)
        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexFlatL2(self.dim)
        self._labels: list[str] = []

    def add(self, label: str, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"vector dim {len(vector)} does not match index dim {self.dim}"
            )
        arr = self._np.array([vector], dtype="float32")
        self._index.add(arr)
        self._labels.append(label)

    def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        if not self._labels:
            return []
        arr = self._np.array([vector], dtype="float32")
        d, idx = self._index.search(arr, min(k, len(self._labels)))
        out: list[tuple[str, float]] = []
        for distance, raw_i in zip(d[0], idx[0]):
            if raw_i < 0 or raw_i >= len(self._labels):
                continue
            out.append((self._labels[int(raw_i)], float(distance)))
        return out

    def size(self) -> int:
        return len(self._labels)


__all__ = ["FAISSVectorIndex", "faiss_available"]
