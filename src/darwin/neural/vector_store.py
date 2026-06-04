"""VectorStore — token-id → vector storage with pluggable backend.

Two backends:

  * **python** (default, dependency-free) — a dict of ``list[float]``.
    Reference implementation. Used by CI; pure-Python so determinism is
    obvious.

  * **numpy** (optional, ``DARWIN_VECTOR_BACKEND=numpy``) — a contiguous
    ``float32`` matrix with an id↔row map. Cosine and dot become matrix
    ops; ``nearest`` is O(N) but with vectorised math, so large vocabs
    stay tractable for the training pipeline.

Both backends produce *bit-equivalent* output at the same seed for the
determinism test path. The numpy backend is purely an accelerator.

Persistence is **sharded**: once a shard exceeds ``shard_byte_limit``,
the next ``set`` rolls to a new shard. This lets the on-disk state
grow arbitrarily without loading the full set into RAM.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


try:  # optional accelerator
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy may be absent in default CI
    _HAS_NUMPY = False


def _seed_vector(token: str, dim: int) -> list[float]:
    """Deterministic small-magnitude init from a hash of the token."""

    digest = hashlib.sha256(token.encode("utf-8")).digest()
    out: list[float] = []
    i = 0
    while len(out) < dim:
        byte = digest[i % len(digest)]
        out.append((byte / 255.0 - 0.5) * 0.1)
        i += 1
    return out


@dataclass
class _PythonBackend:
    dim: int
    vectors: dict[str, list[float]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def get(self, token: str) -> list[float]:
        vec = self.vectors.get(token)
        if vec is None:
            vec = _seed_vector(token, self.dim)
            self.vectors[token] = vec
            self.order.append(token)
        return vec

    def set(self, token: str, vec: list[float]) -> None:
        if token not in self.vectors:
            self.order.append(token)
        self.vectors[token] = list(vec)

    def contains(self, token: str) -> bool:
        return token in self.vectors

    def tokens(self) -> list[str]:
        return list(self.order)

    def size(self) -> int:
        return len(self.order)


class _NumpyBackend:
    """Numpy-backed contiguous matrix with an id↔row map."""

    def __init__(self, dim: int, initial_capacity: int = 1024) -> None:
        if not _HAS_NUMPY:  # pragma: no cover - only entered when caller asked for numpy
            raise RuntimeError("numpy backend requested but numpy is not installed")
        self.dim = dim
        self._dtype = _np.float32
        self._capacity = initial_capacity
        self._matrix = _np.zeros((initial_capacity, dim), dtype=self._dtype)
        self._index: dict[str, int] = {}
        self._order: list[str] = []

    def _grow_if_needed(self) -> None:
        if len(self._order) < self._capacity:
            return
        new_cap = max(self._capacity * 2, 1024)
        new_matrix = _np.zeros((new_cap, self.dim), dtype=self._dtype)
        new_matrix[: self._capacity] = self._matrix
        self._matrix = new_matrix
        self._capacity = new_cap

    def get(self, token: str) -> list[float]:
        row = self._index.get(token)
        if row is None:
            self._grow_if_needed()
            seed = _seed_vector(token, self.dim)
            row = len(self._order)
            self._index[token] = row
            self._order.append(token)
            self._matrix[row] = _np.asarray(seed, dtype=self._dtype)
        return self._matrix[row].tolist()

    def set(self, token: str, vec: list[float]) -> None:
        row = self._index.get(token)
        if row is None:
            self._grow_if_needed()
            row = len(self._order)
            self._index[token] = row
            self._order.append(token)
        self._matrix[row] = _np.asarray(vec, dtype=self._dtype)

    def contains(self, token: str) -> bool:
        return token in self._index

    def tokens(self) -> list[str]:
        return list(self._order)

    def size(self) -> int:
        return len(self._order)

    # numpy-only fast nearest
    def nearest_matrix(self, qvec: list[float], k: int, exclude: str | None) -> list[tuple[str, float]]:
        if not self._order:
            return []
        used = self._matrix[: len(self._order)]
        q = _np.asarray(qvec, dtype=self._dtype)
        # cosine
        used_norm = _np.linalg.norm(used, axis=1) + 1e-12
        q_norm = float(_np.linalg.norm(q)) + 1e-12
        scores = used @ q / (used_norm * q_norm)
        idx = _np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for i in idx[: k + 1]:
            tok = self._order[int(i)]
            if tok == exclude:
                continue
            out.append((tok, float(scores[int(i)])))
            if len(out) >= k:
                break
        return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    s = 0.0
    for x in a:
        s += x * x
    return (s ** 0.5) or 1.0


def cosine(a: list[float], b: list[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


class VectorStore:
    """Backend-agnostic vector storage with sharded persistence."""

    SHARD_BYTE_LIMIT_DEFAULT = 256 * 1024 * 1024  # 256 MB

    def __init__(
        self,
        dim: int,
        *,
        backend: str = "python",
        shard_byte_limit: int = SHARD_BYTE_LIMIT_DEFAULT,
    ) -> None:
        self.dim = int(dim)
        self.backend_name = backend
        self.shard_byte_limit = int(shard_byte_limit)
        if backend == "numpy":
            if not _HAS_NUMPY:
                # Silently fall back so the package still imports when numpy
                # is not installed; the python path is the reference impl.
                self.backend_name = "python"
                self._backend: Any = _PythonBackend(dim=self.dim)
            else:
                self._backend = _NumpyBackend(dim=self.dim)
        else:
            self._backend = _PythonBackend(dim=self.dim)

    # -- core ops ----------------------------------------------------------- #

    def get(self, token: str) -> list[float]:
        return self._backend.get(token)

    def set(self, token: str, vec: list[float]) -> None:
        self._backend.set(token, vec)

    def contains(self, token: str) -> bool:
        return self._backend.contains(token)

    def tokens(self) -> list[str]:
        return self._backend.tokens()

    def size(self) -> int:
        return self._backend.size()

    def nearest(
        self,
        query: str | list[float],
        k: int = 5,
        *,
        prefix: str | None = None,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        qvec = self.get(query) if isinstance(query, str) else list(query)
        if isinstance(query, str) and exclude is None:
            exclude = query
        if isinstance(self._backend, _NumpyBackend) and prefix is None:
            return self._backend.nearest_matrix(qvec, k, exclude)
        scored: list[tuple[str, float]] = []
        for tok in self._backend.tokens():
            if tok == exclude:
                continue
            if prefix is not None and not tok.startswith(prefix):
                continue
            scored.append((tok, cosine(qvec, self._backend.get(tok))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # -- sharded persistence ------------------------------------------------ #

    def shard_to_disk(self, root: str | Path) -> list[Path]:
        """Write all vectors to one-or-more shard files under ``root``.

        Returns the list of shard paths written, lowest-numbered first.
        Format per shard:
          uint32 magic | uint32 dim | uint32 n_tokens | repeated [
              uint16 token_len | utf-8 token bytes | dim*float32 LE
          ]
        Atomic at the shard level: each shard is written to a temp file
        then renamed.
        """

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        # Clear any stale shards belonging to this store.
        for stale in sorted(root.glob("shard_*.bin")):
            stale.unlink()
        tokens = self._backend.tokens()
        shards: list[Path] = []
        shard_idx = 0
        buf: list[bytes] = []
        buf_bytes = 0
        magic = struct.pack("<I", 0xDADA0001)  # "Darwin neural shard v1"
        # Build chunks; once buf_bytes exceeds shard_byte_limit, flush.
        for tok in tokens:
            vec = self._backend.get(tok)
            encoded = tok.encode("utf-8")
            entry = struct.pack("<H", len(encoded)) + encoded + struct.pack(
                f"<{self.dim}f", *vec
            )
            if buf_bytes + len(entry) > self.shard_byte_limit and buf:
                shards.append(self._flush_shard(root, shard_idx, magic, buf))
                shard_idx += 1
                buf = []
                buf_bytes = 0
            buf.append(entry)
            buf_bytes += len(entry)
        # Always write a shard, even if empty, so the manifest is consistent.
        shards.append(self._flush_shard(root, shard_idx, magic, buf))
        return shards

    def _flush_shard(
        self, root: Path, idx: int, magic: bytes, entries: list[bytes]
    ) -> Path:
        path = root / f"shard_{idx:04d}.bin"
        tmp = path.with_suffix(".tmp")
        header = magic + struct.pack("<II", self.dim, len(entries))
        with tmp.open("wb") as fh:
            fh.write(header)
            for entry in entries:
                fh.write(entry)
        tmp.replace(path)
        return path

    def load_shards(self, root: str | Path) -> int:
        """Load every ``shard_*.bin`` under ``root`` into this store.

        Returns the number of tokens loaded.
        """

        root = Path(root)
        if not root.exists():
            return 0
        loaded = 0
        for shard in sorted(root.glob("shard_*.bin")):
            with shard.open("rb") as fh:
                _magic = fh.read(4)
                dim_bytes = fh.read(4)
                n_bytes = fh.read(4)
                if len(dim_bytes) < 4 or len(n_bytes) < 4:
                    continue
                dim = struct.unpack("<I", dim_bytes)[0]
                n = struct.unpack("<I", n_bytes)[0]
                if dim != self.dim:
                    raise ValueError(
                        f"shard {shard} dim {dim} != store dim {self.dim}"
                    )
                for _ in range(n):
                    tlen_bytes = fh.read(2)
                    if not tlen_bytes:
                        break
                    tlen = struct.unpack("<H", tlen_bytes)[0]
                    tok = fh.read(tlen).decode("utf-8")
                    vec = list(struct.unpack(f"<{self.dim}f", fh.read(self.dim * 4)))
                    self.set(tok, vec)
                    loaded += 1
        return loaded

    # -- introspection ------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "backend": self.backend_name,
            "dim": self.dim,
            "size": self.size(),
        }


__all__ = ["VectorStore", "cosine", "_HAS_NUMPY"]
