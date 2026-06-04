"""FeatureFlags — read DARWIN_* environment variables.

Each substrate has an injection seam (mesh.set_propagator, embedding_space.
set_index, etc.). V-Scale only swaps which implementation rides at the
seam; pure-Python remains the reference. Semantics never change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class FeatureFlags:
    """Snapshot of the DARWIN_* flag set."""

    mesh_backend: str = "python"       # "python" | "torch"
    retrieval_backend: str = "python"  # "python" | "faiss"
    vector_backend: str = "python"     # "python" | "numpy" (V-Neural)
    neural_dim: int = 128              # default LearnedCausalSpace dim
    rust_kernel: bool = False
    multiprocess: bool = False

    @classmethod
    def read_env(cls) -> "FeatureFlags":
        try:
            dim = int(os.environ.get("DARWIN_NEURAL_DIM", "128"))
            if dim <= 0:
                dim = 128
        except ValueError:
            dim = 128
        return cls(
            mesh_backend=os.environ.get("DARWIN_MESH_BACKEND", "python").strip().lower(),
            retrieval_backend=os.environ.get(
                "DARWIN_RETRIEVAL_BACKEND", "python",
            ).strip().lower(),
            vector_backend=os.environ.get(
                "DARWIN_VECTOR_BACKEND", "python",
            ).strip().lower(),
            neural_dim=dim,
            rust_kernel=_bool_env("DARWIN_RUST_KERNEL", False),
            multiprocess=_bool_env("DARWIN_MULTIPROCESS", False),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "mesh_backend": self.mesh_backend,
            "retrieval_backend": self.retrieval_backend,
            "vector_backend": self.vector_backend,
            "neural_dim": self.neural_dim,
            "rust_kernel": self.rust_kernel,
            "multiprocess": self.multiprocess,
        }


__all__ = ["FeatureFlags"]
