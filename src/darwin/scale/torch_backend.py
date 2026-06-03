"""TorchMeshPropagator — optional torch-backed mesh activation kernel.

Mirrors the pure-Python CorticalMesh.propagate semantics exactly. The
import is lazy so the module is safe to load even when torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class TorchMeshPropagator:
    """Drop-in replacement for the pure-Python propagator.

    The propagator does not own the mesh; it receives the cell+connection
    state on each call. We materialise to dense tensors when feasible and
    fall back to sparse-style accumulation otherwise.
    """

    decay: float = 0.6
    threshold: float = 0.5

    def available(self) -> bool:
        return torch_available()

    def propagate(
        self,
        cells: dict[str, float],
        connections: list[tuple[str, str, float]],
        *,
        steps: int = 3,
        decay: float | None = None,
    ) -> dict[str, float]:
        """Return updated activations after ``steps`` rounds of propagation."""

        if not self.available():
            raise RuntimeError("torch backend not available; install torch")
        import torch  # local import keeps module-load safe

        decay_factor = float(decay if decay is not None else self.decay)
        names = sorted(cells.keys())
        if not names:
            return dict(cells)
        index = {n: i for i, n in enumerate(names)}
        activations = torch.tensor(
            [cells[n] for n in names], dtype=torch.float32,
        )
        # Sparse-style edge tensor.
        src_idx: list[int] = []
        tgt_idx: list[int] = []
        weights: list[float] = []
        for src, tgt, w in connections:
            if src not in index or tgt not in index:
                continue
            src_idx.append(index[src])
            tgt_idx.append(index[tgt])
            weights.append(float(w))
        if src_idx:
            src_t = torch.tensor(src_idx, dtype=torch.long)
            tgt_t = torch.tensor(tgt_idx, dtype=torch.long)
            wt = torch.tensor(weights, dtype=torch.float32)
        else:
            src_t = tgt_t = torch.empty(0, dtype=torch.long)
            wt = torch.empty(0, dtype=torch.float32)
        for _ in range(steps):
            firing = (activations >= self.threshold).float()
            transmitted = firing[src_t] * wt if src_t.numel() else torch.empty(0)
            new_activations = activations * decay_factor
            if transmitted.numel():
                new_activations.scatter_add_(0, tgt_t, transmitted)
            activations = new_activations
        return {n: float(activations[i].item()) for n, i in index.items()}


__all__ = ["TorchMeshPropagator", "torch_available"]
