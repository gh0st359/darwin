"""DEPRECATION SHIM — `CausalEmbeddingSpace` now lives in :mod:`darwin.neural`.

The legacy dim=32 toy SGNS class has been superseded by
:class:`darwin.neural.learned_space.LearnedCausalSpace`, which is API-
compatible (same call surface: ``observe / observe_transition /
train_tokens / embed / nearest / save / load / stats / checkpoint_hash``)
but adds real capacity (configurable dim, AdamW, subsampling, LR decay,
context window, sharded persistence, pluggable numpy backend).

This module re-exports the new class under the old name so every
pre-V-Neural import keeps working without change. Direct use of the old
name will be removed in a future phase; new code should import from
``darwin.neural``.
"""

from __future__ import annotations

from darwin.neural.learned_space import (
    LearnedCausalSpace as CausalEmbeddingSpace,
    cosine,
    tokens_for_transition,
)


__all__ = ["CausalEmbeddingSpace", "cosine", "tokens_for_transition"]
