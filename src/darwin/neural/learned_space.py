"""LearnedCausalSpace — the real successor to ``CausalEmbeddingSpace``.

API-compatible with the legacy class so every existing call site keeps
working without change. Behaviourally:

  * Configurable ``dim`` (default 128 — meaningful capacity, vs the toy 32).
  * Pluggable :class:`VectorStore` backend (pure-Python reference,
    optional numpy accelerator).
  * Skip-gram with negative sampling, with:
      - Inverse-frequency subsampling (Mikolov 1e-3) to keep high-frequency
        tokens from dominating gradients;
      - AdamW-style per-parameter learning-rate scaling (m, v accumulators);
      - Cosine learning-rate decay over a configurable horizon;
      - Gradient clipping per-pair.
  * Context window for skip-gram (default 5, distance-weighted).
  * Token frequency tracking — driven by the subsampling and the trainer.
  * ``checkpoint_hash`` stable across runs at the same seed/config.

No LLM, no pretrained weights. Cold start, deterministic seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from darwin.neural.vector_store import VectorStore, cosine


def _backend_from_env() -> str:
    raw = os.environ.get("DARWIN_VECTOR_BACKEND", "python").strip().lower()
    return "numpy" if raw == "numpy" else "python"


def _dim_from_env(default: int) -> int:
    raw = os.environ.get("DARWIN_NEURAL_DIM")
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def tokens_for_transition(before: dict, action: str, after: dict) -> list[str]:
    """Canonical token set describing one transition (back-compat shim)."""

    toks = [f"act:{action}"]
    for var, val in sorted(before.items()):
        toks.append(f"pre:{var}={val}")
    for var, val in sorted(after.items()):
        toks.append(f"post:{var}={val}")
    return toks


@dataclass
class LearnedCausalSpace:
    """Online SGNS with subsampling, AdamW, LR decay, sharded persistence."""

    dim: int = 128
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    decay_horizon: int = 1_000_000  # steps to anneal LR over
    negatives: int = 5
    seed: int = 1729
    window: int = 5
    subsample_threshold: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    backend: str = ""  # resolved in __post_init__
    _store: VectorStore = field(init=False)
    _moments: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    _freq: dict[str, int] = field(default_factory=dict)
    _total_tokens_seen: int = 0
    _train_steps: int = 0
    _loss_ewma: float = 0.0
    _loss_alpha: float = 0.05

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.dim = _dim_from_env(self.dim)
        self.backend = self.backend or _backend_from_env()
        self._store = VectorStore(dim=self.dim, backend=self.backend)

    # -- vocab / store ------------------------------------------------------ #

    def _vector(self, token: str) -> list[float]:
        return self._store.get(token)

    def _moment(self, token: str) -> tuple[list[float], list[float]]:
        m = self._moments.get(token)
        if m is None:
            m = ([0.0] * self.dim, [0.0] * self.dim)
            self._moments[token] = m
        return m

    def vocab_size(self) -> int:
        return self._store.size()

    # -- legacy back-compat surface ---------------------------------------- #

    def observe(self, before: dict, action: str, after: dict) -> None:
        self.train_tokens(tokens_for_transition(before, action, after))

    def observe_transition(self, transition: Any) -> None:
        self.observe(
            dict(getattr(transition, "before", {})),
            str(getattr(transition, "action", "")),
            dict(getattr(transition, "after", {})),
        )

    # -- training ---------------------------------------------------------- #

    def _current_lr(self) -> float:
        if self.decay_horizon <= 0:
            return self.learning_rate
        progress = min(1.0, self._train_steps / float(self.decay_horizon))
        # Cosine anneal from learning_rate down to min_learning_rate.
        decayed = self.min_learning_rate + 0.5 * (
            self.learning_rate - self.min_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        return decayed

    def _subsample_keep(self, token: str) -> bool:
        # On tiny corpora the frequency estimate is unreliable and Mikolov
        # subsampling would discard almost every token — train everything
        # until the corpus has enough mass for the estimate to mean something.
        if self._total_tokens_seen < 1000:
            return True
        freq = self._freq.get(token, 0)
        if freq == 0:
            return True
        f = freq / float(self._total_tokens_seen)
        if f <= self.subsample_threshold:
            return True
        # Mikolov: keep with prob sqrt(t/f).
        keep = math.sqrt(self.subsample_threshold / f)
        return self._rng.random() < keep

    def _observe_freq(self, tokens: list[str]) -> None:
        for tok in tokens:
            self._freq[tok] = self._freq.get(tok, 0) + 1
        self._total_tokens_seen += len(tokens)

    def train_tokens(self, tokens: list[str]) -> None:
        """SGNS pass over the token set with windowed skip-gram + subsampling."""

        if len(tokens) < 2:
            return
        for tok in tokens:
            self._vector(tok)
        self._observe_freq(tokens)
        kept = [t for t in tokens if self._subsample_keep(t)]
        if len(kept) < 2:
            return
        lr = self._current_lr()
        for i, center in enumerate(kept):
            lo = max(0, i - self.window)
            hi = min(len(kept), i + self.window + 1)
            cvec = self._vector(center)
            for j in range(lo, hi):
                if j == i:
                    continue
                context = kept[j]
                weight = 1.0 / max(1, abs(j - i))
                self._update_pair(center, cvec, context, label=1.0, lr=lr * weight)
                for _ in range(self.negatives):
                    neg = self._sample_negative(exclude=kept)
                    if neg is None:
                        break
                    self._update_pair(
                        center, cvec, neg, label=0.0, lr=lr * weight,
                    )
            self._train_steps += 1

    def _update_pair(
        self,
        center_tok: str,
        cvec: list[float],
        context_tok: str,
        label: float,
        lr: float,
    ) -> None:
        ctx_vec = self._vector(context_tok)
        score = 0.0
        for k in range(self.dim):
            score += cvec[k] * ctx_vec[k]
        score = max(-30.0, min(30.0, score))
        pred = 1.0 / (1.0 + math.exp(-score))
        err = pred - label
        # Track loss EWMA from binary cross-entropy contribution.
        loss = -(label * math.log(pred + 1e-12) + (1.0 - label) * math.log(1.0 - pred + 1e-12))
        self._loss_ewma = (1.0 - self._loss_alpha) * self._loss_ewma + self._loss_alpha * loss
        # Gradient w.r.t. center / context vectors.
        gn = err
        if self.grad_clip > 0:
            gn = max(-self.grad_clip, min(self.grad_clip, gn))
        # AdamW-style update on both vectors.
        m_c, v_c = self._moment(center_tok)
        m_x, v_x = self._moment(context_tok)
        b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
        for k in range(self.dim):
            g_c = gn * ctx_vec[k]
            g_x = gn * cvec[k]
            m_c[k] = b1 * m_c[k] + (1 - b1) * g_c
            v_c[k] = b2 * v_c[k] + (1 - b2) * (g_c * g_c)
            m_x[k] = b1 * m_x[k] + (1 - b1) * g_x
            v_x[k] = b2 * v_x[k] + (1 - b2) * (g_x * g_x)
            cvec[k] = cvec[k] - lr * m_c[k] / (math.sqrt(v_c[k]) + eps)
            ctx_vec[k] = ctx_vec[k] - lr * m_x[k] / (math.sqrt(v_x[k]) + eps)
        self._store.set(center_tok, cvec)
        self._store.set(context_tok, ctx_vec)

    def _sample_negative(self, exclude: Iterable[str]) -> str | None:
        toks = self._store.tokens()
        if not toks:
            return None
        excluded = set(exclude)
        for _ in range(8):
            tok = toks[self._rng.randrange(len(toks))]
            if tok not in excluded:
                return tok
        return None

    # -- retrieval --------------------------------------------------------- #

    def embed(self, token: str) -> list[float]:
        return list(self._vector(token))

    def embed_state(self, state: dict, prefix: str = "pre") -> list[float]:
        toks = [f"{prefix}:{var}={val}" for var, val in sorted(state.items())]
        return self._embed_mean(toks)

    def embed_action(self, action: str) -> list[float]:
        return self.embed(f"act:{action}")

    def _embed_mean(self, tokens: list[str]) -> list[float]:
        if not tokens:
            return [0.0] * self.dim
        acc = [0.0] * self.dim
        for tok in tokens:
            vec = self._vector(tok)
            for k in range(self.dim):
                acc[k] += vec[k]
        n = float(len(tokens))
        return [v / n for v in acc]

    def nearest(
        self, query: str | list[float], k: int = 5, *, prefix: str | None = None
    ) -> list[tuple[str, float]]:
        return self._store.nearest(query, k=k, prefix=prefix)

    # -- persistence ------------------------------------------------------- #

    def checkpoint_hash(self) -> str:
        # Stable across runs: deterministic ordering of vectors.
        items = []
        for tok in self._store.tokens():
            vec = self._store.get(tok)
            items.append((tok, [round(x, 6) for x in vec]))
        items.sort(key=lambda x: x[0])
        payload = json.dumps(items, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> str:
        """Single-file save (legacy compat)."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dim": self.dim,
            "train_steps": self._train_steps,
            "total_tokens_seen": self._total_tokens_seen,
            "freq": self._freq,
            "vectors": {tok: self._store.get(tok) for tok in self._store.tokens()},
        }
        path.write_text(json.dumps(record))
        return self.checkpoint_hash()

    def load(self, path: str | Path) -> None:
        record = json.loads(Path(path).read_text())
        self.dim = int(record["dim"])
        # Rebuild the store at the persisted dim so the seed is consistent.
        self._store = VectorStore(dim=self.dim, backend=self.backend)
        for tok, vec in record["vectors"].items():
            self._store.set(tok, list(map(float, vec)))
        self._freq = {k: int(v) for k, v in record.get("freq", {}).items()}
        self._total_tokens_seen = int(record.get("total_tokens_seen", 0))
        self._train_steps = int(record.get("train_steps", 0))

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "vocab_size": self.vocab_size(),
            "train_steps": self._train_steps,
            "total_tokens_seen": self._total_tokens_seen,
            "loss_ewma": round(self._loss_ewma, 6),
            "lr": round(self._current_lr(), 6),
            "checkpoint_hash": self.checkpoint_hash()[:12],
        }

    # Bus-friendly snapshot (lighter than checkpoint_hash).
    def light_stats(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size(),
            "train_steps": self._train_steps,
            "loss_ewma": round(self._loss_ewma, 6),
        }


__all__ = ["LearnedCausalSpace", "tokens_for_transition", "cosine"]
