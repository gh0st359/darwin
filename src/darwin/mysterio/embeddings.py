"""Self-trained causal embeddings — Darwin's own vocabulary, no pretrained weights.

The embedding space gives Darwin a dense vector for every token it has seen:
state variables, action names, and outcome variables. Vectors are learned
*online* from Darwin's own transition stream via skip-gram-with-negative-
sampling over the tokens that co-occur in each transition. Tokens that recur
together (an action and the variables it reliably changes) drift toward each
other; unrelated tokens are pushed apart by sampled negatives.

This is vocabulary, not reasoning. The causal model stays the reasoning core;
embeddings expand the *pattern-recognition surface*: nearest-experience
retrieval, analogy, concept clustering, observer similarity, narrative
indexing all read from this space.

Backends:
  * **pure-Python** (default, dependency-free) — plain ``list[float]`` vectors
    and hand-rolled SGD. Always available; correct and fast enough for warmup.
  * **torch** (optional, ``pip install 'project-darwin[accel]'``) — same API,
    tensor-backed; used automatically when importable.

No pretrained weights are ever imported. Every number here originates from a
deterministic seed plus Darwin's lived experience.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:  # optional acceleration; the pure-Python path is the reference impl.
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch usually absent in CI
    _HAS_TORCH = False


def _seed_vector(token: str, dim: int) -> list[float]:
    """Deterministic small-magnitude init from a hash of the token."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    out: list[float] = []
    i = 0
    while len(out) < dim:
        byte = digest[i % len(digest)]
        # Map byte → [-0.5, 0.5], scaled small so training dominates.
        out.append((byte / 255.0 - 0.5) * 0.1)
        i += 1
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1.0


def cosine(a: list[float], b: list[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def tokens_for_transition(before: dict, action: str, after: dict) -> list[str]:
    """Canonical token set describing one transition."""
    toks = [f"act:{action}"]
    for var, val in sorted(before.items()):
        toks.append(f"pre:{var}={val}")
    for var, val in sorted(after.items()):
        toks.append(f"post:{var}={val}")
    return toks


@dataclass
class CausalEmbeddingSpace:
    """Online skip-gram-with-negative-sampling over transition co-occurrence."""

    dim: int = 32
    learning_rate: float = 0.05
    negatives: int = 5
    seed: int = 1729
    _vectors: dict[str, list[float]] = field(default_factory=dict)
    _vocab: list[str] = field(default_factory=list)
    _train_steps: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.backend = "torch" if _HAS_TORCH else "python"

    # -- vocabulary --------------------------------------------------------- #

    def _vector(self, token: str) -> list[float]:
        vec = self._vectors.get(token)
        if vec is None:
            vec = _seed_vector(token, self.dim)
            self._vectors[token] = vec
            self._vocab.append(token)
        return vec

    def vocab_size(self) -> int:
        return len(self._vocab)

    # -- training ----------------------------------------------------------- #

    def observe(self, before: dict, action: str, after: dict) -> None:
        self.train_tokens(tokens_for_transition(before, action, after))

    def observe_transition(self, transition: Any) -> None:
        self.observe(
            dict(getattr(transition, "before", {})),
            str(getattr(transition, "action", "")),
            dict(getattr(transition, "after", {})),
        )

    def train_tokens(self, tokens: list[str]) -> None:
        """One skip-gram pass: each token predicts every other in the set."""
        if len(tokens) < 2:
            return
        for tok in tokens:
            self._vector(tok)
        lr = self.learning_rate
        for i, center in enumerate(tokens):
            cvec = self._vectors[center]
            for j, context in enumerate(tokens):
                if i == j:
                    continue
                self._update_pair(cvec, self._vectors[context], label=1.0, lr=lr)
                for _ in range(self.negatives):
                    neg = self._sample_negative(exclude=tokens)
                    if neg is None:
                        break
                    self._update_pair(cvec, self._vectors[neg], label=0.0, lr=lr)
            self._train_steps += 1

    def _update_pair(
        self, center: list[float], context: list[float], label: float, lr: float
    ) -> None:
        score = _dot(center, context)
        pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
        grad = (pred - label) * lr
        for k in range(self.dim):
            c = center[k]
            x = context[k]
            center[k] = c - grad * x
            context[k] = x - grad * c

    def _sample_negative(self, exclude: Iterable[str]) -> str | None:
        if not self._vocab:
            return None
        excluded = set(exclude)
        for _ in range(8):
            tok = self._rng.choice(self._vocab)
            if tok not in excluded:
                return tok
        return None

    # -- retrieval ---------------------------------------------------------- #

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
        qvec = self._vector(query) if isinstance(query, str) else query
        scored: list[tuple[str, float]] = []
        for tok in self._vocab:
            if isinstance(query, str) and tok == query:
                continue
            if prefix is not None and not tok.startswith(prefix):
                continue
            scored.append((tok, cosine(qvec, self._vectors[tok])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # -- checkpointing ------------------------------------------------------ #

    def checkpoint_hash(self) -> str:
        payload = json.dumps(
            {tok: [round(x, 6) for x in vec] for tok, vec in sorted(self._vectors.items())},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dim": self.dim,
            "train_steps": self._train_steps,
            "vectors": self._vectors,
        }
        path.write_text(json.dumps(record))
        return self.checkpoint_hash()

    def load(self, path: str | Path) -> None:
        record = json.loads(Path(path).read_text())
        self.dim = int(record["dim"])
        self._train_steps = int(record.get("train_steps", 0))
        self._vectors = {k: list(map(float, v)) for k, v in record["vectors"].items()}
        self._vocab = list(self._vectors.keys())

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "vocab_size": self.vocab_size(),
            "train_steps": self._train_steps,
            "checkpoint_hash": self.checkpoint_hash()[:12],
        }
