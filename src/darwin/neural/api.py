"""TrainingClient — programmatic operator-facing training API.

What it gives operators (Claude Code / Codex CLI sessions, scripts,
embedded Python integrations):

  * ``feed_corpus(text)`` / ``feed_corpus_iter(iterator)`` — drain a
    text stream into Darwin's learned representation.
  * ``feed_pairs(pairs)`` — direct co-occurrence injection for
    structured / pre-tokenised data.
  * ``probe(token)`` — look up learned neighbours, embedding norm,
    frequency, and last-trained-step diagnostics.
  * ``stats()`` — full training-state snapshot (vocab, loss, throughput).
  * ``checkpoint(label)`` / ``rollback(label)`` / ``list_checkpoints()`` —
    labelled save/restore so the operator can branch training runs.
  * ``flush()`` — wait until the trainer queue drains; useful for tests
    and for scripted "feed N, save, exit" sessions.
  * ``save()`` / ``load()`` — manual persistence triggers.

A TrainingClient binds against a live :class:`DarwinRuntime` (whose
``embedding_space``, ``embedding_trainer``, and ``corpus_stream`` are
the actual substrate). When no runtime is provided, the client
constructs the substrate itself — useful for headless training jobs
that never want a full DarwinRuntime startup cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from darwin.neural.corpus_stream import CorpusStream
from darwin.neural.learned_space import LearnedCausalSpace
from darwin.neural.persistence import NeuralPersistence
from darwin.neural.tokenizer import NeuralTokenizer
from darwin.neural.trainer import EmbeddingTrainer


def _default_root() -> Path:
    from darwin.paths import data_dir

    return data_dir() / "neural"


@dataclass
class TrainingClient:
    """Programmatic operator-facing surface over the V-Neural substrate."""

    runtime: Any = None
    space: LearnedCausalSpace | None = None
    trainer: EmbeddingTrainer | None = None
    stream: CorpusStream | None = None
    tokenizer: NeuralTokenizer | None = None
    persistence: NeuralPersistence | None = None
    root: Path | None = None
    _owns_trainer_thread: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root) if self.root else _default_root()
        self.persistence = self.persistence or NeuralPersistence(self.root)
        self.tokenizer = self.tokenizer or NeuralTokenizer()
        if self.runtime is not None:
            # Bind to a live runtime — the trainer is already running.
            self.space = self.space or self.runtime.embedding_space
            self.trainer = self.trainer or self.runtime.embedding_trainer
            self.stream = self.stream or self.runtime.corpus_stream
        else:
            # Headless mode — own the substrate ourselves.
            import threading as _threading

            self.space = self.space or LearnedCausalSpace()
            self.trainer = self.trainer or EmbeddingTrainer(space=self.space)
            self.stream = self.stream or CorpusStream(
                trainer=self.trainer, tokenizer=self.tokenizer,
            )
            # Drain in background so feed_* never blocks.
            self._trainer_thread = _threading.Thread(
                target=self.trainer.run,
                name="training-client-trainer",
                daemon=True,
            )
            self._trainer_thread.start()
            self._owns_trainer_thread = True

    # -- feed -------------------------------------------------------------- #

    def feed_corpus(self, text: str, *, source: str = "api") -> int:
        """Tokenize and feed ``text`` into the trainer. Returns pair count."""

        if not text:
            return 0
        pairs = self.stream.feed_text(text)
        self._record_source(source, len(text))
        return pairs

    def feed_corpus_iter(self, texts: Iterable[str], *, source: str = "api") -> int:
        """Drain an iterable of text chunks into the trainer."""

        total = 0
        for chunk in texts:
            total += self.feed_corpus(chunk, source=source)
        return total

    def feed_pairs(self, pairs: Iterable) -> int:
        """Direct co-occurrence injection."""

        if not pairs:
            return 0
        return self.trainer.submit_pairs(pairs)

    # -- diagnostics ------------------------------------------------------- #

    def vocab(self) -> int:
        return self.space.vocab_size()

    def nearest(self, token: str, k: int = 5) -> list[tuple[str, float]]:
        return self.space.nearest(token, k=k)

    def probe(self, token: str) -> dict[str, Any]:
        """Diagnostic snapshot for ``token`` — neighbours, freq, norm."""

        import math

        try:
            vec = self.space.embed(token)
        except Exception:
            return {"token": token, "in_vocab": False}
        norm = math.sqrt(sum(x * x for x in vec))
        return {
            "token": token,
            "in_vocab": self.space._store.contains(token),
            "frequency": int(self.space._freq.get(token, 0)),
            "vector_norm": round(norm, 6),
            "nearest": [
                {"token": t, "score": round(s, 6)}
                for t, s in self.space.nearest(token, k=8)
            ],
        }

    def stats(self) -> dict[str, Any]:
        base = self.space.stats()
        base.update({
            "queue_size": self.trainer.queue_size(),
            "throttle_active": self.trainer.throttle_active(),
            "stream_chunks_seen": self.stream.stats.chunks_seen,
            "stream_pairs_submitted": self.stream.stats.pairs_submitted,
            "stream_fact_events_seen": self.stream.stats.fact_events_seen,
            "checkpoints": self.persistence.list_checkpoints(),
        })
        return base

    # -- persistence ------------------------------------------------------- #

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until the trainer queue is empty or ``timeout`` elapses."""

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.trainer.queue_size() == 0:
                return True
            time.sleep(0.05)
        return self.trainer.queue_size() == 0

    def save(self, label: str = "") -> dict[str, Any]:
        self.flush()
        with self.trainer.training_lock:
            manifest = self.persistence.save(
                space=self.space, tokenizer=self.tokenizer, label=label,
            )
        return manifest.to_record()

    def load(self) -> dict[str, Any] | None:
        with self.trainer.training_lock:
            manifest = self.persistence.load(
                space=self.space, tokenizer=self.tokenizer,
            )
        return manifest.to_record() if manifest else None

    def checkpoint(self, label: str) -> dict[str, Any]:
        self.flush()
        with self.trainer.training_lock:
            # Save current active set, then copy it under the label.
            self.persistence.save(
                space=self.space, tokenizer=self.tokenizer, label=label,
            )
            path = self.persistence.checkpoint(label)
        return {"label": label, "path": str(path)}

    def list_checkpoints(self) -> list[str]:
        return self.persistence.list_checkpoints()

    def rollback(self, label: str) -> dict[str, Any]:
        self.flush()
        with self.trainer.training_lock:
            self.persistence.rollback(label)
            # Reload into the live space.
            self.persistence.load(space=self.space, tokenizer=self.tokenizer)
        return {"label": label, "vocab_size": self.space.vocab_size()}

    # -- cursor (resumable ingest) ---------------------------------------- #

    def cursor(self) -> dict[str, Any]:
        return self.persistence.read_cursor()

    def write_cursor(self, payload: dict[str, Any]) -> None:
        self.persistence.write_cursor(payload)

    def _record_source(self, source: str, byte_delta: int) -> None:
        """Update the resumable cursor for ``source`` (additive)."""

        cur = self.persistence.read_cursor()
        per_source = dict(cur.get("sources", {}))
        entry = dict(per_source.get(source, {}))
        entry["bytes"] = int(entry.get("bytes", 0)) + int(byte_delta)
        entry["last_fed_at"] = time.time()
        per_source[source] = entry
        cur["sources"] = per_source
        self.persistence.write_cursor(cur)


__all__ = ["TrainingClient"]
