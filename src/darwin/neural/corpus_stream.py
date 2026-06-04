"""CorpusStream — bridge from ingest events to the embedding trainer.

This is the *missing edge* in pre-V-Neural Darwin: the ingest pipeline
extracted facts into the universe and mesh, but the embedding space only
ever saw chat turns. CorpusStream closes that loop. It:

  1. Subscribes to ``BusTopic.FACT_EXTRACTED`` and ``BusTopic.CORPUS_CHUNK``.
  2. Tokenizes the incoming text through :class:`NeuralTokenizer`.
  3. Pushes tokens through :class:`CooccurrenceWindow` to produce skip-gram pairs.
  4. Submits the pairs to an :class:`EmbeddingTrainer`.

If no bus is provided, the stream still works in pure-API mode via
``feed_text`` / ``feed_tokens`` — used by tests and the operator-facing
training CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.neural.cooccurrence import CooccurrenceWindow
from darwin.neural.tokenizer import NeuralTokenizer
from darwin.neural.trainer import EmbeddingTrainer


@dataclass
class CorpusStreamStats:
    chunks_seen: int = 0
    tokens_emitted: int = 0
    pairs_submitted: int = 0
    fact_events_seen: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "chunks_seen": self.chunks_seen,
            "tokens_emitted": self.tokens_emitted,
            "pairs_submitted": self.pairs_submitted,
            "fact_events_seen": self.fact_events_seen,
        }


class CorpusStream:
    """Ingest → tokens → pairs → trainer."""

    def __init__(
        self,
        *,
        trainer: EmbeddingTrainer,
        tokenizer: NeuralTokenizer | None = None,
        window: int = 5,
        bus: Any = None,
    ) -> None:
        self.trainer = trainer
        self.tokenizer = tokenizer or NeuralTokenizer()
        self.window = CooccurrenceWindow(window=window)
        self.bus = bus
        self.stats = CorpusStreamStats()
        self._unsubs: list = []
        if bus is not None:
            self._attach_bus(bus)

    # -- API --------------------------------------------------------------- #

    def feed_text(self, text: str) -> int:
        """Tokenize ``text``, emit pairs through the trainer queue.

        Returns the number of pairs submitted.
        """

        if not text or not text.strip():
            return 0
        tokens = self.tokenizer.tokenize(text)
        return self.feed_tokens(tokens)

    def feed_tokens(self, tokens: list[str]) -> int:
        if not tokens:
            return 0
        self.stats.chunks_seen += 1
        self.stats.tokens_emitted += len(tokens)
        pairs = list(self.window.push_stream(tokens))
        if pairs:
            self.trainer.submit_pairs(pairs)
            self.stats.pairs_submitted += len(pairs)
        return len(pairs)

    def reset_window(self) -> None:
        """Reset the sliding window — call between unrelated documents."""

        self.window.reset()

    # -- bus wiring -------------------------------------------------------- #

    def _attach_bus(self, bus: Any) -> None:
        try:
            from darwin.mysterio.bus import BusTopic
        except Exception:
            return
        self._unsubs.append(bus.subscribe(BusTopic.CORPUS_CHUNK, self._on_corpus_chunk))
        self._unsubs.append(bus.subscribe(BusTopic.FACT_EXTRACTED, self._on_fact_extracted))

    def detach_bus(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                continue
        self._unsubs = []

    def _on_corpus_chunk(self, event) -> None:
        text = event.payload.get("text", "")
        if isinstance(text, str) and text:
            self.feed_text(text)
        else:
            tokens = event.payload.get("tokens")
            if isinstance(tokens, list):
                self.feed_tokens([str(t) for t in tokens])

    def _on_fact_extracted(self, event) -> None:
        self.stats.fact_events_seen += 1
        # Build a small co-occurrence triple per fact so even purely
        # structural ingest contributes to the learned space.
        payload = event.payload or {}
        subj = str(payload.get("subject", "")).strip()
        obj = str(payload.get("object", "")).strip()
        pred = str(payload.get("predicate", "")).strip()
        if not subj or not obj:
            return
        toks = [subj, pred or "related_to", obj]
        # Reset the window so the fact triple doesn't bleed into adjacent text.
        self.window.reset()
        self.feed_tokens(toks)
        self.window.reset()


__all__ = ["CorpusStream", "CorpusStreamStats"]
