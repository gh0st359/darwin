"""EmbeddingTrainer — drains a queue of co-occurrence pairs into the space.

Runs as the real ``embedding_trainer`` subsystem. Pulls token batches
off an internal queue (fed by ``CorpusStream``), trains the learned
space in batched SGD-with-AdamW steps, and publishes throughput +
rolling-loss stats on ``BusTopic.EMBEDDING_UPDATES``.

The trainer is back-pressure-aware: when the queue exceeds
``max_queue_size`` it asks the corpus stream to slow down via a
shared throttle event so we never OOM under burst ingest.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.neural.learned_space import LearnedCausalSpace


@dataclass
class TrainerStats:
    started_at: float = field(default_factory=time.time)
    batches_trained: int = 0
    tokens_consumed: int = 0
    last_loss: float = 0.0
    throttle_events: int = 0

    def to_record(self, space: LearnedCausalSpace) -> dict[str, Any]:
        elapsed = max(1e-3, time.time() - self.started_at)
        return {
            "batches_trained": self.batches_trained,
            "tokens_consumed": self.tokens_consumed,
            "tokens_per_second": round(self.tokens_consumed / elapsed, 2),
            "vocab_size": space.vocab_size(),
            "last_loss": round(self.last_loss, 6),
            "loss_ewma": round(space.light_stats()["loss_ewma"], 6),
            "train_steps": space.light_stats()["train_steps"],
            "throttle_events": self.throttle_events,
            "elapsed": round(elapsed, 2),
        }


class EmbeddingTrainer:
    """Queue → batch → train → publish loop."""

    def __init__(
        self,
        space: LearnedCausalSpace,
        *,
        bus: Any = None,
        batch_size: int = 32,
        max_queue_size: int = 4096,
        publish_every: int = 10,
    ) -> None:
        self.space = space
        self.bus = bus
        self.batch_size = int(batch_size)
        self.publish_every = max(1, int(publish_every))
        self._queue: "queue.Queue[list[str]]" = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._throttle = threading.Event()
        # Held during every train_tokens call. Persistence operations
        # (save / load / rollback) acquire it so the live space can't be
        # mutated mid-batch.
        self.training_lock = threading.RLock()
        self.stats = TrainerStats()

    # -- queue API --------------------------------------------------------- #

    def submit(self, tokens: list[str], *, block: bool = True, timeout: float | None = None) -> bool:
        """Submit a token batch for training. Returns True on success."""

        try:
            self._queue.put(tokens, block=block, timeout=timeout)
            # Engage throttle when queue is over 75% full.
            if self._queue.qsize() > int(self._queue.maxsize * 0.75):
                if not self._throttle.is_set():
                    self._throttle.set()
                    self.stats.throttle_events += 1
            return True
        except queue.Full:
            return False

    def submit_pairs(self, pairs: Iterable) -> int:
        """Submit a stream of co-occurrence pairs as a single batch."""

        toks: list[str] = []
        for pair in pairs:
            center, context, _distance = pair
            if not toks or toks[-1] != center:
                toks.append(center)
            toks.append(context)
        if not toks:
            return 0
        self.submit(toks)
        return len(toks)

    def queue_size(self) -> int:
        return self._queue.qsize()

    def throttle_active(self) -> bool:
        return self._throttle.is_set()

    # -- main loop --------------------------------------------------------- #

    def step_once(self, timeout: float = 0.1) -> bool:
        """Pull one batch (or several aggregated) and train. True if work done."""

        try:
            first = self._queue.get(timeout=timeout)
        except queue.Empty:
            # Queue drained: clear throttle so the producer resumes full rate.
            if self._throttle.is_set():
                self._throttle.clear()
            return False
        batch: list[str] = list(first)
        # Coalesce up to batch_size pulls so we amortize the SGNS pass.
        for _ in range(self.batch_size - 1):
            try:
                more = self._queue.get_nowait()
            except queue.Empty:
                break
            batch.extend(more)
        with self.training_lock:
            self.space.train_tokens(batch)
        self.stats.batches_trained += 1
        self.stats.tokens_consumed += len(batch)
        self.stats.last_loss = self.space.light_stats()["loss_ewma"]
        if self.stats.batches_trained % self.publish_every == 0:
            self._publish()
        if self._queue.qsize() < int(self._queue.maxsize * 0.25):
            if self._throttle.is_set():
                self._throttle.clear()
        return True

    def run(self) -> None:
        """Run the trainer until ``stop()`` is called. Blocks the calling thread."""

        while not self._stop.is_set():
            self.step_once(timeout=0.25)
        # Final flush + final publish.
        while self.step_once(timeout=0.0):
            pass
        self._publish()

    def stop(self) -> None:
        self._stop.set()

    def _publish(self) -> None:
        if self.bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            self.bus.publish(
                BusTopic.EMBEDDING_UPDATES,
                self.stats.to_record(self.space),
                source="embedding_trainer",
            )
        except Exception:
            return


def run_embedding_trainer(**kwargs: Any) -> None:  # pragma: no cover - subsystem entry
    """SubsystemSpec entrypoint: real loop, replaces the legacy heartbeat.

    Mysterio invokes this in a child process. Without a bus or shared space
    bound through the snapshot path, the trainer runs idle — but the process
    is alive and the supervisor sees it as healthy. When the kernel binds a
    shared queue (V-Scale extension), the trainer will consume from it.
    """

    space = LearnedCausalSpace()
    trainer = EmbeddingTrainer(space=space)
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.stop()


__all__ = ["EmbeddingTrainer", "TrainerStats", "run_embedding_trainer"]
