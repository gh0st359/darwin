"""EmbeddingTrainer — queue, batching, back-pressure, publish."""

from __future__ import annotations

from darwin.mysterio.bus import BusTopic, CognitionBus
from darwin.neural.cooccurrence import CoocPair
from darwin.neural.learned_space import LearnedCausalSpace
from darwin.neural.trainer import EmbeddingTrainer


def test_submit_and_step_once_trains():
    space = LearnedCausalSpace(dim=8)
    trainer = EmbeddingTrainer(space=space, batch_size=1, publish_every=1)
    assert trainer.submit(["a", "b", "c"]) is True
    assert trainer.step_once(timeout=0.01) is True
    assert space.vocab_size() == 3


def test_step_once_empty_queue_returns_false():
    space = LearnedCausalSpace(dim=8)
    trainer = EmbeddingTrainer(space=space, batch_size=1)
    assert trainer.step_once(timeout=0.01) is False


def test_submit_pairs_compacts_to_token_batch():
    space = LearnedCausalSpace(dim=8)
    trainer = EmbeddingTrainer(space=space, batch_size=2)
    pairs = [
        CoocPair("a", "b", 1),
        CoocPair("a", "c", 2),
    ]
    submitted = trainer.submit_pairs(pairs)
    assert submitted > 0
    trainer.step_once(timeout=0.01)
    assert space.vocab_size() >= 3


def test_throttle_engages_on_full_queue():
    space = LearnedCausalSpace(dim=4)
    trainer = EmbeddingTrainer(
        space=space, batch_size=1, max_queue_size=4, publish_every=1,
    )
    for _ in range(4):
        trainer.submit(["x", "y"], block=False)
    assert trainer.throttle_active() is True
    # Draining clears the throttle.
    while trainer.step_once(timeout=0.0):
        pass
    assert trainer.throttle_active() is False


def test_trainer_publishes_on_bus():
    bus = CognitionBus()
    received: list = []
    bus.subscribe(BusTopic.EMBEDDING_UPDATES, lambda e: received.append(e))
    space = LearnedCausalSpace(dim=4)
    trainer = EmbeddingTrainer(space=space, bus=bus, batch_size=1, publish_every=1)
    trainer.submit(["m", "n"])
    trainer.step_once(timeout=0.01)
    assert len(received) >= 1
    rec = received[-1].payload
    assert "tokens_consumed" in rec and "vocab_size" in rec


def test_trainer_stats_record_includes_throughput():
    space = LearnedCausalSpace(dim=4)
    trainer = EmbeddingTrainer(space=space, batch_size=1, publish_every=1)
    trainer.submit(["p", "q"])
    trainer.step_once(timeout=0.01)
    rec = trainer.stats.to_record(space)
    assert rec["tokens_consumed"] >= 2
    assert rec["batches_trained"] >= 1
    assert rec["vocab_size"] >= 2
