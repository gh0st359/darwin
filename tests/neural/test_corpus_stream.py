"""CorpusStream — ingest events feed the embedding trainer."""

from __future__ import annotations

from darwin.mysterio.bus import BusTopic, CognitionBus
from darwin.neural.corpus_stream import CorpusStream
from darwin.neural.learned_space import LearnedCausalSpace
from darwin.neural.trainer import EmbeddingTrainer


def test_feed_text_grows_vocab():
    space = LearnedCausalSpace(dim=8)
    trainer = EmbeddingTrainer(space=space, batch_size=1, publish_every=1)
    stream = CorpusStream(trainer=trainer, window=2)
    stream.feed_text("Darwin learns from streamed text reliably.")
    while trainer.step_once(timeout=0.01):
        pass
    assert space.vocab_size() >= 5
    assert stream.stats.chunks_seen == 1
    assert stream.stats.pairs_submitted > 0


def test_feed_empty_text_is_a_noop():
    space = LearnedCausalSpace(dim=4)
    trainer = EmbeddingTrainer(space=space, batch_size=1)
    stream = CorpusStream(trainer=trainer)
    assert stream.feed_text("") == 0
    assert stream.feed_text("   ") == 0
    assert stream.stats.chunks_seen == 0


def test_bus_corpus_chunk_event_drives_training():
    bus = CognitionBus()
    space = LearnedCausalSpace(dim=8)
    trainer = EmbeddingTrainer(space=space, batch_size=1, publish_every=1)
    stream = CorpusStream(trainer=trainer, window=2, bus=bus)
    bus.publish(
        BusTopic.CORPUS_CHUNK,
        {"text": "Mesh and embedding learn together over corpus chunks."},
        source="test",
    )
    while trainer.step_once(timeout=0.01):
        pass
    assert stream.stats.chunks_seen == 1
    assert space.vocab_size() >= 5


def test_bus_fact_extracted_event_drives_training():
    bus = CognitionBus()
    space = LearnedCausalSpace(dim=8)
    trainer = EmbeddingTrainer(space=space, batch_size=1, publish_every=1)
    stream = CorpusStream(trainer=trainer, bus=bus)
    bus.publish(
        BusTopic.FACT_EXTRACTED,
        {"subject": "widget", "predicate": "is_a", "object": "gadget"},
        source="test",
    )
    while trainer.step_once(timeout=0.01):
        pass
    assert stream.stats.fact_events_seen == 1
    assert space.vocab_size() >= 3


def test_detach_bus_stops_consumption():
    bus = CognitionBus()
    space = LearnedCausalSpace(dim=4)
    trainer = EmbeddingTrainer(space=space, batch_size=1)
    stream = CorpusStream(trainer=trainer, bus=bus)
    stream.detach_bus()
    bus.publish(BusTopic.CORPUS_CHUNK, {"text": "nope nope nope"}, source="t")
    assert stream.stats.chunks_seen == 0
