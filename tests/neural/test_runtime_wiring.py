"""runtime.embedding_space is now a LearnedCausalSpace (drop-in).

Also: ingest pipeline emits CORPUS_CHUNK events on _absorb_facts, so
the embedding space actually grows from ingest (not just chat).
"""

from __future__ import annotations

from darwin.ingest.nl_parser import Fact
from darwin.ingest.pipeline import IngestPipeline
from darwin.mysterio.bus import BusTopic, CognitionBus
from darwin.neural.learned_space import LearnedCausalSpace


def test_legacy_import_path_returns_learned_class():
    from darwin.mysterio.embeddings import CausalEmbeddingSpace

    space = CausalEmbeddingSpace()
    assert isinstance(space, LearnedCausalSpace)


def test_ingest_pipeline_publishes_corpus_chunk_event():
    bus = CognitionBus()
    received: list = []
    bus.subscribe(BusTopic.CORPUS_CHUNK, lambda e: received.append(e))
    pipeline = IngestPipeline(bus=bus)
    facts = [
        Fact(
            subject="darwin",
            predicate="is_a",
            object="system",
            source_sentence="Darwin is a system that learns.",
            confidence=0.9,
        ),
        Fact(
            subject="system",
            predicate="has",
            object="brain",
            source_sentence="The system has a brain.",
            confidence=0.9,
        ),
    ]
    pipeline.ingest_facts(facts)
    assert len(received) == 1
    payload = received[0].payload
    assert "text" in payload
    assert payload["sentence_count"] == 2


def test_processes_roster_uses_real_embedding_trainer():
    from darwin.mysterio.processes import DEFAULT_ROSTER

    spec = next(s for s in DEFAULT_ROSTER if s.name == "embedding_trainer")
    assert spec.entrypoint == "darwin.neural.trainer:run_embedding_trainer"
