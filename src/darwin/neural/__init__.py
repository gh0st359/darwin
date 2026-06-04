"""V-Neural — the learned causal-embedding substrate.

This package replaces the dim=32 toy embedding space in
``darwin.mysterio.embeddings`` with a real online learner that scales
from chat-turn co-occurrence to operator-driven corpus ingest.

Nothing here imports an LLM. Nothing here loads pretrained weights.
Every number originates from a deterministic seed plus the text the
operator has fed Darwin. The pure-Python path is the reference; the
numpy path is byte-equivalent at default seeds when enabled via
``DARWIN_VECTOR_BACKEND=numpy``.

Public surface mirrors the legacy ``CausalEmbeddingSpace`` so every
existing call site continues to work without change.
"""

from __future__ import annotations

from darwin.neural.cooccurrence import CooccurrenceWindow
from darwin.neural.corpus_stream import CorpusStream
from darwin.neural.learned_space import LearnedCausalSpace
from darwin.neural.persistence import NeuralPersistence
from darwin.neural.tokenizer import NeuralTokenizer
from darwin.neural.trainer import EmbeddingTrainer, run_embedding_trainer
from darwin.neural.vector_store import VectorStore

__all__ = [
    "CooccurrenceWindow",
    "CorpusStream",
    "EmbeddingTrainer",
    "LearnedCausalSpace",
    "NeuralPersistence",
    "NeuralTokenizer",
    "VectorStore",
    "run_embedding_trainer",
]
