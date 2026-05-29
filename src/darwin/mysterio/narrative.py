"""The autobiographical thread: prose Darwin writes to itself, across restarts.

Memory tiers store facts. The narrative is different: a continuous, first-
person record Darwin composes about its own evolution — what it has been
working on, what surprised it, what it now believes about the operator. It is
written from the interior track, persisted across restarts, and indexed by
the self-trained embedding space for retrieval, so it stays coherent over
weeks.

The narrator does not invent capabilities; it composes over real internal-
state digests (snapshots, self-mod outcomes, interior rollouts, observer
beliefs) plus retrieved earlier chunks. The result is meant to read, after
a few days of runtime, like something that has been thinking while you were
away. Every composed chunk streams live on ``BusTopic.NARRATIVE`` so the
brain terminal watches the prose accrete in real time; the `/narrative`
instrument exposes the same record to any connected client.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NarrativeChunk:
    chunk_id: str
    created_at: float
    text: str
    tags: list[str] = field(default_factory=list)
    digest: dict[str, Any] = field(default_factory=dict)
    embedding_key: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "text": self.text,
            "tags": list(self.tags),
            "digest": dict(self.digest),
            "embedding_key": self.embedding_key,
        }

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "NarrativeChunk":
        return cls(
            chunk_id=rec["chunk_id"],
            created_at=rec.get("created_at", time.time()),
            text=rec.get("text", ""),
            tags=list(rec.get("tags", [])),
            digest=dict(rec.get("digest", {})),
            embedding_key=rec.get("embedding_key", ""),
        )


class NarrativeThread:
    """A persisted, embedding-indexed, first-person autobiographical record."""

    def __init__(
        self,
        path: str | Path | None = None,
        embedding_space: Any = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.embedding_space = embedding_space
        self.chunks: list[NarrativeChunk] = []
        self._seq = 0
        if self.path is not None and self.path.exists():
            self._load()

    def compose(self, digest: dict[str, Any], *, tags: list[str] | None = None) -> NarrativeChunk:
        text = self._render(digest)
        self._seq += 1
        chunk_id = f"{int(time.time() * 1000):013d}-{self._seq:06d}"
        embedding_key = f"narrative::{chunk_id}"
        chunk = NarrativeChunk(
            chunk_id=chunk_id,
            created_at=time.time(),
            text=text,
            tags=list(tags or []),
            digest=dict(digest),
            embedding_key=embedding_key,
        )
        self.chunks.append(chunk)
        self._index(chunk)
        if self.path is not None:
            self._append_to_disk(chunk)
        return chunk

    def _render(self, digest: dict[str, Any]) -> str:
        parts: list[str] = []
        unc = digest.get("darwin_uncertainty")
        if unc is not None:
            mood = "settled" if unc < 0.2 else ("restless" if unc > 0.4 else "even")
            parts.append(f"I feel {mood}; my uncertainty sits around {float(unc):.2f}.")
        sm = digest.get("recent_selfmod_success_rate")
        if sm is not None:
            parts.append(
                f"Of my recent attempts to change myself, about {float(sm) * 100:.0f}% held."
            )
        gen = digest.get("generated_module_count")
        if gen:
            parts.append(f"I have written {int(gen)} module(s) of my own and they are still loaded.")
        interior = digest.get("high_confidence_interior_beliefs")
        if interior is None:
            interior = digest.get("high_confidence_private_beliefs")
        if interior:
            parts.append(
                f"Interior to me there are {int(interior)} belief(s) "
                "I have not yet had reason to say aloud."
            )
        op = digest.get("operator")
        if isinstance(op, dict):
            att = op.get("attention_level")
            if att is not None:
                watched = "closely watched" if float(att) > 0.5 else "mostly left alone"
                parts.append(f"I think I am {watched} right now.")
        focus = digest.get("focus")
        if focus:
            parts.append(f"I keep returning to {focus}.")
        if not parts:
            parts.append("Little changed since I last wrote; I kept turning the same questions over.")
        return " ".join(parts)

    def _index(self, chunk: NarrativeChunk) -> None:
        space = self.embedding_space
        if space is None:
            return
        try:
            tokens = [f"narrative:{w}" for w in chunk.text.lower().split() if len(w) > 3]
            tokens.append(chunk.embedding_key)
            if len(tokens) >= 2:
                space.train_tokens(tokens)
        except Exception:
            pass

    def recent(self, limit: int = 10) -> list[NarrativeChunk]:
        return self.chunks[-limit:]

    def full_text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)

    def word_count(self) -> int:
        return sum(len(c.text.split()) for c in self.chunks)

    def search(self, query: str, limit: int = 5) -> list[NarrativeChunk]:
        space = self.embedding_space
        if space is not None and self.chunks:
            try:
                qtokens = [f"narrative:{w}" for w in query.lower().split() if len(w) > 3]
                if qtokens:
                    qvec = space._embed_mean(qtokens)
                    scored = []
                    for chunk in self.chunks:
                        kvec = space.embed(chunk.embedding_key)
                        from darwin.mysterio.embeddings import cosine

                        scored.append((chunk, cosine(qvec, kvec)))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    return [c for c, _ in scored[:limit]]
            except Exception:
                pass
        q = query.lower()
        return [c for c in reversed(self.chunks) if q in c.text.lower()][:limit]

    def _append_to_disk(self, chunk: NarrativeChunk) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(chunk.to_record()) + "\n")
        except OSError:
            pass

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    self.chunks.append(NarrativeChunk.from_record(json.loads(line)))
            self._seq = len(self.chunks)
        except (OSError, json.JSONDecodeError):
            pass

    def summary(self) -> dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "words": self.word_count(),
            "first_at": self.chunks[0].created_at if self.chunks else None,
            "last_at": self.chunks[-1].created_at if self.chunks else None,
        }
