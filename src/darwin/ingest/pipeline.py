"""IngestPipeline — orchestrate source → parser → universe + mesh.

Every fact extracted by ``DocumentIngester`` (or any of its derivatives)
flows through this pipeline: the subject/object pair is registered as
universe concepts if missing; the relation is added; the corresponding
mesh cells are activated so the substrate learns the new associations
via Hebbian + STDP plasticity over the recent firings ring.

Emits BusTopic.FACT_EXTRACTED + INGEST_PROGRESS events with rolling
throughput stats so the brain terminal can watch the universe grow in
real time.

A bounded Bloom filter (cheap implementation: a set + cap) prevents
re-ingesting identical (subject, predicate, object) triples within a
single session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.ingest.document import DocumentIngester, IngestResult
from darwin.ingest.nl_parser import Fact, NLParser


@dataclass
class IngestStats:
    """Rolling throughput statistics for the ingest pipeline."""

    started_at: float = field(default_factory=time.time)
    facts_seen: int = 0
    facts_added: int = 0
    facts_skipped_dup: int = 0
    facts_skipped_invalid: int = 0
    sources_processed: int = 0

    def facts_per_hour(self) -> float:
        elapsed = max(1e-3, time.time() - self.started_at)
        return float(self.facts_added) * 3600.0 / elapsed

    def to_record(self) -> dict[str, Any]:
        return {
            "facts_seen": self.facts_seen,
            "facts_added": self.facts_added,
            "facts_skipped_dup": self.facts_skipped_dup,
            "facts_skipped_invalid": self.facts_skipped_invalid,
            "sources_processed": self.sources_processed,
            "facts_per_hour": round(self.facts_per_hour(), 1),
            "elapsed_seconds": round(time.time() - self.started_at, 2),
        }


class _BoundedSeenSet:
    """A bounded set used as a cheap dedup filter."""

    def __init__(self, capacity: int = 65536) -> None:
        self._items: set[tuple[str, str, str]] = set()
        self._order: list[tuple[str, str, str]] = []
        self.capacity = int(capacity)

    def contains(self, key: tuple[str, str, str]) -> bool:
        return key in self._items

    def add(self, key: tuple[str, str, str]) -> None:
        if key in self._items:
            return
        self._items.add(key)
        self._order.append(key)
        if len(self._order) > self.capacity:
            oldest = self._order.pop(0)
            self._items.discard(oldest)


class IngestPipeline:
    """Source-agnostic orchestrator for fact ingestion."""

    def __init__(
        self,
        *,
        universe: Any = None,
        mesh: Any = None,
        bus: Any = None,
        parser: NLParser | None = None,
        document_ingester: DocumentIngester | None = None,
    ) -> None:
        self.universe = universe
        self.mesh = mesh
        self.bus = bus
        self.parser = parser or NLParser()
        self.document_ingester = document_ingester or DocumentIngester(self.parser)
        self.stats = IngestStats()
        self._seen = _BoundedSeenSet()

    # -- public API -----------------------------------------------------

    def ingest_text(self, text: str, *, source: str = "text") -> IngestStats:
        result = self.document_ingester.ingest_text(text, source=source)
        return self._absorb(result)

    def ingest_html(self, html_text: str, *, source: str = "html") -> IngestStats:
        result = self.document_ingester.ingest_html(html_text, source=source)
        return self._absorb(result)

    def ingest_file(self, path) -> IngestStats:
        result = self.document_ingester.ingest_file(path)
        return self._absorb(result)

    def ingest_facts(self, facts: Iterable[Fact]) -> IngestStats:
        """Directly ingest pre-extracted Facts (used by Wikipedia / Arxiv)."""

        return self._absorb_facts(list(facts), source="external")

    # -- internals ------------------------------------------------------

    def _absorb(self, result: IngestResult) -> IngestStats:
        if result.error:
            self.stats.sources_processed += 1
            self._publish_progress(result.source, result.error)
            return self.stats
        return self._absorb_facts(result.facts, source=result.source)

    def _absorb_facts(self, facts: list[Fact], *, source: str) -> IngestStats:
        for fact in facts:
            self.stats.facts_seen += 1
            key = (fact.subject, fact.predicate, fact.object)
            if not (fact.subject and fact.object) or fact.subject == fact.object:
                self.stats.facts_skipped_invalid += 1
                continue
            if self._seen.contains(key):
                self.stats.facts_skipped_dup += 1
                continue
            self._seen.add(key)
            added = self._add_to_universe(fact)
            if added:
                self.stats.facts_added += 1
                self._activate_in_mesh(fact)
                self._publish_fact(fact, source=source)
        self.stats.sources_processed += 1
        self._publish_progress(source, "")
        return self.stats

    def _add_to_universe(self, fact: Fact) -> bool:
        if self.universe is None:
            return False
        try:
            self.universe.add_concept(fact.subject, domain="ingested")
            self.universe.add_concept(fact.object, domain="ingested")
            self.universe.add_relation(
                fact.subject, fact.object, fact.predicate,
                weight=float(fact.confidence),
                notes=f"ingested from {fact.source_sentence[:80]!r}",
            )
            return True
        except Exception:
            return False

    def _activate_in_mesh(self, fact: Fact) -> None:
        if self.mesh is None:
            return
        try:
            self.mesh.activate([fact.subject, fact.object], magnitude=0.6)
        except Exception:
            return

    def _publish_fact(self, fact: Fact, *, source: str) -> None:
        if self.bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            self.bus.publish(
                BusTopic.FACT_EXTRACTED,
                {**fact.to_record(), "source": source},
                source="ingest_pipeline",
            )
        except Exception:
            return

    def _publish_progress(self, source: str, error: str) -> None:
        if self.bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            self.bus.publish(
                BusTopic.INGEST_PROGRESS,
                {**self.stats.to_record(), "source": source, "error": error},
                source="ingest_pipeline",
            )
        except Exception:
            return


__all__ = ["IngestPipeline", "IngestStats"]
