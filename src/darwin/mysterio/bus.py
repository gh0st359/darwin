"""The cognition bus: cross-process publish/subscribe for Darwin's subsystems.

In single-thread Darwin, the background loops shared one process and one set
of objects. Mysterio splits cognition across OS processes, so the loops need a
transport. `CognitionBus` is that transport: subsystems publish events on
named `BusTopic`s and subscribe to the topics they care about.

Design:

  * **In-process fast path** — when subsystems share a process (or in tests),
    publishing appends to per-topic ``deque`` ring buffers and notifies
    in-process subscriber callbacks synchronously. No serialization.

  * **Cross-process path** — a `multiprocessing.Queue` fan-out. Each event is a
    plain dict (picklable). A background drain thread in every process moves
    events off its inbound queue into the local ring buffers and fires local
    callbacks. This keeps the API identical whether a subscriber is in-process
    or in a child process.

The bus is intentionally lossy under pressure: ring buffers have a bounded
``maxlen`` and the oldest events are dropped. Cognition is a stream, not a
transaction log — the persistent ledger is `storage.py`, not the bus. A
sampled subset of bus events is mirrored to the ledger by the kernel.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque


class BusTopic(str, Enum):
    TRANSITIONS = "transitions"
    PROPOSALS = "proposals"
    SIMULATIONS = "simulations"
    INTERIOR_SIMULATIONS = "interior_simulations"
    NARRATIVE = "narrative"
    OBSERVER_EVENTS = "observer_events"
    RESEARCH_FINDINGS = "research_findings"
    GATE_HISTORY = "gate_history"
    EMBEDDING_UPDATES = "embedding_updates"
    CODE_GEN = "code_gen"
    DIVERGENCE_REPORTS = "divergence_reports"
    META_PROPOSALS = "meta_proposals"
    SUBSYSTEM_HEALTH = "subsystem_health"
    MESH_FIRING = "mesh_firing"
    MESH_PLASTICITY = "mesh_plasticity"
    INGEST_PROGRESS = "ingest_progress"
    FACT_EXTRACTED = "fact_extracted"
    REASONING_STEP = "reasoning_step"
    PROOF_FOUND = "proof_found"
    DEFEAT_FIRED = "defeat_fired"


@dataclass
class BusEvent:
    topic: str
    payload: dict[str, Any]
    priority: int = 0
    source: str = ""
    seq: int = 0
    created_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "priority": self.priority,
            "source": self.source,
            "seq": self.seq,
            "created_at": self.created_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BusEvent":
        return cls(
            topic=record["topic"],
            payload=record.get("payload", {}),
            priority=record.get("priority", 0),
            source=record.get("source", ""),
            seq=record.get("seq", 0),
            created_at=record.get("created_at", time.time()),
        )


Subscriber = Callable[[BusEvent], None]


def _topic_value(topic: BusTopic | str) -> str:
    return topic.value if isinstance(topic, BusTopic) else str(topic)


class CognitionBus:
    """Topic-routed pub/sub with bounded ring buffers per topic.

    Thread-safe for the in-process path. The cross-process fan-out is opt-in
    via :meth:`attach_inbound` / :meth:`attach_outbound`, so the bus works
    standalone in tests and single-process runs without any multiprocessing
    machinery spun up.
    """

    def __init__(self, ring_size: int = 4096) -> None:
        self.ring_size = ring_size
        self._buffers: dict[str, Deque[BusEvent]] = defaultdict(
            lambda: deque(maxlen=ring_size)
        )
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._lock = threading.RLock()
        self._seq = 0
        self._outbound: Any = None  # multiprocessing.Queue | None
        self._inbound: Any = None
        self._drain_thread: threading.Thread | None = None
        self._draining = False
        self._published_count = 0
        self._dropped_estimate = 0

    # -- publish / subscribe ------------------------------------------------- #

    def publish(
        self,
        topic: BusTopic | str,
        payload: dict[str, Any],
        *,
        priority: int = 0,
        source: str = "",
    ) -> BusEvent:
        topic_str = _topic_value(topic)
        with self._lock:
            self._seq += 1
            event = BusEvent(
                topic=topic_str,
                payload=payload,
                priority=priority,
                source=source,
                seq=self._seq,
            )
            buf = self._buffers[topic_str]
            if len(buf) == buf.maxlen:
                self._dropped_estimate += 1
            buf.append(event)
            self._published_count += 1
            local_subs = list(self._subscribers[topic_str])
        # Fire local callbacks outside the lock to avoid re-entrancy deadlock.
        for sub in local_subs:
            try:
                sub(event)
            except Exception:
                # A misbehaving subscriber must not take down the publisher.
                continue
        # Forward to other processes if a fan-out queue is attached.
        if self._outbound is not None:
            try:
                self._outbound.put(event.to_record())
            except Exception:
                pass
        return event

    def subscribe(self, topic: BusTopic | str, callback: Subscriber) -> Callable[[], None]:
        topic_str = _topic_value(topic)
        with self._lock:
            self._subscribers[topic_str].append(callback)

        def unsubscribe() -> None:
            with self._lock:
                subs = self._subscribers.get(topic_str, [])
                if callback in subs:
                    subs.remove(callback)

        return unsubscribe

    def recent(self, topic: BusTopic | str, limit: int = 64) -> list[BusEvent]:
        topic_str = _topic_value(topic)
        with self._lock:
            buf = self._buffers.get(topic_str)
            if not buf:
                return []
            return list(buf)[-limit:]

    def topics(self) -> list[str]:
        with self._lock:
            return [t for t, buf in self._buffers.items() if buf]

    # -- cross-process fan-out ---------------------------------------------- #

    def attach_outbound(self, queue: Any) -> None:
        """Set the queue published events are forwarded to (parent → children)."""
        self._outbound = queue

    def attach_inbound(self, queue: Any) -> None:
        """Set the queue this process drains foreign events from, and start it."""
        self._inbound = queue
        if self._drain_thread is None:
            self._draining = True
            self._drain_thread = threading.Thread(
                target=self._drain_loop, name="cognition-bus-drain", daemon=True
            )
            self._drain_thread.start()

    def _drain_loop(self) -> None:
        while self._draining:
            try:
                record = self._inbound.get(timeout=0.25)
            except Exception:
                continue
            if record is None:  # poison pill
                break
            event = BusEvent.from_record(record)
            with self._lock:
                self._buffers[event.topic].append(event)
                local_subs = list(self._subscribers[event.topic])
            for sub in local_subs:
                try:
                    sub(event)
                except Exception:
                    continue

    def stop(self) -> None:
        self._draining = False
        if self._inbound is not None:
            try:
                self._inbound.put(None)
            except Exception:
                pass
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=1.0)
            self._drain_thread = None

    # -- introspection ------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "published": self._published_count,
                "dropped_estimate": self._dropped_estimate,
                "active_topics": len([t for t, b in self._buffers.items() if b]),
                "subscribers": {t: len(s) for t, s in self._subscribers.items() if s},
            }
