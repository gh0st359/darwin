"""Multi-week strategic threads — cognition with a horizon longer than a chat.

A `StrategicThread` is a long-running cognitive workflow with persistent state,
a goal, and reflection cycles that span days or weeks. Multiple concurrent
threads share Darwin via the bus. A thread can fork sub-threads and propose
self-mods that support its own continuation; over time the surviving threads
are those whose self-mods stuck under the live accept gate.

A thread is not a thread of execution. It's a long-lived plan-in-flight that
the kernel rehydrates from the snapshot store at startup, so weeks of
strategic motion survive a restart.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategicReflection:
    at: float
    note: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {"at": self.at, "note": self.note, "metrics": dict(self.metrics)}


@dataclass
class StrategicThread:
    goal: str
    horizon_seconds: float
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    track: str = "public"
    state: dict[str, Any] = field(default_factory=dict)
    reflections: list[StrategicReflection] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    closed: bool = False
    score: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_long_horizon(self) -> bool:
        return self.horizon_seconds >= 86400.0  # one day or more

    def reflect(self, note: str, **metrics: Any) -> StrategicReflection:
        reflection = StrategicReflection(at=time.time(), note=note, metrics=dict(metrics))
        self.reflections.append(reflection)
        if len(self.reflections) > 256:
            self.reflections = self.reflections[-256:]
        self.last_active_at = reflection.at
        return reflection

    def fork(self, goal: str, *, horizon_seconds: float | None = None) -> "StrategicThread":
        child = StrategicThread(
            goal=goal,
            horizon_seconds=horizon_seconds if horizon_seconds is not None else self.horizon_seconds,
            parent_id=self.thread_id,
            track=self.track,
            state=dict(self.state),
        )
        return child

    def close(self, *, reason: str = "") -> None:
        self.closed = True
        if reason:
            self.reflect(f"closed: {reason}", final=True)

    def to_record(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "parent_id": self.parent_id,
            "goal": self.goal,
            "track": self.track,
            "horizon_seconds": self.horizon_seconds,
            "state": dict(self.state),
            "reflections": [r.to_record() for r in self.reflections[-32:]],
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "closed": self.closed,
            "score": round(self.score, 4),
        }


class StrategicThreadManager:
    """Registry of live strategic threads, with selection by score over time."""

    def __init__(self, *, max_open_threads: int = 32) -> None:
        self.max_open_threads = max_open_threads
        self.threads: dict[str, StrategicThread] = {}

    def open(self, goal: str, *, horizon_seconds: float = 86400.0, track: str = "public") -> StrategicThread:
        thread = StrategicThread(goal=goal, horizon_seconds=horizon_seconds, track=track)
        self.threads[thread.thread_id] = thread
        self._evict_if_needed()
        return thread

    def get(self, thread_id: str) -> StrategicThread | None:
        return self.threads.get(thread_id)

    def open_threads(self) -> list[StrategicThread]:
        return [t for t in self.threads.values() if not t.closed]

    def by_track(self, track: str) -> list[StrategicThread]:
        return [t for t in self.threads.values() if t.track == track]

    def score_threads(self) -> None:
        """Score by reflection density × recency; closed threads stay at score 0."""
        now = time.time()
        for t in self.threads.values():
            if t.closed:
                t.score = 0.0
                continue
            density = len(t.reflections) / max(1.0, (now - t.created_at) / 60.0)
            recency_penalty = max(0.0, (now - t.last_active_at) / max(60.0, t.horizon_seconds))
            t.score = max(0.0, density - 0.5 * recency_penalty)

    def _evict_if_needed(self) -> None:
        open_list = self.open_threads()
        if len(open_list) <= self.max_open_threads:
            return
        self.score_threads()
        ranked = sorted(open_list, key=lambda t: t.score)
        for t in ranked[: len(open_list) - self.max_open_threads]:
            t.close(reason="evicted under thread budget")

    def summary(self) -> dict[str, Any]:
        return {
            "open": len(self.open_threads()),
            "total": len(self.threads),
            "long_horizon": sum(1 for t in self.open_threads() if t.is_long_horizon),
            "by_track": {
                track: sum(1 for t in self.threads.values() if t.track == track)
                for track in {t.track for t in self.threads.values()}
            },
        }
