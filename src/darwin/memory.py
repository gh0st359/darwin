from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from darwin.concepts import ConceptIndex
from darwin.storage import PersistentStore
from darwin.types import Transition


@dataclass
class EpisodicMemory:
    capacity: int = 10_000
    _events: deque[Transition] = field(default_factory=deque)

    def append(self, transition: Transition) -> None:
        self._events.append(transition)
        while len(self._events) > self.capacity:
            self._events.popleft()

    def recent(self, limit: int = 20) -> list[Transition]:
        if limit <= 0:
            return []
        return list(self._events)[-limit:]

    def all(self) -> Iterable[Transition]:
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)


@dataclass
class Memory:
    episodes: EpisodicMemory = field(default_factory=EpisodicMemory)
    concepts: ConceptIndex = field(default_factory=ConceptIndex)
    store: PersistentStore | None = None

    def learn(self, transition: Transition, persist: bool = True) -> None:
        self.episodes.append(transition)
        self.concepts.learn(transition)
        if persist and self.store is not None:
            self.store.record_transition(transition)
            for concept in self.concepts.salient(limit=50):
                self.store.record_concept(concept.to_record())

    def load(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.episodes.append(transition)
            self.concepts.learn(transition)
