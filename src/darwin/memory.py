from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from darwin.concepts import ConceptIndex
from darwin.storage import PersistentStore
from darwin.types import Transition


@dataclass
class EpisodicMemory:
    capacity: int = 10_000
    _events: deque[Transition] = field(default_factory=deque)
    _by_action: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    _by_variable: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    _timestamps: list[float] = field(default_factory=list)

    def append(self, transition: Transition) -> None:
        self._events.append(transition)
        index = len(self._timestamps)
        self._timestamps.append(time.time())
        self._by_action[transition.action].append(index)
        for variable in set(dict(transition.before)) | set(dict(transition.after)):
            self._by_variable[variable].append(index)
        while len(self._events) > self.capacity:
            self._events.popleft()
            self._timestamps.pop(0)
            # Indices in by_action/by_variable will be re-derived lazily.
            self._rebuild_indices()
            break

    def _rebuild_indices(self) -> None:
        self._by_action.clear()
        self._by_variable.clear()
        for index, transition in enumerate(self._events):
            self._by_action[transition.action].append(index)
            for variable in set(dict(transition.before)) | set(dict(transition.after)):
                self._by_variable[variable].append(index)

    def recent(self, limit: int = 20) -> list[Transition]:
        if limit <= 0:
            return []
        return list(self._events)[-limit:]

    def by_action(self, action: str, limit: int = 20) -> list[Transition]:
        events = list(self._events)
        indices = self._by_action.get(action, [])
        return [events[index] for index in indices[-limit:] if index < len(events)]

    def by_variable(self, variable: str, limit: int = 20) -> list[Transition]:
        events = list(self._events)
        indices = self._by_variable.get(variable, [])
        return [events[index] for index in indices[-limit:] if index < len(events)]

    def changed_variable(
        self,
        variable: str,
        limit: int = 20,
        polarity: str = "any",
    ) -> list[Transition]:
        results: list[Transition] = []
        for transition in self.by_variable(variable, limit=limit * 4):
            before_value = dict(transition.before).get(variable)
            after_value = dict(transition.after).get(variable)
            if before_value == after_value:
                continue
            if polarity == "increase":
                if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                    if float(after_value) <= float(before_value):
                        continue
            elif polarity == "decrease":
                if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                    if float(after_value) >= float(before_value):
                        continue
            results.append(transition)
            if len(results) >= limit:
                break
        return results

    def positive_reward(self, limit: int = 20, threshold: float = 0.0) -> list[Transition]:
        results: list[Transition] = []
        for transition in reversed(self._events):
            if float(transition.reward) > threshold:
                results.append(transition)
                if len(results) >= limit:
                    break
        return list(reversed(results))

    def temporal_distance(self, index: int) -> float:
        """Fraction of how recent the indexed event is (1.0 = newest)."""

        if not self._timestamps:
            return 0.0
        total = len(self._timestamps)
        if index < 0 or index >= total:
            return 0.0
        return (index + 1) / total

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
