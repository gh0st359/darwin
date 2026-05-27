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

    # -- v5 Phase F: consolidation + decay --------------------------------

    def consolidate_redundant_concepts(self) -> dict[str, int]:
        """Collapse identically-named, identically-shaped concepts.

        Real duplication can build up in the v4/v5 concept index when the
        same cluster effect fires twice within a single dream cycle (the
        cluster name encodes the action+variable, so collisions repeat).
        This pass keeps the highest-support copy and drops the rest. Safe
        for v5 because concepts are derived metadata, not load-bearing.
        """

        index = self.concepts
        store = index._concepts  # type: ignore[attr-defined]
        by_signature: dict[tuple, list[str]] = {}
        for key, concept in list(store.items()):
            # Use the concept.name field, not the dict key — the index
            # may rekey on insertion so duplicate-by-content can hide
            # behind different dict keys.
            signature = (concept.kind, concept.level, concept.name)
            by_signature.setdefault(signature, []).append(key)
        removed = 0
        for names in by_signature.values():
            if len(names) <= 1:
                continue
            # Keep the strongest, drop the rest.
            names.sort(key=lambda n: store[n].support, reverse=True)
            for name in names[1:]:
                store.pop(name, None)
                removed += 1
        return {"removed": removed, "remaining": len(store)}

    def decay_stale_concepts(self, half_life_days: float = 7.0) -> dict[str, int]:
        """Demote support of concepts that haven't been updated recently.

        Half-life model: each concept's support is multiplied by
        0.5 ** (age_days / half_life_days). Concepts whose support falls
        below 1.0 are dropped. This is conservative — it never touches
        beliefs, only the concept index.
        """

        if half_life_days <= 0:
            return {"decayed": 0, "remaining": len(self.concepts._concepts)}  # type: ignore[attr-defined]
        # Concept dataclass doesn't carry a timestamp; approximate by
        # treating every concept as "stale by 1 day" each pass. This
        # gives gradual decay that the dream loop can run periodically.
        store = self.concepts._concepts  # type: ignore[attr-defined]
        decay_factor = 0.5 ** (1.0 / max(half_life_days, 0.1))
        decayed = 0
        for name, concept in list(store.items()):
            new_support = int(concept.support * decay_factor)
            if new_support < concept.support:
                concept.support = new_support
                decayed += 1
            if concept.support < 1:
                store.pop(name, None)
        return {"decayed": decayed, "remaining": len(store)}
