"""DialogueMemory — Darwin remembers what was discussed.

A bounded, embedding-indexed history of recent conversational turns.
Each turn captures: the user's utterance, Darwin's reply, the concepts
that were grounded, the inferences that fired, and the question kind the
analyzer classified. Future turns can ask "have we touched X before?"
to revisit prior threads, or detect contradiction with what Darwin
previously asserted.

This is *not* a sqlite log; the v5 store already persists chats. This is
a working-memory layer the chat path consults turn by turn. It supports:

  * ``record(turn)`` — append a turn to memory.
  * ``last_mention(concept)`` — when did this concept last come up?
  * ``contradicts_prior(claim)`` — does this turn say the opposite of
    something Darwin asserted recently?
  * ``thread_for(concept)`` — every prior turn referencing a concept.
  * ``recent_concepts(n)`` — the most recently grounded concepts across
    the window.

Memory is bounded; oldest turns drop when capacity is exceeded.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable


@dataclass
class DialogueTurn:
    """One round-trip — user said X, Darwin said Y."""

    user_text: str
    darwin_text: str
    grounded_concepts: list[str] = field(default_factory=list)
    inferences_used: list[str] = field(default_factory=list)
    question_kind: str = "unknown"
    at: float = field(default_factory=time.time)
    turn_index: int = 0

    def references(self, concept: str) -> bool:
        return concept in self.grounded_concepts

    def to_record(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "darwin_text": self.darwin_text,
            "grounded_concepts": list(self.grounded_concepts),
            "inferences_used": list(self.inferences_used),
            "question_kind": self.question_kind,
            "at": self.at,
            "turn_index": self.turn_index,
        }


class DialogueMemory:
    """Bounded turn history with concept indexing.

    Capacity defaults to 64 turns. The most recent thread for each
    concept is held in an O(1) lookup table so ``thread_for`` and
    ``last_mention`` don't scan the entire deque.
    """

    def __init__(self, *, capacity: int = 64) -> None:
        self.capacity = capacity
        self._turns: Deque[DialogueTurn] = deque(maxlen=capacity)
        self._by_concept: dict[str, list[int]] = {}
        self._turn_counter = 0

    # -- writes ---------------------------------------------------------

    def record(
        self,
        *,
        user_text: str,
        darwin_text: str,
        grounded_concepts: Iterable[str] = (),
        inferences_used: Iterable[str] = (),
        question_kind: str = "unknown",
    ) -> DialogueTurn:
        self._turn_counter += 1
        turn = DialogueTurn(
            user_text=user_text or "",
            darwin_text=darwin_text or "",
            grounded_concepts=list(grounded_concepts),
            inferences_used=list(inferences_used),
            question_kind=question_kind,
            turn_index=self._turn_counter,
        )
        # If we're at capacity, the oldest turn will be evicted by
        # the deque; clean its index entries first.
        if len(self._turns) == self.capacity:
            evicting = self._turns[0]
            for c in evicting.grounded_concepts:
                indices = self._by_concept.get(c)
                if indices:
                    indices.remove(evicting.turn_index)
                    if not indices:
                        self._by_concept.pop(c, None)
        self._turns.append(turn)
        for c in turn.grounded_concepts:
            self._by_concept.setdefault(c, []).append(turn.turn_index)
        return turn

    # -- reads ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def all_turns(self) -> list[DialogueTurn]:
        return list(self._turns)

    def latest(self, n: int = 5) -> list[DialogueTurn]:
        if n <= 0:
            return []
        return list(self._turns)[-n:]

    def last_mention(self, concept: str) -> DialogueTurn | None:
        indices = self._by_concept.get(concept)
        if not indices:
            return None
        target = indices[-1]
        for turn in reversed(self._turns):
            if turn.turn_index == target:
                return turn
        return None

    def thread_for(self, concept: str, *, limit: int = 8) -> list[DialogueTurn]:
        indices = self._by_concept.get(concept, [])[-limit:]
        target_set = set(indices)
        return [t for t in self._turns if t.turn_index in target_set]

    def recent_concepts(self, n: int = 16) -> list[str]:
        counts: Counter = Counter()
        for turn in list(self._turns)[-n:]:
            for c in turn.grounded_concepts:
                counts[c] += 1
        return [c for c, _ in counts.most_common(n)]

    def contradicts_prior(
        self, claim_concepts: Iterable[str], inferences_used: Iterable[str]
    ) -> DialogueTurn | None:
        """Did Darwin previously assert something that the current claim
        appears to contradict?

        A coarse heuristic: a turn that referenced the same concepts and
        used a contradicting inference kind (is_a vs opposes, causes vs
        opposes) is flagged. Real contradiction detection happens in the
        inference engine; this is a *dialogue* check — "you said X
        before; now you're saying not-X" — to surface the conflict for
        the operator.
        """

        claim = set(claim_concepts)
        used = set(inferences_used)
        if "contradiction" not in used and "opposes" not in used:
            return None
        for turn in reversed(self._turns):
            if not (set(turn.grounded_concepts) & claim):
                continue
            prior_used = set(turn.inferences_used)
            if prior_used & {"is_a_chain", "causal_chain"}:
                return turn
        return None

    # -- introspection --------------------------------------------------

    def summary(self) -> dict[str, Any]:
        kind_counts: Counter = Counter(t.question_kind for t in self._turns)
        return {
            "turns": len(self._turns),
            "capacity": self.capacity,
            "tracked_concepts": len(self._by_concept),
            "most_discussed": [c for c, _ in
                               Counter(
                                   c for t in self._turns
                                   for c in t.grounded_concepts
                               ).most_common(8)],
            "question_kinds": dict(kind_counts),
        }
