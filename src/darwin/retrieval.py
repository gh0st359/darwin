from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.semantics import STOPWORDS, SemanticFrame


@dataclass
class RetrievedMemory:
    kind: str
    title: str
    content: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "payload": self.payload,
        }


@dataclass
class RetrievalPacket:
    query: SemanticFrame
    items: list[RetrievedMemory]
    active_goals: dict[str, Any]
    values: dict[str, int]
    unknown_terms: dict[str, int]

    def top(self, limit: int = 5) -> list[RetrievedMemory]:
        return self.items[:limit]

    def to_record(self) -> dict[str, Any]:
        return {
            "query": self.query.to_record(),
            "items": [item.to_record() for item in self.items],
            "active_goals": self.active_goals,
            "values": self.values,
            "unknown_terms": self.unknown_terms,
        }

    def summary(self) -> str:
        if not self.items:
            return "no relevant memories retrieved"
        top = "; ".join(f"{item.kind}:{item.title}({item.score:.2f})" for item in self.items[:5])
        return top


class ContextRetriever:
    """Retrieves Darwin memory relevant to a semantic frame."""

    def retrieve(
        self,
        darwin: Any,
        frame: SemanticFrame,
        recent_events: Iterable[Any] = (),
        limit: int = 12,
    ) -> RetrievalPacket:
        items: list[RetrievedMemory] = []
        query_terms = self._terms(frame.normalized_text)
        query_groundings = {grounding.name for grounding in frame.groundings}

        semantic_frames = darwin.semantic_memory.frames[-800:]
        total = len(semantic_frames)
        for index, memory_frame in enumerate(semantic_frames):
            if memory_frame is frame:
                continue
            if memory_frame.source == "darwin" and frame.topic != "self":
                continue
            if memory_frame.source == "darwin" and self._contains_internal_notation(memory_frame.original_text):
                continue
            score = self._score_frame(
                frame,
                memory_frame,
                query_terms,
                query_groundings,
                recency=(index + 1) / max(1, total),
            )
            if score <= 0:
                continue
            content = self._frame_content(memory_frame)
            if self._contains_internal_notation(content):
                continue
            items.append(
                RetrievedMemory(
                    kind="semantic",
                    title=f"{memory_frame.speech_act}/{memory_frame.topic}",
                    content=content,
                    score=score,
                    payload=memory_frame.to_record(),
                )
            )

        for concept in darwin.memory.concepts.salient(limit=30):
            concept_terms = self._terms(concept.name.replace(":", " ").replace("_", " "))
            overlap = len(query_terms & concept_terms)
            score = 0.15 * overlap + min(0.35, concept.salience / 120.0)
            if concept.kind in {grounding.kind for grounding in frame.groundings}:
                score += 0.05
            if score > 0.18:
                items.append(
                    RetrievedMemory(
                        kind="concept",
                        title=concept.name,
                        content=(
                            f"{concept.kind} concept with support {concept.support} "
                            f"and reward mean {concept.reward_mean:.2f}"
                        ),
                        score=score,
                        payload=concept.to_record(),
                    )
                )

        for belief in darwin.causal_model.beliefs(limit=25):
            belief_terms = self._terms(f"{belief.action} {belief.variable} {belief.effect} {belief.condition}")
            overlap = len(query_terms & belief_terms)
            grounded = 1 if belief.variable in query_groundings or belief.action in query_groundings else 0
            score = 0.2 * overlap + 0.25 * grounded + 0.2 * belief.confidence
            if score > 0.2:
                items.append(
                    RetrievedMemory(
                        kind="causal_belief",
                        title=f"{belief.action}->{belief.variable}",
                        content=(
                            f"if {belief.condition}, {belief.action} changes "
                            f"{belief.variable} as {belief.effect} "
                            f"(confidence {belief.confidence:.2f}, n={belief.samples})"
                        ),
                        score=score,
                        payload={
                            "action": belief.action,
                            "variable": belief.variable,
                            "condition": belief.condition,
                            "effect": belief.effect,
                            "confidence": belief.confidence,
                            "samples": belief.samples,
                        },
                    )
                )

        for event in list(recent_events)[-8:]:
            content = getattr(event, "content", "")
            if frame.topic != "self" and getattr(event, "kind", "") in {"chat", "thought"}:
                continue
            if self._contains_internal_notation(content):
                continue
            event_terms = self._terms(getattr(event, "content", ""))
            score = 0.1 + 0.1 * len(query_terms & event_terms)
            if score > 0.15:
                items.append(
                    RetrievedMemory(
                        kind="runtime_event",
                        title=getattr(event, "kind", "event"),
                        content=content,
                        score=score,
                    )
                )

        items.sort(key=lambda item: item.score, reverse=True)
        return RetrievalPacket(
            query=frame,
            items=items[:limit],
            active_goals=darwin.semantic_memory.active_goals(),
            values=dict(darwin.semantic_memory.values.most_common(12)),
            unknown_terms=dict(darwin.semantic_memory.unknown_terms.most_common(12)),
        )

    def _score_frame(
        self,
        query: SemanticFrame,
        memory: SemanticFrame,
        query_terms: set[str],
        query_groundings: set[str],
        recency: float,
    ) -> float:
        memory_terms = self._terms(memory.normalized_text)
        overlap = len(query_terms & memory_terms)
        grounding_overlap = len(query_groundings & {grounding.name for grounding in memory.groundings})
        score = 0.0
        score += min(0.45, 0.08 * overlap)
        score += min(0.35, 0.14 * grounding_overlap)
        score += 0.2 if memory.topic == query.topic else 0.0
        score += 0.12 if memory.speech_act == query.speech_act else 0.0
        score += 0.12 if memory.source == "user" else 0.02
        score += 0.08 * recency
        score += min(0.12, 0.03 * len(memory.values))
        score += min(0.12, 0.03 * len(memory.goals))
        score += min(0.14, 0.035 * len(memory.propositions))
        return score

    def _frame_content(self, frame: SemanticFrame) -> str:
        pieces: list[str] = []
        if frame.propositions:
            pieces.extend(
                f"{item.subject} {item.relation} {item.object}" for item in frame.propositions[:3]
            )
        if frame.goals:
            pieces.append(
                "goals: " + ", ".join(f"{key}={value!r}" for key, value in frame.goals.items())
            )
        if frame.values:
            pieces.append(
                "values: " + ", ".join(f"{key}={value:.2f}" for key, value in frame.values.items())
            )
        if not pieces:
            pieces.append(frame.original_text)
        return "; ".join(pieces)

    def _terms(self, text: str) -> set[str]:
        raw = [term.lower() for term in text.replace("_", " ").replace(":", " ").split()]
        cleaned = {term.strip(".,;!?()[]{}'\"") for term in raw}
        return {term for term in cleaned if len(term) > 2 and term not in STOPWORDS}

    def _contains_internal_notation(self, text: str) -> bool:
        return any(marker in text for marker in ["act=", "topic=", "intent=", "source=", "confidence="])
