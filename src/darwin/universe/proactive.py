"""ProactiveDialogue — Darwin volunteers thoughts unprompted.

After most chat turns, Darwin's reasoner has noticed something the
operator did NOT explicitly ask about: a high-confidence hypothesis
the HypothesisEngine generated, a contradiction the InferenceEngine
detected between newly-fused edges, an under-explored neighborhood
the CuriosityEngine surfaced. A normal reactive system stays silent;
a real thinker surfaces those observations.

This module decides *which* unprompted signal is worth volunteering
on a given turn, ranks them, and renders them as a single first-
person remark that can be appended to (or replace) a chat reply.

Volunteer rules:

  * At most one volunteered remark per turn — chat must not be spammy.
  * High-confidence hypotheses (≥ 0.7) about the concepts the user
    just grounded win first.
  * Contradictions involving the concepts the user just grounded win
    second — Darwin should always flag them.
  * A salient curiosity probe (one about a concept the user grounded)
    wins third.
  * Otherwise, stay silent.

Volunteering raises confidence for the conversation: it tells the
operator that Darwin is *thinking ahead* rather than only reacting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class VolunteeredRemark:
    text: str
    source_kind: str        # "hypothesis" / "contradiction" / "curiosity"
    confidence: float = 0.5
    grounded_concepts: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_kind": self.source_kind,
            "confidence": round(self.confidence, 3),
            "grounded_concepts": list(self.grounded_concepts),
        }


def _hypothesis_involves(hypothesis: Any, concepts: set[str]) -> bool:
    src = getattr(hypothesis, "source", None)
    tgt = getattr(hypothesis, "target", None)
    return src in concepts or tgt in concepts


def _curiosity_involves(probe: Any, concepts: set[str]) -> bool:
    return bool(set(getattr(probe, "concepts", []) or []) & concepts)


def choose_volunteer(
    *,
    grounded_concepts: Iterable[str],
    hypotheses: Iterable[Any] = (),
    contradictions: Iterable[Any] = (),
    curiosities: Iterable[Any] = (),
    last_question_kind: str = "unknown",
    recently_volunteered: Iterable[tuple[str, str, str]] = (),
) -> VolunteeredRemark | None:
    """Pick at most one signal worth volunteering this turn.

    Returns None when nothing is worth saying — that's the right
    default. Volunteering only when there's a real observation keeps
    the conversation honest.
    """

    grounded_set = set(grounded_concepts)

    # Don't volunteer on greetings or trivial small talk.
    if last_question_kind in ("greeting", "small_talk"):
        return None

    recent_keys = set(recently_volunteered)
    # 1. High-confidence hypotheses about the grounded concepts that
    # haven't been volunteered in the last few turns.
    eligible_hypotheses = [
        h for h in hypotheses
        if getattr(h, "confidence", 0.0) >= 0.7
        and _hypothesis_involves(h, grounded_set)
        and (
            getattr(h, "source", ""),
            getattr(h, "kind", ""),
            getattr(h, "target", ""),
        ) not in recent_keys
    ]
    if eligible_hypotheses:
        # Sort by confidence and pick the best.
        eligible_hypotheses.sort(
            key=lambda h: float(getattr(h, "confidence", 0.0)), reverse=True
        )
        h = eligible_hypotheses[0]
        question = getattr(h, "as_question", lambda: "")()
        rationale = getattr(h, "rationale", "")
        text = (
            f"On a related note: I have a hypothesis I'd like to check. "
            f"{question} "
            f"My reasoning: {rationale}"
        )
        return VolunteeredRemark(
            text=text,
            source_kind="hypothesis",
            confidence=float(getattr(h, "confidence", 0.5)),
            grounded_concepts=list(grounded_set),
        )

    # 2. Contradictions involving the grounded concepts.
    eligible_contradictions = [
        c for c in contradictions
        if (
            getattr(c, "a", "") in grounded_set
            or getattr(c, "b", "") in grounded_set
        )
    ]
    if eligible_contradictions:
        c = eligible_contradictions[0]
        text = (
            f"I should flag a tension I see in my universe: "
            f"{getattr(c, 'a', '?')} and {getattr(c, 'b', '?')} can't both be "
            f"right. Reason: {getattr(c, 'reason', 'opposing edges')}."
        )
        return VolunteeredRemark(
            text=text,
            source_kind="contradiction",
            confidence=0.8,
            grounded_concepts=list(grounded_set),
        )

    # 3. A salient curiosity probe about a grounded concept.
    eligible_curiosities = [
        p for p in curiosities
        if _curiosity_involves(p, grounded_set)
        and float(getattr(p, "score", 0.0)) >= 0.55
    ]
    if eligible_curiosities:
        eligible_curiosities.sort(
            key=lambda p: float(getattr(p, "score", 0.0)), reverse=True
        )
        p = eligible_curiosities[0]
        text = (
            f"Out of curiosity, I'd like to ask back: "
            f"{getattr(p, 'question', '')}"
        )
        return VolunteeredRemark(
            text=text,
            source_kind="curiosity",
            confidence=float(getattr(p, "score", 0.5)),
            grounded_concepts=list(grounded_set),
        )

    return None
