"""Question understanding — what is the operator actually asking?

A real thinker hears a question and identifies what *kind* of answer it
wants. "Is X a Y?" calls for a kind-chain. "Why does X cause Y?" calls for
a causal chain. "How does X relate to Y?" calls for a shortest-path
explanation. "What is X?" calls for a definition + neighborhood.

This module classifies the surface form of a question into a structured
``QuestionAnalysis`` the chat path can route through the right inference
operator. The classifier is rule-based and bounded — no machine-learning
black box. It looks for cue patterns plus the grounded concepts already
extracted by the LanguageGrounder.

Question kinds:
  * ``definition``     — "What is X?"
  * ``kind_check``     — "Is X a Y?"
  * ``causal_why``     — "Why does X happen?", "Why does X cause Y?"
  * ``causal_how``     — "How does X cause Y?", "How does X work?"
  * ``relation``       — "How does X relate to Y?", "How are X and Y connected?"
  * ``compare``        — "How is X different from Y?", "Compare X and Y."
  * ``contradiction``  — "Can X and Y both be true?"
  * ``opinion``        — "What do you think about X?", "What's your view on X?"
  * ``greeting``       — surface-level social.
  * ``unknown``        — could not classify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class QuestionAnalysis:
    """Structured understanding of one user utterance."""

    kind: str = "unknown"
    primary_concepts: list[str] = field(default_factory=list)
    secondary_concepts: list[str] = field(default_factory=list)
    is_question: bool = False
    sentiment: str = "neutral"
    cue: str = ""

    def to_record(self) -> dict:
        return {
            "kind": self.kind,
            "primary_concepts": list(self.primary_concepts),
            "secondary_concepts": list(self.secondary_concepts),
            "is_question": self.is_question,
            "sentiment": self.sentiment,
            "cue": self.cue,
        }


# Regex cues. Conservative — false negative > false positive.
_RX = {
    "definition":    re.compile(r"\bwhat (?:is|are)\b|\bdefine\b|\btell me about\b"),
    "kind_check":    re.compile(r"\b(is|are)\b.+\b(a|an|the)\b"),
    "causal_why":    re.compile(r"\bwhy\b"),
    "causal_how":    re.compile(r"\bhow does\b|\bhow do\b|\bhow can\b|\bhow would\b"),
    "relation":      re.compile(r"\brelate(?:d|s)? to\b|\bconnected to\b|\brelationship\b|\bbetween\b"),
    "compare":       re.compile(r"\bcompare\b|\bdifferent\b|\bvs\b|\bversus\b|\bcontrast\b"),
    "contradiction": re.compile(r"\bcontradict\b|\bopposite\b|\bboth (?:be )?true\b"),
    "opinion":       re.compile(r"\bwhat do you think\b|\byour view\b|\byour opinion\b|\bdo you believe\b"),
    "greeting":      re.compile(r"^\s*(?:hi|hello|hey|yo|greetings)\b"),
}


def analyze_question(
    text: str,
    grounded: Iterable[str],
) -> QuestionAnalysis:
    """Return a QuestionAnalysis for one user utterance.

    ``grounded`` is the list of concept names the LanguageGrounder
    extracted, in order. The first two seeds are usually the primary and
    secondary subject of the question.
    """

    norm = (text or "").lower().strip()
    is_q = norm.endswith("?") or any(
        norm.startswith(w) for w in ("what", "is", "are", "why", "how", "can", "do", "does", "compare")
    )
    seeds = list(grounded)
    primary = seeds[:1]
    secondary = seeds[1:3]

    if _RX["greeting"].search(norm):
        return QuestionAnalysis(
            kind="greeting", is_question=False, cue="greeting",
        )
    # Order matters: compare and relation can both match the same text,
    # but compare is more specific.
    for kind in (
        "compare",
        "contradiction",
        "relation",
        "causal_why",
        "causal_how",
        "kind_check",
        "opinion",
        "definition",
    ):
        if _RX[kind].search(norm):
            return QuestionAnalysis(
                kind=kind,
                primary_concepts=primary,
                secondary_concepts=secondary,
                is_question=is_q or kind != "opinion",
                cue=kind,
            )
    if is_q:
        return QuestionAnalysis(
            kind="definition" if primary else "unknown",
            primary_concepts=primary,
            secondary_concepts=secondary,
            is_question=True,
            cue="trailing_question_mark",
        )
    return QuestionAnalysis(
        kind="unknown",
        primary_concepts=primary,
        secondary_concepts=secondary,
        is_question=False,
    )
