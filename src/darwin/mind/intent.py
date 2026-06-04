"""Intent / MindReply — internal data classes the brain never leaks to chat.

The cognition path produces an :class:`Intent` (what kind of thinking is
needed) and resolves it into a :class:`MindReply` (the prose Darwin will
say). Neither object's metadata is allowed into the rendered chat reply
— that's the SpeechPipeline's job to enforce. The names "intent kind"
and "faculty" are deliberately *internal*: the operator only ever sees
Darwin's voice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentKind(str, Enum):
    """Internal categorisation of the cognitive work a message requires.

    Never rendered into chat. Used only by :class:`Mind` to recruit the
    right internal faculties.
    """

    COMPUTE = "compute"      # numeric / symbolic calculation
    DERIVE = "derive"        # code / programming derivation
    RECALL = "recall"        # query the universe + memory
    PLAN = "plan"            # multi-step planning
    RESEARCH = "research"    # information gathering / synthesis
    SYNTHESIZE = "synthesize"  # multi-faculty composition
    DIALOGUE = "dialogue"    # plain conversational turn
    DECLINE = "decline"      # Mind has nothing to add; fall through


@dataclass
class Intent:
    """What kind of thinking the message requires."""

    kind: IntentKind = IntentKind.DECLINE
    confidence: float = 0.0
    faculties: list[str] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)
    # Optional embedding signature of the message, used by AgenticLoop for
    # nearest-prior lookups.
    embedding: list[float] = field(default_factory=list)

    def is_actionable(self) -> bool:
        return self.kind not in (IntentKind.DECLINE, IntentKind.DIALOGUE)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": round(self.confidence, 4),
            "faculties": list(self.faculties),
            "cue_count": len(self.cues),
        }


@dataclass
class MindReply:
    """The composed prose Darwin will speak, plus internal provenance."""

    text: str
    intent_kind: str
    faculties_used: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    declined: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "text_length": len(self.text),
            "intent_kind": self.intent_kind,
            "faculties_used": list(self.faculties_used),
            "step_count": len(self.steps),
            "confidence": round(self.confidence, 4),
            "declined": self.declined,
        }


__all__ = ["Intent", "IntentKind", "MindReply"]
