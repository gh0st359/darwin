from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtStep:
    label: str
    content: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class ThoughtTrace:
    user_text: str
    semantic_summary: str
    steps: list[ThoughtStep] = field(default_factory=list)
    final_mode: str = ""
    final_confidence: float = 0.0

    def add(
        self,
        label: str,
        content: str,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
    ) -> None:
        self.steps.append(
            ThoughtStep(
                label=label,
                content=content,
                confidence=confidence,
                evidence=list(evidence or []),
            )
        )

    def compact(self) -> str:
        if not self.steps:
            return self.semantic_summary
        parts = [f"{step.label}: {step.content}" for step in self.steps[-6:]]
        return " | ".join(parts)

    def to_record(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "semantic_summary": self.semantic_summary,
            "steps": [step.to_record() for step in self.steps],
            "final_mode": self.final_mode,
            "final_confidence": self.final_confidence,
            "compact": self.compact(),
        }

