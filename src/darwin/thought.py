from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtStep:
    label: str
    content: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "payload": self.payload,
            "started_at": self.started_at,
        }


@dataclass
class ThoughtTrace:
    user_text: str
    semantic_summary: str
    steps: list[ThoughtStep] = field(default_factory=list)
    final_mode: str = ""
    final_confidence: float = 0.0
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)

    def add(
        self,
        label: str,
        content: str,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.steps.append(
            ThoughtStep(
                label=label,
                content=content,
                confidence=confidence,
                evidence=list(evidence or []),
                payload=dict(payload or {}),
            )
        )

    def compact(self) -> str:
        if not self.steps:
            return self.semantic_summary
        parts = [f"{step.label}: {step.content}" for step in self.steps[-6:]]
        return " | ".join(parts)

    def duration_ms(self) -> float:
        if not self.steps:
            return 0.0
        return (time.time() - self.started_at) * 1000.0

    def to_record(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "user_text": self.user_text,
            "semantic_summary": self.semantic_summary,
            "steps": [step.to_record() for step in self.steps],
            "final_mode": self.final_mode,
            "final_confidence": self.final_confidence,
            "compact": self.compact(),
            "started_at": self.started_at,
            "duration_ms": self.duration_ms(),
        }
