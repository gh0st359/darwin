"""SpeechDLM — implements the DarwinLanguageModule Protocol.

A drop-in replacement for ``StubDLM`` / ``GemmaDLM`` that routes through
the non-LLM ``SpeechPipeline`` instead of a template composer or an
external LLM. Returns a ``DLMRenderResult`` so the existing chat loop
in ``DarwinRuntime._respond`` consumes it without modification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from darwin.dlm import DLMRenderResult
from darwin.speech.pipeline import SpeechPipeline


@dataclass
class SpeechDLM:
    """Adapter that fulfils the DLM Protocol via SpeechPipeline."""

    pipeline: SpeechPipeline
    name: str = "speech"

    def render(self, plan: Any, frame: Any, trace: Any) -> DLMRenderResult:
        started = time.perf_counter()
        user_id = None
        # ThoughtTrace doesn't carry user_id today; if frame has it use that,
        # otherwise leave None and the pipeline uses the anonymous model.
        try:
            user_id = getattr(frame, "user_id", None)
        except Exception:
            user_id = None
        result = self.pipeline.render(plan, user_id=user_id)
        validation_notes: list[str] = []
        if not result.leak_passed:
            validation_notes.append(
                "leak_gate rejected primary output; sanitised fallback used"
            )
            validation_notes.extend(result.leak_reasons[:4])
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        return DLMRenderResult(
            text=result.text,
            renderer=self.name,
            valid=result.valid,
            validation_notes=validation_notes,
            raw_output=result.text,
            duration_ms=duration_ms,
        )


__all__ = ["SpeechDLM"]
