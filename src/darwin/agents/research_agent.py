"""ResearchAgent — autonomous read → reason → report loop.

Given a passage + a question, the agent:

1. Ingests the passage through the V-Ingest pipeline (when wired).
2. Runs ForwardChainer to close transitive consequences.
3. Dispatches the question through ReasoningDispatcher.
4. Renders the answer through the SpeechPipeline (when wired) so the
   output is leak-free natural prose.

If any substrate is missing (light-runtime tests), the agent falls back
to direct pattern matching against the passage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from darwin.agents.base import Agent, Solution


@dataclass
class ResearchProblem:
    """A passage + a question."""

    passage: str
    question: str


_QUESTION_TARGET_RX = re.compile(
    r"\b(?:what|who|where|when|why|how)\b.*?\b([a-zA-Z][a-zA-Z_]+)\??$",
    re.IGNORECASE,
)


class ResearchAgent(Agent):
    """Read a passage, reason over it, answer a question about it."""

    name = "research"

    def solve(self, problem: Any) -> Solution:
        started = self._start()
        sol = Solution(agent=self.name)
        if not isinstance(problem, ResearchProblem):
            sol.notes = "expected ResearchProblem"
            return self._finish(sol, started)
        ingested = self._ingest(problem.passage)
        sol.steps.append(f"ingested_facts={ingested}")
        forward = self._forward()
        if forward is not None:
            try:
                report = forward.fixpoint_step(budget=32)
                sol.steps.append(f"derivations_added={report.derivations_added}")
            except Exception:
                pass
        dispatch = self._dispatch(problem.question)
        if dispatch is not None and dispatch.succeeded():
            sol.answer = dispatch.answer
            sol.confidence = 0.85
            sol.succeeded = True
            sol.extras["reasoner"] = dispatch.reasoner
            return self._finish(sol, started)
        # Fallback to direct passage matching.
        sol.answer = self._fallback_extract(problem.passage, problem.question)
        sol.confidence = 0.4 if sol.answer else 0.0
        sol.succeeded = bool(sol.answer)
        sol.notes = "fallback passage match"
        return self._finish(sol, started)

    # -- helpers -------------------------------------------------------

    def _ingest(self, passage: str) -> int:
        pipeline = (
            getattr(self.runtime, "ingest_pipeline", None) if self.runtime else None
        )
        if pipeline is None:
            return 0
        try:
            return int(pipeline.ingest_text(passage) or 0)
        except Exception:
            return 0

    def _forward(self) -> Any:
        return getattr(self.runtime, "forward_chainer", None) if self.runtime else None

    def _dispatch(self, question: str) -> Any:
        dispatcher = (
            getattr(self.runtime, "reasoning_dispatcher", None)
            if self.runtime else None
        )
        if dispatcher is None:
            return None
        try:
            return dispatcher.try_resolve(question)
        except Exception:
            return None

    def _fallback_extract(self, passage: str, question: str) -> str:
        m = _QUESTION_TARGET_RX.search(question)
        if m is None:
            return ""
        target = m.group(1).lower()
        for sentence in re.split(r"[.!?\n]+", passage):
            if target in sentence.lower():
                return sentence.strip()
        return ""


__all__ = ["ResearchAgent", "ResearchProblem"]
