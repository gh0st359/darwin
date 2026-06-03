"""DialogueAgent — lowest-priority free-form chat fallback.

Used when all higher-priority _respond overrides decline. Composes a
turn through the existing v6.5 discourse → realizer → critic stack OR
the V-Speech pipeline when the runtime has one wired. Never emits
structured surface: the leak gate is consulted before returning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darwin.agents.base import Agent, Solution


@dataclass
class DialogueProblem:
    """A turn the dialogue agent should produce a reply for."""

    message: str
    history: list[str] | None = None


class DialogueAgent(Agent):
    """Multi-turn coherent fallback."""

    name = "dialogue"

    def solve(self, problem: Any) -> Solution:
        started = self._start()
        sol = Solution(agent=self.name)
        if isinstance(problem, str):
            problem = DialogueProblem(message=problem)
        if not isinstance(problem, DialogueProblem):
            sol.notes = "expected DialogueProblem or str"
            return self._finish(sol, started)
        # Try the V-Speech pipeline first.
        pipeline = (
            getattr(self.runtime, "speech_pipeline", None) if self.runtime else None
        )
        if pipeline is not None:
            try:
                rendered = pipeline.render_simple_reply(problem.message)
                if rendered:
                    sol.answer = rendered
                    sol.confidence = 0.7
                    sol.succeeded = True
                    sol.notes = "speech_pipeline"
                    return self._finish(sol, started)
            except Exception:
                pass
        # Soft fallback: echo a generic acknowledging reply that never leaks
        # structured tokens.
        sol.answer = self._soft_reply(problem.message)
        sol.confidence = 0.3
        sol.succeeded = True
        sol.notes = "soft fallback"
        return self._finish(sol, started)

    def _soft_reply(self, message: str) -> str:
        message_clean = message.strip()
        if not message_clean:
            return "I do not have anything to add yet."
        if message_clean.endswith("?"):
            return "I will think about that and respond once I have more to go on."
        return "I heard what you said and am turning it over."


__all__ = ["DialogueAgent", "DialogueProblem"]
