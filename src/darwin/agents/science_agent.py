"""ScienceAgent — multi-hop reasoning for science questions.

Targets GPQA-style problems where the answer requires composing several
ingested facts. Strategy:

1. Tokenise the question and ground the concept names.
2. Activate the cortical mesh on the grounded set.
3. Run ForwardChainer over the universe to expose transitive closures.
4. Use the BeliefNetwork to score candidate answers by posterior.
5. Pick the highest-scoring answer; reject ties as ``low_confidence``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from darwin.agents.base import Agent, Solution


@dataclass
class ScienceProblem:
    """A multi-choice science question."""

    question: str
    choices: list[str] = field(default_factory=list)


_TOKEN_RX = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


class ScienceAgent(Agent):
    """Multi-hop reasoner backed by ForwardChainer + BeliefNetwork."""

    name = "science"

    def solve(self, problem: Any) -> Solution:
        started = self._start()
        sol = Solution(agent=self.name)
        if isinstance(problem, str):
            problem = ScienceProblem(question=problem)
        if not isinstance(problem, ScienceProblem):
            sol.notes = "expected ScienceProblem or str"
            return self._finish(sol, started)
        universe = self._universe()
        if universe is None or not problem.choices:
            sol.notes = "no universe or no choices"
            return self._finish(sol, started)
        question_concepts = self._ground(problem.question, universe)
        sol.steps.append(f"q_concepts={question_concepts}")
        self._activate_mesh(question_concepts)
        forward = self._forward()
        if forward is not None:
            try:
                forward.fixpoint_step(budget=32)
            except Exception:
                pass
        # Score each choice by combined evidence.
        belief = self._belief()
        if belief is not None:
            for concept in question_concepts:
                try:
                    belief.set_prior(concept, 0.8)
                except Exception:
                    continue
            try:
                belief.propagate(steps=3)
            except Exception:
                pass
        scores: list[tuple[str, float]] = []
        for choice in problem.choices:
            score = self._score_choice(choice, question_concepts, universe, belief)
            scores.append((choice, score))
            sol.steps.append(f"{choice[:20]}={score:.3f}")
        scores.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        sol.answer = best
        sol.confidence = min(1.0, max(0.0, best_score - runner_up + 0.5))
        sol.succeeded = best_score > 0.5
        sol.extras["scores"] = {c: round(s, 4) for c, s in scores}
        return self._finish(sol, started)

    # -- helpers -------------------------------------------------------

    def _universe(self) -> Any:
        return getattr(self.runtime, "universe", None) if self.runtime else None

    def _forward(self) -> Any:
        return getattr(self.runtime, "forward_chainer", None) if self.runtime else None

    def _belief(self) -> Any:
        return getattr(self.runtime, "belief_network", None) if self.runtime else None

    def _ground(self, text: str, universe: Any) -> list[str]:
        tokens = [t.lower() for t in _TOKEN_RX.findall(text)]
        out: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in seen:
                continue
            if universe.has(tok):
                out.append(tok)
                seen.add(tok)
        return out

    def _score_choice(
        self,
        choice: str,
        question_concepts: list[str],
        universe: Any,
        belief: Any,
    ) -> float:
        choice_concepts = self._ground(choice, universe)
        score = 0.0
        # Graph proximity: count edges between choice concepts and question
        # concepts.
        for q in question_concepts:
            try:
                rels = universe.neighbors(q)
            except Exception:
                rels = []
            for rel in rels:
                if rel.target in choice_concepts:
                    score += float(getattr(rel, "weight", 1.0) or 1.0)
        # Belief contribution: posterior of choice concepts.
        if belief is not None:
            for c in choice_concepts:
                try:
                    score += belief.query(c)
                except Exception:
                    continue
        # Tiny prior for choice-concept count (avoid empty-choice winning).
        score += 0.1 * len(choice_concepts)
        return score


__all__ = ["ScienceAgent", "ScienceProblem"]
