"""ActiveLearner — Darwin asks the operator for the facts it needs.

When Darwin can't derive an answer because a specific edge or concept is
missing, a passive system shrugs and says "I don't know." A frontier
learner identifies *the precise gap* and asks the operator to fill it.

This module inspects the universe + last reasoning trace + last question
and constructs concrete sub-questions ("Does X cause Y?", "Is Y a kind
of Z?") whose answers, if given, would unblock the original question.

Strategies:

  1. **Missing-link probe** — when the user asked "Does X cause Z?" and
     Darwin has no X→Z chain, look for X→Y and Y→Z patterns one hop
     away. If only one of the two halves exists, ask for the other.

  2. **Bridge probe** — for "Is X a Z?" with no chain: walk both
     descendants from X and ancestors from Z and propose a meeting in
     the middle.

  3. **Definition probe** — for any concept with an empty definition
     that the user just grounded, ask the operator to define it.

  4. **Cross-domain probe** — for two grounded concepts in different
     domains with no bridge, ask how they're related.

Every sub-question is structured (source, target, kind it expects) so
the runtime can route follow-up replies through fusion automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningProbe:
    """A sub-question Darwin would like answered to fill a gap."""

    question: str
    source: str = ""
    target: str = ""
    expected_kind: str = ""    # the relation kind a positive answer would create
    rationale: str = ""
    score: float = 0.5

    def to_record(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "source": self.source,
            "target": self.target,
            "expected_kind": self.expected_kind,
            "rationale": self.rationale,
            "score": round(self.score, 3),
        }


class ActiveLearner:
    """Construct targeted sub-questions to fill graph gaps."""

    def __init__(
        self,
        universe,
        *,
        max_probes: int = 4,
    ) -> None:
        self.universe = universe
        self.max_probes = max_probes
        self._asked: set[tuple[str, str, str]] = set()

    def probe(
        self,
        *,
        question_kind: str,
        grounded_concepts: list[str],
        inferences: list[Any],
    ) -> list[LearningProbe]:
        if not grounded_concepts:
            return []
        # If Darwin already has a confident inference, no need to learn.
        if inferences and any(
            getattr(inf, "confidence", 0.0) >= 0.7 for inf in inferences
        ):
            return []
        probes: list[LearningProbe] = []
        seeds = grounded_concepts[:4]
        if len(seeds) >= 2 and question_kind in ("kind_check", "causal_why", "causal_how", "relation", "compare"):
            probes.extend(self._missing_link_probes(seeds[0], seeds[1], question_kind))
        # Definition probes for any grounded concept with no definition.
        for name in seeds:
            c = self.universe.get(name)
            if c is None or not c.definition:
                key = (name, "define", "")
                if key in self._asked:
                    continue
                self._asked.add(key)
                probes.append(
                    LearningProbe(
                        question=f"How would you define {name!r}?",
                        source=name,
                        target="",
                        expected_kind="definition",
                        rationale=f"I have no definition for {name}.",
                        score=0.5,
                    )
                )
        # Cross-domain probes if grounded concepts span domains with no
        # bridge.
        if len(seeds) >= 2:
            a, b = seeds[0], seeds[1]
            ca, cb = self.universe.get(a), self.universe.get(b)
            if ca is not None and cb is not None and ca.domain != cb.domain:
                already = any(
                    rel.target == b for rel in self.universe.neighbors(a)
                ) or any(
                    rel.target == a for rel in self.universe.neighbors(b)
                )
                if not already:
                    key = (a, "cross", b)
                    if key not in self._asked:
                        self._asked.add(key)
                        probes.append(
                            LearningProbe(
                                question=(
                                    f"How does {a} (in {ca.domain}) "
                                    f"relate to {b} (in {cb.domain})?"
                                ),
                                source=a,
                                target=b,
                                expected_kind="related_to",
                                rationale=(
                                    f"{a} and {b} live in different domains "
                                    f"and aren't connected yet."
                                ),
                                score=0.6,
                            )
                        )
        probes.sort(key=lambda p: p.score, reverse=True)
        return probes[: self.max_probes]

    def _missing_link_probes(
        self, a: str, b: str, question_kind: str,
    ) -> list[LearningProbe]:
        """If A→B is asked but the chain is broken, hunt for the missing link."""

        if not (self.universe.has(a) and self.universe.has(b)):
            return []
        # Forward search from A: every node A reaches via is_a / causes.
        forward: dict[str, str] = {}
        for rel in self.universe.neighbors(a):
            forward[rel.target] = rel.kind
        # Backward search from B: every node that points at B.
        backward: dict[str, str] = {}
        for rel in self.universe.neighbors(b, include_incoming=True):
            if rel.target == b:
                backward[rel.source] = rel.kind
        # A meeting in the middle would close the chain. If forward has X
        # but backward doesn't, ask whether X→B holds.
        out: list[LearningProbe] = []
        kind_for_question = (
            "is_a" if question_kind == "kind_check"
            else "causes" if question_kind in ("causal_why", "causal_how")
            else "related_to"
        )
        kind_verb = {
            "is_a": "a kind of",
            "causes": "cause",
            "related_to": "related to",
        }.get(kind_for_question, "related to")
        for x in forward.keys():
            if x == a or x == b:
                continue
            key = (x, kind_for_question, b)
            if key in self._asked:
                continue
            self._asked.add(key)
            if kind_for_question == "is_a":
                question = f"Is {x} a kind of {b}?"
            elif kind_for_question == "causes":
                question = f"Does {x} cause {b}?"
            else:
                question = f"Is {x} related to {b}?"
            out.append(
                LearningProbe(
                    question=question,
                    source=x,
                    target=b,
                    expected_kind=kind_for_question,
                    rationale=(
                        f"You're asking about {a} and {b}; I know "
                        f"{a} {forward[x].replace('_', ' ')} {x}, "
                        f"but I don't know if {x} {kind_verb} {b}. "
                        f"That edge would let me answer."
                    ),
                    score=0.7,
                )
            )
            if len(out) >= 3:
                break
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "asked": len(self._asked),
        }
