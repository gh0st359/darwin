"""ResolutionProver — clause-form theorem prover with iterative deepening.

Operates over a small set of typed Horn-clause-style rules over the
universe's edges. Given a goal clause, the prover tries to derive
contradiction-with-negation via resolution; if it succeeds, the goal
is proved.

This is a compact implementation. Real first-order resolution requires
unification over free variables; here we restrict to ground clauses
(specific concept names) so the search space stays tractable. Iterative
deepening prevents unbounded depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Literal:
    """A signed atomic claim: 'X kind Y' or 'NOT X kind Y'."""

    source: str
    kind: str
    target: str
    negated: bool = False

    def negate(self) -> "Literal":
        return Literal(self.source, self.kind, self.target, not self.negated)

    def __str__(self) -> str:
        sign = "¬" if self.negated else ""
        return f"{sign}{self.source}-[{self.kind}]->{self.target}"


@dataclass
class Clause:
    """A disjunction of literals. Empty clause = ⊥ (contradiction)."""

    literals: tuple[Literal, ...] = ()

    def is_empty(self) -> bool:
        return len(self.literals) == 0

    def __str__(self) -> str:
        return " ∨ ".join(str(lit) for lit in self.literals) or "⊥"


@dataclass
class ResolutionProof:
    """A successful proof: the steps + the derived empty clause."""

    goal: Literal
    steps: list[tuple[Clause, Clause, Clause]] = field(default_factory=list)
    depth: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "goal": str(self.goal),
            "depth": self.depth,
            "step_count": len(self.steps),
        }


class ResolutionProver:
    """Iteratively-deepening ground-resolution prover."""

    def __init__(self, universe: Any, *, max_depth: int = 5) -> None:
        self.universe = universe
        self.max_depth = int(max_depth)

    def prove(self, goal: Literal) -> ResolutionProof | None:
        """Try to prove ``goal`` from the universe's edges."""

        # Build the clause base from the universe's edges (ground positive
        # literals) plus the negation of the goal (as a unit clause).
        positives = self._universe_clauses()
        negation = Clause(literals=(goal.negate(),))
        clauses = list(positives) + [negation]
        for depth in range(1, self.max_depth + 1):
            proof = self._resolve_to_depth(clauses, goal, depth)
            if proof is not None:
                proof.depth = depth
                return proof
        return None

    # -- internals -----------------------------------------------------

    def _universe_clauses(self) -> list[Clause]:
        clauses: list[Clause] = []
        if self.universe is None:
            return clauses
        try:
            for rel in self.universe.relations():
                clauses.append(Clause(literals=(
                    Literal(source=rel.source, kind=rel.kind, target=rel.target),
                )))
        except Exception:
            return clauses
        return clauses

    def _resolve_to_depth(
        self, clauses: list[Clause], goal: Literal, depth: int,
    ) -> ResolutionProof | None:
        """One iterative-deepening pass."""

        new_clauses = list(clauses)
        seen = {tuple(c.literals) for c in clauses}
        proof = ResolutionProof(goal=goal)
        for _ in range(depth):
            generated: list[tuple[Clause, Clause, Clause]] = []
            for i, c1 in enumerate(new_clauses):
                for c2 in new_clauses[i + 1:]:
                    resolved = self._resolve(c1, c2)
                    if resolved is None:
                        continue
                    if tuple(resolved.literals) in seen:
                        continue
                    seen.add(tuple(resolved.literals))
                    generated.append((c1, c2, resolved))
                    if resolved.is_empty():
                        proof.steps.extend(generated)
                        return proof
            if not generated:
                return None
            for _, _, r in generated:
                new_clauses.append(r)
            proof.steps.extend(generated)
        return None

    def _resolve(self, c1: Clause, c2: Clause) -> Clause | None:
        """Return the resolvent if c1 and c2 share complementary literals."""

        for lit1 in c1.literals:
            for lit2 in c2.literals:
                if (
                    lit1.source == lit2.source
                    and lit1.kind == lit2.kind
                    and lit1.target == lit2.target
                    and lit1.negated != lit2.negated
                ):
                    remaining = tuple(
                        l for l in c1.literals if l is not lit1
                    ) + tuple(
                        l for l in c2.literals if l is not lit2
                    )
                    return Clause(literals=remaining)
        return None


__all__ = ["Clause", "Literal", "ResolutionProof", "ResolutionProver"]
