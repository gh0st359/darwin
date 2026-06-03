"""ReasoningDispatcher — route a query to the right reasoner.

Given a natural-language question + the concepts grounded in it, the
dispatcher chooses between BackwardChainer (kind-check / part-of /
inheritance), ResolutionProver (yes/no), HypotheticalReasoner ("what if
X were Y"), BeliefNetwork (probabilistic queries), and DefeasibleReasoner
(default-with-exception queries). When no reasoner matches confidently,
falls back to the existing v6.5 InferenceEngine.

The dispatcher returns a ``DispatchResult`` carrying a derivation, a
proof tree, or both — whichever the chosen reasoner produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from darwin.reasoning.backward import BackwardChainer, ProofTree
from darwin.reasoning.bayesian import BeliefNetwork
from darwin.reasoning.defeasible import DefeasibleReasoner, DefeasibleVerdict
from darwin.reasoning.forward import ForwardChainer
from darwin.reasoning.hypothetical import HypotheticalReasoner
from darwin.reasoning.resolution import Literal, ResolutionProof, ResolutionProver


@dataclass
class DispatchResult:
    """Outcome of one dispatch call."""

    reasoner: str
    answer: str = ""
    proof_tree: ProofTree | None = None
    verdict: DefeasibleVerdict | None = None
    resolution_proof: ResolutionProof | None = None
    probability: float | None = None
    notes: str = ""

    def succeeded(self) -> bool:
        return any([
            self.proof_tree is not None,
            self.verdict is not None,
            self.resolution_proof is not None,
            self.probability is not None,
        ])

    def to_record(self) -> dict[str, Any]:
        return {
            "reasoner": self.reasoner,
            "answer": self.answer,
            "succeeded": self.succeeded(),
            "notes": self.notes,
            "proof_tree": self.proof_tree.to_record() if self.proof_tree else None,
            "verdict": self.verdict.to_record() if self.verdict else None,
            "resolution_proof": (
                self.resolution_proof.to_record() if self.resolution_proof else None
            ),
            "probability": self.probability,
        }


_KIND_CHECK_RX = re.compile(
    r"\bis\s+(?:a|an|the)\s+([a-z][a-z_]+)\s+(?:a|an|the)\s+([a-z][a-z_]+)\??",
    re.IGNORECASE,
)
_PART_CHECK_RX = re.compile(
    r"\bis\s+(?:a|an|the)\s+([a-z][a-z_]+)\s+(?:part\s+of)\s+(?:a|an|the)\s+([a-z][a-z_]+)\??",
    re.IGNORECASE,
)
_CAUSAL_CHECK_RX = re.compile(
    r"\bdoes\s+(?:a|an|the)?\s*([a-z][a-z_]+)\s+cause\s+(?:a|an|the)?\s*([a-z][a-z_]+)\??",
    re.IGNORECASE,
)
_PROB_RX = re.compile(
    r"\b(?:how likely|what(?:'s| is) the probability) (?:that|of) "
    r"([a-z][a-z_]+)",
    re.IGNORECASE,
)


class ReasoningDispatcher:
    """Compose all reasoners behind a single try_resolve entrypoint."""

    def __init__(
        self,
        *,
        universe: Any,
        forward: ForwardChainer | None = None,
        backward: BackwardChainer | None = None,
        hypothetical: HypotheticalReasoner | None = None,
        bayesian: BeliefNetwork | None = None,
        defeasible: DefeasibleReasoner | None = None,
        resolution: ResolutionProver | None = None,
    ) -> None:
        self.universe = universe
        self.forward = forward or ForwardChainer(universe)
        self.backward = backward or BackwardChainer(universe)
        self.hypothetical = hypothetical or HypotheticalReasoner(universe)
        self.bayesian = bayesian or BeliefNetwork(universe)
        self.defeasible = defeasible or DefeasibleReasoner(universe)
        self.resolution = resolution or ResolutionProver(universe)

    def try_resolve(
        self, message: str, grounded_concepts: list[str] | None = None,
    ) -> DispatchResult | None:
        """Inspect the message; dispatch to the appropriate reasoner."""

        if not message:
            return None
        # 1. Kind check: "is a foo a bar"
        m = _KIND_CHECK_RX.search(message)
        if m:
            source, target = m.group(1).lower(), m.group(2).lower()
            # Try defeasible first.
            verdict = self.defeasible.query(source, "is_a", target)
            if verdict is not None:
                return DispatchResult(
                    reasoner="defeasible",
                    answer=f"{source} is a {target}" if verdict.holds else f"{source} is NOT a {target}",
                    verdict=verdict,
                )
            proof = self.backward.prove(source, target, kind="is_a")
            if proof is not None:
                return DispatchResult(
                    reasoner="backward",
                    answer=f"{source} is a {target}",
                    proof_tree=proof,
                )
            # Resolution as a last resort.
            res_proof = self.resolution.prove(
                Literal(source=source, kind="is_a", target=target)
            )
            if res_proof is not None:
                return DispatchResult(
                    reasoner="resolution",
                    answer=f"{source} is a {target}",
                    resolution_proof=res_proof,
                )
            return DispatchResult(
                reasoner="backward",
                answer=f"no proof that {source} is a {target}",
                notes="no chain found",
            )
        # 2. Part-of check.
        m = _PART_CHECK_RX.search(message)
        if m:
            source, target = m.group(1).lower(), m.group(2).lower()
            proof = self.backward.prove(source, target, kind="part_of")
            if proof is not None:
                return DispatchResult(
                    reasoner="backward",
                    answer=f"{source} is part of {target}",
                    proof_tree=proof,
                )
            return DispatchResult(
                reasoner="backward",
                answer=f"no proof that {source} is part of {target}",
                notes="no chain found",
            )
        # 3. Causal check.
        m = _CAUSAL_CHECK_RX.search(message)
        if m:
            source, target = m.group(1).lower(), m.group(2).lower()
            proof = self.backward.prove(source, target, kind="causes")
            if proof is not None:
                return DispatchResult(
                    reasoner="backward",
                    answer=f"{source} causes {target}",
                    proof_tree=proof,
                )
            return DispatchResult(
                reasoner="backward",
                answer=f"no proof that {source} causes {target}",
                notes="no chain found",
            )
        # 4. Probability query.
        m = _PROB_RX.search(message)
        if m:
            concept = m.group(1).lower()
            self.bayesian.propagate(steps=2)
            posterior = self.bayesian.query(concept)
            return DispatchResult(
                reasoner="bayesian",
                answer=(
                    f"my current belief about {concept} sits at "
                    f"{posterior * 100:.0f}%"
                ),
                probability=posterior,
            )
        return None


__all__ = ["DispatchResult", "ReasoningDispatcher"]
