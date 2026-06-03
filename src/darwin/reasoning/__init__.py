"""V-Reason — six extended inference modes over Darwin's universe.

  * :class:`ForwardChainer` — apply transitive + causal closure rules
    until fixpoint or budget exhaustion.
  * :class:`BackwardChainer` — goal-directed proof search returning a
    structured ProofTree.
  * :class:`HypotheticalReasoner` — copy-on-write universe overlay for
    counterfactuals.
  * :class:`BeliefNetwork` — probabilistic belief propagation over the
    concept graph (noisy-OR aggregation, bounded iterations).
  * :class:`DefeasibleReasoner` — default rules with exceptions.
  * :class:`ResolutionProver` — clause-form ground resolution with
    iterative deepening.
  * :class:`ReasoningDispatcher` — routes natural-language questions to
    the right reasoner.

All six are advisory layers on top of the existing v6.5 InferenceEngine.
Pure-Python, no external dependencies.
"""

from darwin.reasoning.backward import BackwardChainer, ProofStep, ProofTree
from darwin.reasoning.bayesian import BeliefNetwork, BeliefNode, BeliefReport
from darwin.reasoning.defeasible import (
    DefaultRule,
    DefeasibleReasoner,
    DefeasibleVerdict,
    Exception_,
)
from darwin.reasoning.dispatcher import DispatchResult, ReasoningDispatcher
from darwin.reasoning.forward import (
    DerivedFact,
    ForwardChainReport,
    ForwardChainer,
)
from darwin.reasoning.hypothetical import HypotheticalReasoner, HypotheticalResult
from darwin.reasoning.resolution import (
    Clause,
    Literal,
    ResolutionProof,
    ResolutionProver,
)


__all__ = [
    "BackwardChainer",
    "BeliefNetwork",
    "BeliefNode",
    "BeliefReport",
    "Clause",
    "DefaultRule",
    "DefeasibleReasoner",
    "DefeasibleVerdict",
    "DerivedFact",
    "DispatchResult",
    "Exception_",
    "ForwardChainReport",
    "ForwardChainer",
    "HypotheticalReasoner",
    "HypotheticalResult",
    "Literal",
    "ProofStep",
    "ProofTree",
    "ReasoningDispatcher",
    "ResolutionProof",
    "ResolutionProver",
]
