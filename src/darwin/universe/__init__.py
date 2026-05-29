"""Darwin's internal universe — concepts, relations, reasoning.

This is the brain's world. Concepts are nodes; relations are edges;
domains are labelled subgraphs. Darwin lives here.

The default universe loads only structural primitives (``thing``, ``change``,
``cause``, ``same``, ``different``, etc.) — the meta-vocabulary needed to
form concepts about anything. Domain knowledge (physics, math, music, ...)
is meant to be *derived* from chat, observation, and reasoning rather than
pre-baked. A demo seed exists for testing only and must be opted into
explicitly via ``demo_seed_universe``.
"""

from darwin.universe.concept_universe import (
    Concept,
    ConceptUniverse,
    Domain,
    RELATION_KINDS,
    Relation,
)
from darwin.universe.derivation import ConceptDeriver, DerivedConcept
from darwin.universe.language_universe import (
    GroundedTerm,
    GroundingResult,
    LanguageGrounder,
    content_words,
    tokenize,
)
from darwin.universe.primitive_seed import primitive_names, seed_primitives
from darwin.universe.reasoning import (
    ConceptualReasoner,
    ReasoningStep,
    ReasoningTrace,
)
from darwin.universe.world import CONCEPTUAL_ACTIONS, ConceptualWorld


def build_default_universe() -> ConceptUniverse:
    """The standard universe Darwin boots with: primitives only.

    Domain knowledge is derived from use, not pre-baked. To opt into a
    rich head-start for demos, see ``darwin.universe.demo_universe``.
    """

    universe = ConceptUniverse()
    seed_primitives(universe)
    return universe


__all__ = [
    "CONCEPTUAL_ACTIONS",
    "Concept",
    "ConceptDeriver",
    "ConceptUniverse",
    "ConceptualReasoner",
    "ConceptualWorld",
    "DerivedConcept",
    "Domain",
    "GroundedTerm",
    "GroundingResult",
    "LanguageGrounder",
    "RELATION_KINDS",
    "ReasoningStep",
    "ReasoningTrace",
    "Relation",
    "build_default_universe",
    "content_words",
    "primitive_names",
    "seed_primitives",
    "tokenize",
]
