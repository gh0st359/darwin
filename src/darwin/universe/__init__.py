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
from darwin.universe.curiosity import CuriosityEngine, CuriosityProbe
from darwin.universe.derivation import ConceptDeriver, DerivedConcept
from darwin.universe.inference import (
    Contradiction,
    Inference,
    InferenceEngine,
)
from darwin.universe.language_universe import (
    GroundedTerm,
    GroundingResult,
    LanguageGrounder,
    content_words,
    tokenize,
)
from darwin.universe.answer import (
    RenderedAnswer,
    build_answer,
    render_chain,
    render_contradiction,
    render_definition,
    render_inference,
    render_reasoning_summary,
)
from darwin.universe.dialogue_memory import DialogueMemory, DialogueTurn
from darwin.universe.fusion import ConceptFusion, FusedRelation, FusionResult
from darwin.universe.active_learning import ActiveLearner, LearningProbe
from darwin.universe.correction import (
    Correction,
    apply_correction,
    detect_correction,
)
from darwin.universe.hypothesis import Hypothesis, HypothesisEngine
from darwin.universe.proactive import VolunteeredRemark, choose_volunteer
from darwin.universe.reflection import (
    Reflection,
    is_reflective_prompt,
    reflect_on_last_reply,
)
from darwin.universe.primitive_seed import primitive_names, seed_primitives
from darwin.universe.question import QuestionAnalysis, analyze_question
from darwin.universe.reasoning import (
    ConceptualReasoner,
    ReasoningStep,
    ReasoningTrace,
)
from darwin.universe.synthesis import (
    SynthesizedAnswer,
    synthesize,
    synthesize_self_introspection,
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
    "Contradiction",
    "CuriosityEngine",
    "CuriosityProbe",
    "DerivedConcept",
    "Domain",
    "GroundedTerm",
    "GroundingResult",
    "Inference",
    "InferenceEngine",
    "LanguageGrounder",
    "RELATION_KINDS",
    "ConceptFusion",
    "DialogueMemory",
    "DialogueTurn",
    "FusedRelation",
    "FusionResult",
    "ActiveLearner",
    "Correction",
    "Hypothesis",
    "HypothesisEngine",
    "LearningProbe",
    "VolunteeredRemark",
    "apply_correction",
    "choose_volunteer",
    "detect_correction",
    "QuestionAnalysis",
    "ReasoningStep",
    "ReasoningTrace",
    "Reflection",
    "is_reflective_prompt",
    "reflect_on_last_reply",
    "Relation",
    "RenderedAnswer",
    "SynthesizedAnswer",
    "analyze_question",
    "build_answer",
    "build_default_universe",
    "content_words",
    "primitive_names",
    "render_chain",
    "render_contradiction",
    "render_definition",
    "render_inference",
    "render_reasoning_summary",
    "seed_primitives",
    "synthesize",
    "synthesize_self_introspection",
    "tokenize",
]
