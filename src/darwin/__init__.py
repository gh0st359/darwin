"""Project Darwin: a causal-adaptive AI kernel."""

from darwin.agent import Darwin
from darwin.causal_chain import CausalChain, CausalChainEngine, CausalGraph
from darwin.discourse import CausalClaim, ReferencedExperience, ResponsePlan, UncertaintyLevel
from darwin.dlm import DLMRenderResult, DarwinLanguageModule, GemmaDLM, StubDLM, gemma_dlm_available
from darwin.generative import GenerativeUniverse, GenerativeUniverseAdapter, WorldSpec
from darwin.instrumentation import BackgroundLogEntry, PlanLogEntry, StructuredLogger
from darwin.knowledge import CorpusIngestor, KnowledgeAtom, KnowledgeGraph
from darwin.retrieval import RetrievalPacket, RetrievedMemory
from darwin.runtime import DarwinRuntime
from darwin.self_modification import ModificationOutcome, ProposedModification, SelfModificationEngine
from darwin.semantics import SemanticFrame, SemanticParser
from darwin.thought import ThoughtTrace
from darwin.training_data import TrainingDataCollector, TrainingPair
from darwin.types import Action, Goal, Transition

__all__ = [
    "Action",
    "BackgroundLogEntry",
    "CausalChain",
    "CausalChainEngine",
    "CausalClaim",
    "CausalGraph",
    "CorpusIngestor",
    "Darwin",
    "DarwinLanguageModule",
    "DarwinRuntime",
    "DLMRenderResult",
    "GemmaDLM",
    "GenerativeUniverse",
    "GenerativeUniverseAdapter",
    "Goal",
    "KnowledgeAtom",
    "KnowledgeGraph",
    "ModificationOutcome",
    "PlanLogEntry",
    "ProposedModification",
    "ReferencedExperience",
    "ResponsePlan",
    "RetrievalPacket",
    "RetrievedMemory",
    "SelfModificationEngine",
    "SemanticFrame",
    "SemanticParser",
    "StructuredLogger",
    "StubDLM",
    "ThoughtTrace",
    "TrainingDataCollector",
    "TrainingPair",
    "Transition",
    "UncertaintyLevel",
    "WorldSpec",
    "gemma_dlm_available",
]
