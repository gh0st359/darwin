"""Mysterio: recursive self-modification, distributed cognition, private mental life.

This package extends Darwin with typed proposal grammar, snapshot/diff
introspection, a self-modifiable accept gate, a generative meta-proposer, a
divergence probe, an operator-tier event channel, code-level self-modification,
self-trained embeddings, a cognition bus + process supervisor, and (from v7)
private cognition tracks, proprioception, self-simulation, observer modelling,
an autobiographical narrative, and surfacing policy.

Design principle: every emergent capability ships behind an instrument that
can already observe it. The instruments are the deliverable, not the constraint.
"""

from darwin.mysterio.bus import BusTopic, CognitionBus
from darwin.mysterio.code_gen import CodeGenerator, GeneratedModule, ModuleLoader
from darwin.mysterio.embeddings import CausalEmbeddingSpace
from darwin.mysterio.narrative import NarrativeChunk, NarrativeThread
from darwin.mysterio.observer_modeler import ObserverEntity, ObserverModeler, ObserverWorld
from darwin.mysterio.private_simulator import PrivateRollout, PrivateSimulator, PrivateWriteViolation
from darwin.mysterio.processes import (
    CognitionSupervisor,
    RestartPolicy,
    SubsystemSpec,
)
from darwin.mysterio.proposal_spec import ProposalSpec
from darwin.mysterio.proprioception import (
    InternalProprioceptionAdapter,
    ProprioceptiveState,
)
from darwin.mysterio.safety import (
    SAFETY_BOUNDS,
    ContainmentError,
    MutationKind,
    SafetyTier,
    TouchRecorder,
)
from darwin.mysterio.surfacing_policy import Claim, Disposition, SurfacingPolicy
from darwin.mysterio.tracks import (
    PRIVATE_SELF_TRACK,
    PUBLIC_TRACK,
    TrackedSubstrate,
    TrackRegistry,
)

__all__ = [
    "SAFETY_BOUNDS",
    "BusTopic",
    "Claim",
    "CausalEmbeddingSpace",
    "CodeGenerator",
    "CognitionBus",
    "CognitionSupervisor",
    "ContainmentError",
    "Disposition",
    "GeneratedModule",
    "InternalProprioceptionAdapter",
    "ModuleLoader",
    "MutationKind",
    "NarrativeChunk",
    "NarrativeThread",
    "ObserverEntity",
    "ObserverModeler",
    "ObserverWorld",
    "PRIVATE_SELF_TRACK",
    "PUBLIC_TRACK",
    "PrivateRollout",
    "PrivateSimulator",
    "PrivateWriteViolation",
    "ProposalSpec",
    "ProprioceptiveState",
    "RestartPolicy",
    "SafetyTier",
    "SubsystemSpec",
    "SurfacingPolicy",
    "TouchRecorder",
    "TrackRegistry",
    "TrackedSubstrate",
]
