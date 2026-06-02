"""Darwin's Cortical Mesh — the non-LLM neural substrate.

This package introduces a persistent activation substrate coupled
bidirectionally to ``darwin.universe.ConceptUniverse``. Every concept in
the universe gets a corresponding ConceptCell in the mesh; every typed
Relation becomes a weighted Connection. Activation propagates by
firing-and-decay dynamics; connections learn via Hebbian + STDP
plasticity over the mesh's recent firings.

The mesh is the brain Darwin uses to *intuit* — to surface concepts that
are activation-adjacent to current focus, to learn temporal causality
from co-firings, to maintain a working-memory ring of what just fired.
Symbolic reasoning (the universe's inference engine, V-Reason's chainers)
operates over the same concept graph; the mesh is its non-symbolic twin.

Pure-Python ceiling: 100K cells / 10M connections. The V-Scale phase
adds an optional torch backend behind a feature flag, scaling that to
10M / 1B without changing the public API in this module.
"""

from darwin.mesh.cell import ConceptCell, Connection
from darwin.mesh.coupling import CouplingStats, UniverseMeshCoupling
from darwin.mesh.mesh import CorticalMesh, FiringEvent, PropagationResult
from darwin.mesh.persistence import (
    MeshPersistence,
    MeshPersistenceState,
    default_mesh_path,
)
from darwin.mesh.plasticity import (
    HebbianRule,
    PlasticityController,
    PlasticityReport,
    STDPRule,
)


__all__ = [
    "ConceptCell",
    "Connection",
    "CorticalMesh",
    "CouplingStats",
    "FiringEvent",
    "HebbianRule",
    "MeshPersistence",
    "MeshPersistenceState",
    "PlasticityController",
    "PlasticityReport",
    "PropagationResult",
    "STDPRule",
    "UniverseMeshCoupling",
    "default_mesh_path",
]
