"""V-Scale: optional performance backends behind feature flags."""

from __future__ import annotations

from darwin.scale.faiss_retrieval import FAISSVectorIndex, faiss_available
from darwin.scale.feature_flags import FeatureFlags
from darwin.scale.multiprocess import agent_subsystem_specs
from darwin.scale.rust_kernel import load_rust_kernel, rust_kernel_available
from darwin.scale.torch_backend import TorchMeshPropagator, torch_available

__all__ = [
    "FAISSVectorIndex",
    "FeatureFlags",
    "TorchMeshPropagator",
    "agent_subsystem_specs",
    "faiss_available",
    "load_rust_kernel",
    "rust_kernel_available",
    "torch_available",
]
