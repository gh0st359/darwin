"""Mysterio: the recursive self-modification + apparatus layer for Darwin.

This package extends Darwin with typed proposal grammar, snapshot/diff
introspection, a self-modifiable accept gate, a generative meta-proposer,
a divergence probe, an operator-tier event channel, and (in later phases)
private simulation tracks, code-level self-modification, and a distributed
cognition bus.

The single design principle of this package: every emergent capability ships
behind an instrument that can already observe it. The instruments are the
deliverable, not the constraint.
"""

from darwin.mysterio.proposal_spec import ProposalSpec
from darwin.mysterio.safety import (
    SAFETY_BOUNDS,
    ContainmentError,
    MutationKind,
    SafetyTier,
    TouchRecorder,
)

__all__ = [
    "SAFETY_BOUNDS",
    "ContainmentError",
    "MutationKind",
    "ProposalSpec",
    "SafetyTier",
    "TouchRecorder",
]
