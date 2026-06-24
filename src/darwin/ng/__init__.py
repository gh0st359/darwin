"""Darwin NG orchestration layer.

This package is the first concrete Darwin NG landing: an inspectable
coordinator that binds the existing universe, mesh, neural, Mysterio,
autonomy, and self-modification substrates into a single cognitive cycle.
"""

from darwin.ng.core import (
    DarwinNG,
    DarwinNGState,
    GoalCandidate,
    NGContent,
    NGPlan,
    SafetyAssessment,
)

__all__ = [
    "DarwinNG",
    "DarwinNGState",
    "GoalCandidate",
    "NGContent",
    "NGPlan",
    "SafetyAssessment",
]
