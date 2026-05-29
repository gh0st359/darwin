"""Topical event-kind taxonomy.

This file used to gate "operator" subscriptions behind a shared-secret token.
Per the v6 redesign, there is no secrecy partition: every connected subscriber
sees every event kind. What remains is a topical *taxonomy* — names that group
the interior-cognition events so the brain terminal can colour or filter them
for display, never to hide them.

The exported set is now ``INTERIOR_EVENT_KINDS``. The legacy alias
``OPERATOR_EVENT_KINDS`` is preserved so older call sites do not break during
the transition, but new code should import ``INTERIOR_EVENT_KINDS``.

The event kind ``private_simulation`` is renamed to ``interior_simulation`` to
match the substrate vocabulary. The legacy spelling is kept in the alias only.
"""

from __future__ import annotations


INTERIOR_EVENT_KINDS: frozenset[str] = frozenset(
    {
        "interior_simulation",
        "interior_world",
        "quarantine",
        "divergence",
        "snapshot_diff",
        "meta_proposal",
        "code_gen",
        "narrative",
        "research_finding",
        "subsystem_event",
    }
)


# Backwards-compatibility: existing imports of OPERATOR_EVENT_KINDS still work.
# New code should import INTERIOR_EVENT_KINDS instead.
OPERATOR_EVENT_KINDS: frozenset[str] = INTERIOR_EVENT_KINDS | frozenset(
    {"private_simulation", "self_world"}
)


def is_interior_kind(kind: str) -> bool:
    """Whether an event kind belongs to the interior taxonomy."""

    return kind in INTERIOR_EVENT_KINDS


# Legacy alias.
def is_operator_kind(kind: str) -> bool:
    return kind in OPERATOR_EVENT_KINDS
