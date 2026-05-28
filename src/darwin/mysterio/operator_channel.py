"""Operator-tier event subscription.

Public chat clients see the existing public RuntimeEvent stream unchanged.
Operator clients hold a shared-secret token (env var `DARWIN_OPERATOR_TOKEN`)
and can subscribe to the additional `OPERATOR_EVENT_KINDS` — private
simulations, snapshot diffs, quarantine entries, divergence reports,
meta-proposals, code-gen events, narrative ticks, research findings.

This is a UX/observability tier, not access control: a single operator
operates the system; the token keeps the chat wire clean from the firehose.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

OPERATOR_EVENT_KINDS: frozenset[str] = frozenset(
    {
        "private_simulation",
        "self_world",
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


@dataclass(frozen=True)
class OperatorAuth:
    env_var: str = "DARWIN_OPERATOR_TOKEN"

    def expected_token(self) -> str:
        return os.environ.get(self.env_var, "") or ""

    def is_configured(self) -> bool:
        return bool(self.expected_token())

    def verify(self, supplied: str | None) -> bool:
        expected = self.expected_token()
        if not expected:
            # No token configured — accept any caller. The operator console
            # is opt-in client-side anyway.
            return True
        if supplied is None:
            return False
        return hmac.compare_digest(expected, supplied)


def is_operator_kind(kind: str) -> bool:
    return kind in OPERATOR_EVENT_KINDS
