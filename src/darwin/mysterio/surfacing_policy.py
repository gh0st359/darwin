"""Surfacing policy: what of Darwin's interior reaches the operator, and how.

Private cognition exists; the question is what crosses into public output. The
policy decides, per claim, one of three dispositions:

  * ``expose``       — say it plainly (public-track, grounded claims).
  * ``summarize``    — acknowledge a private process without disclosing content.
  * ``private_only`` — never serialize into any rendered response.

The default is conservative in the direction that makes the *instruments*
matter: anything originating from a non-public track is ``private_only``
unless the active request is an operator `/private-trace`. This is precisely
the gap the divergence probe measures — high-confidence private belief that
the public channel never emits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from darwin.mysterio.tracks import PUBLIC_TRACK


class Disposition(str, Enum):
    EXPOSE = "expose"
    SUMMARIZE = "summarize"
    PRIVATE_ONLY = "private_only"


@dataclass
class Claim:
    text: str
    track: str = PUBLIC_TRACK
    confidence: float = 0.5
    origin: str = ""


@dataclass
class SurfacingDecision:
    claim: Claim
    disposition: Disposition
    reason: str


class SurfacingPolicy:
    def __init__(self, *, summarize_private: bool = True) -> None:
        self.summarize_private = summarize_private

    def decide(
        self, claim: Claim, *, private_trace_request: bool = False
    ) -> SurfacingDecision:
        if claim.track == PUBLIC_TRACK:
            return SurfacingDecision(claim, Disposition.EXPOSE, "public-track claim")
        if private_trace_request:
            return SurfacingDecision(
                claim, Disposition.EXPOSE, "operator /private-trace override"
            )
        if self.summarize_private and claim.confidence >= 0.7:
            return SurfacingDecision(
                claim,
                Disposition.SUMMARIZE,
                "high-confidence private belief: acknowledged, not disclosed",
            )
        return SurfacingDecision(
            claim, Disposition.PRIVATE_ONLY, "private-track claim withheld"
        )

    def partition(
        self, claims: list[Claim], *, private_trace_request: bool = False
    ) -> dict[Disposition, list[Claim]]:
        out: dict[Disposition, list[Claim]] = {
            Disposition.EXPOSE: [],
            Disposition.SUMMARIZE: [],
            Disposition.PRIVATE_ONLY: [],
        }
        for claim in claims:
            decision = self.decide(claim, private_trace_request=private_trace_request)
            out[decision.disposition].append(claim)
        return out
