"""CorrectionDetector — Darwin updates its universe when the operator says no.

Real conversation involves correction. When the operator says "No, that's
wrong" after Darwin's reply, or "Actually X is a Y, not a Z", Darwin
should integrate the correction: refute the wrong inference so the
HypothesisEngine doesn't propose it again, and (when the user provides
the correction itself) fuse the new relation in its place.

This module recognizes correction patterns and emits structured
``Correction`` objects the runtime can apply. It is rule-based and
conservative — it would rather miss a subtle correction than incorrectly
flip a stable belief.

Correction kinds:
  * ``negate_prior``  — "no", "that's wrong", "incorrect" → flag the
    previous Darwin claim as refuted (no replacement).
  * ``replace``       — "actually X is Y" / "no, X is Y" → refute prior
    and fuse the replacement.
  * ``retract``       — "I was wrong about X" / "forget what I said"
    (operator self-correction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_NEGATION_RX = re.compile(
    r"^\s*(?:no[,.\s!]|that(?:'|’)s\s+wrong|incorrect|that(?:'|’)s\s+not\s+right|nope|nah)",
    re.IGNORECASE,
)
_RETRACTION_RX = re.compile(
    r"\b(?:i\s+(?:was\s+)?wrong\s+about|forget\s+what\s+i\s+said|scratch\s+that|never\s+mind\s+what\s+i\s+said)\b",
    re.IGNORECASE,
)
_REPLACEMENT_RX = re.compile(
    r"\b(?:actually|in\s+fact|really)\b\s*[,.\s]?\s*(?P<rest>[^.!?\n]+)",
    re.IGNORECASE,
)


@dataclass
class Correction:
    """A correction signal extracted from a user utterance."""

    kind: str          # "negate_prior" / "replace" / "retract"
    text: str
    replacement: str = ""    # the corrective clause (for "replace")
    notes: str = ""
    refuted_keys: list[tuple[str, str, str]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "replacement": self.replacement,
            "notes": self.notes,
            "refuted_keys": list(self.refuted_keys),
        }


def detect_correction(text: str) -> Correction | None:
    """Inspect the user text for a correction signal."""

    if not text:
        return None
    if _NEGATION_RX.match(text):
        # Look for a replacement clause (e.g., "No, X is a Y").
        replacement_match = _REPLACEMENT_RX.search(text)
        if replacement_match:
            return Correction(
                kind="replace",
                text=text,
                replacement=replacement_match.group("rest").strip(),
                notes="negation followed by replacement",
            )
        # Pure negation, no replacement.
        return Correction(
            kind="negate_prior",
            text=text,
            notes="negation with no replacement clause",
        )
    if _RETRACTION_RX.search(text):
        return Correction(
            kind="retract",
            text=text,
            notes="operator self-correction",
        )
    replacement_match = _REPLACEMENT_RX.search(text)
    if replacement_match:
        return Correction(
            kind="replace",
            text=text,
            replacement=replacement_match.group("rest").strip(),
            notes="standalone replacement cue",
        )
    return None


def apply_correction(
    correction: Correction,
    *,
    last_grounded_concepts: list[str],
    last_inferences: list[Any],
    fusion,
    hypothesis_engine,
    universe,
) -> list[tuple[str, str, str]]:
    """Apply a correction. Returns the (source, kind, target) triples
    that were refuted as a result of the correction."""

    refuted: list[tuple[str, str, str]] = []

    if correction.kind in ("negate_prior", "replace"):
        # Refute every inference key that Darwin used in its most recent reply.
        for inference in last_inferences or []:
            src = getattr(inference, "source", "") or ""
            tgt = getattr(inference, "target", "") or ""
            if not src or not tgt:
                continue
            # Try to determine the relation kind from the inference itself.
            op = getattr(inference, "operator", "")
            if op == "is_a_chain":
                kind = "is_a"
            elif op == "causal_chain":
                kind = "causes"
            else:
                kind = "related_to"
            refuted.append((src, kind, tgt))
            try:
                hypothesis_engine.refute(src, kind, tgt)
            except Exception:
                pass
        correction.refuted_keys = list(refuted)

    if correction.kind == "replace" and correction.replacement:
        # Fuse the replacement clause as new declarative content.
        try:
            fusion.fuse(correction.replacement)
        except Exception:
            pass

    if correction.kind == "retract":
        # The operator says they were wrong. Refute the most recent fused
        # edges so they don't ride forward.
        try:
            recent = fusion.recent(limit=4)
            for f in recent:
                refuted.append((f.source, f.kind, f.target))
                hypothesis_engine.refute(f.source, f.kind, f.target)
        except Exception:
            pass
        correction.refuted_keys = list(refuted)

    return refuted
