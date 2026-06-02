"""LeakGate — the hard constraint that chat output never leaks internals.

Hard rule: no JSON, no curly braces, no Python repr fragments, no
``"key": value`` pairs, no `[event ...]` event-stream markers, no
slash-commands, no all-caps category names from the substrate. The
chat client speaks like a person; if the pipeline produces structured
output, the gate rejects it and the runtime falls back to a
deterministic sanitised paraphrase.

The gate operates on the *final* surface string. It does not look at
internal state. False positives are preferred over false negatives —
in case of doubt the gate rejects and the system speaks more boringly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Patterns that ALWAYS indicate a structured-internal leak.
_HARD_FORBIDDEN_PATTERNS = [
    re.compile(r"[{}]"),                               # curly braces
    re.compile(r"\[event\s+\w+\]", re.IGNORECASE),     # event-stream markers
    re.compile(r"^\s*\{", re.MULTILINE),               # JSON-looking line starts
    re.compile(r'\"[A-Za-z_][A-Za-z_0-9]*\"\s*:'),     # JSON key:value
    re.compile(r"^\s*/[a-z][a-z_\-]+\b", re.MULTILINE),  # leading slash-command
    re.compile(r"BusTopic\.\w+"),
    re.compile(r"\bdef\s+\w+\s*\("),                   # Python def fragment
    re.compile(r"<\w+ object at 0x[0-9a-fA-F]+>"),     # default repr
]


# Substring tokens that point at structured-internal payload field names.
# These get rejected as bare tokens in the output (not embedded inside
# legitimate prose).
_PAYLOAD_FIELD_TOKENS = (
    "answer_points",
    "thesis:",
    "causal_claims",
    "uncertainty_levels",
    "clarification_questions",
    "retrieved_used",
    "to_record",
    "structural_numbers",
)


# Bracket patterns that, in concert with internal-looking keywords, mean
# we're echoing structure.
_BRACKETED_INTERNAL_RX = re.compile(
    r"\[(?:is_a_chain|causal_chain|shortest_path|inheritance|contradiction|definition)[^]]*\]",
    re.IGNORECASE,
)


@dataclass
class LeakGateResult:
    """Outcome of one gate check."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    text: str = ""
    sanitized_fallback: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "text": self.text[:300],
            "sanitized_fallback": self.sanitized_fallback[:300],
        }


def _sanitize(text: str) -> str:
    """Last-ditch sanitiser: strip the offending tokens, collapse whitespace.

    The result is never preferable to a real answer — it exists so the
    fallback doesn't speak nonsense when the gate had to step in.
    """

    cleaned = text
    cleaned = re.sub(r"[{}]+", "", cleaned)
    cleaned = re.sub(r"\[event\s+\w+\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = _BRACKETED_INTERNAL_RX.sub("", cleaned)
    cleaned = re.sub(r"\"\w+\"\s*:", "", cleaned)
    cleaned = re.sub(r"^\s*/[a-z][a-z_\-]+\b.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"BusTopic\.\w+", "", cleaned)
    cleaned = re.sub(r"<\w+ object at 0x[0-9a-fA-F]+>", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "I tried to say something but ran into a formatting issue. Can you ask again?"
    return cleaned


class LeakGate:
    """Reject structured-internal leaks in chat output."""

    def __init__(self) -> None:
        pass

    def check(self, text: str, *, fallback_text: str = "") -> LeakGateResult:
        """Return a gate result. If failed, ``sanitized_fallback`` is set."""

        if not isinstance(text, str):
            return LeakGateResult(
                passed=False,
                reasons=["non-string output"],
                text=str(text),
                sanitized_fallback=fallback_text or "I can't speak right now.",
            )
        reasons: list[str] = []
        for pattern in _HARD_FORBIDDEN_PATTERNS:
            if pattern.search(text):
                reasons.append(f"forbidden pattern matched: {pattern.pattern!r}")
        for token in _PAYLOAD_FIELD_TOKENS:
            if token.lower() in text.lower():
                reasons.append(f"payload-field token leaked: {token!r}")
        if _BRACKETED_INTERNAL_RX.search(text):
            reasons.append("operator-bracketed inference tag leaked")
        if reasons:
            # Sanitise BOTH the primary text AND any caller-supplied
            # fallback — a leaky fallback is just as unsafe as a leaky
            # primary. Prefer the cleaned-fallback when it carries usable
            # content; otherwise fall back to the cleaned primary.
            primary_clean = _sanitize(text)
            if fallback_text:
                fallback_clean = _sanitize(fallback_text)
                sanitized = fallback_clean if len(fallback_clean) >= 10 else primary_clean
            else:
                sanitized = primary_clean
            return LeakGateResult(
                passed=False,
                reasons=reasons,
                text=text,
                sanitized_fallback=sanitized,
            )
        return LeakGateResult(passed=True, text=text)


__all__ = ["LeakGate", "LeakGateResult"]
