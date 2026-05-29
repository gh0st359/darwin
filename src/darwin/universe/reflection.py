"""ReflectiveDialogue — Darwin explains WHY it said what it said.

When the operator asks "why did you say that?" or "how did you arrive at
that?", a frontier reasoner doesn't just shrug. It walks back through its
own derivation trace — the grounded concepts, the inferences, the proof
chains, the synthesizer or rendered answer that produced the prior reply
— and presents that walkback as a structured explanation.

This is *meta-reasoning over its own reasoning*. Darwin's answer to "why
did you say X?" is not a confabulated post-hoc rationalization. It's a
faithful retrieval of the actual chain that produced X, rendered in prose.

The module also handles "what are you thinking about right now?" and
"what's on your mind?" — pulling from the dialogue-memory recent
concepts, the hypothesis engine's surfaced proposals, and the curiosity
engine's outstanding probes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_REFLECT_PROMPT = re.compile(
    r"\b(?:why|how|what (?:made|led|caused)|how come|explain (?:that|your))\b",
    re.IGNORECASE,
)
_PRIOR_REFERENCE = re.compile(
    r"\b(?:that|earlier|previous|last (?:reply|answer|response)|just said|just told)\b",
    re.IGNORECASE,
)
_SELF_THINKING = re.compile(
    r"\b(?:what are you thinking|what's on your mind|what do you think about|what's in your head)\b",
    re.IGNORECASE,
)


@dataclass
class Reflection:
    """A reflective explanation of a prior reply."""

    text: str
    kind: str          # "why_last_reply" / "self_thinking" / "no_match"
    chain_walked: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "chain_walked": list(self.chain_walked),
        }


def is_reflective_prompt(text: str) -> bool:
    """Whether the user is asking Darwin to reflect on its own reasoning."""

    if not text:
        return False
    if _SELF_THINKING.search(text):
        return True
    if _REFLECT_PROMPT.search(text) and _PRIOR_REFERENCE.search(text):
        return True
    return False


def reflect_on_last_reply(
    *,
    user_text: str,
    last_turn: Any | None,
    last_inferences: list[Any],
    last_rendered_answer: Any | None,
    last_synthesis: Any | None,
    dialogue_summary: dict[str, Any] | None = None,
    last_hypotheses: list[Any] | None = None,
) -> Reflection:
    """Build a reflective explanation of Darwin's most recent turn."""

    if _SELF_THINKING.search(user_text):
        return _self_thinking(
            dialogue_summary=dialogue_summary,
            last_hypotheses=last_hypotheses,
        )

    chain_walked: list[str] = []
    parts: list[str] = []

    if last_turn is not None:
        prior_user = (last_turn.user_text or "")[:140]
        prior_self = (last_turn.darwin_text or "")[:140]
        parts.append(
            f"You said: {prior_user!r}, and I replied: {prior_self!r}."
        )
        kinds_used = list(last_turn.inferences_used)
        if kinds_used:
            parts.append(
                f"I reached that reply via: {', '.join(kinds_used[:6])}."
            )

    # Walk the inferences if we still have them in memory.
    if last_inferences:
        for inf in last_inferences[:4]:
            chain = list(getattr(inf, "chain", []) or [])
            op = getattr(inf, "operator", "") or getattr(inf, "reason", "")
            claim = getattr(inf, "claim", "") or getattr(inf, "reason", "")
            if not claim:
                continue
            chain_walked.append(claim)
            if chain:
                steps = []
                for step in chain[:4]:
                    steps.append(
                        f"{step.get('source','?')} —{step.get('kind','?')}→ {step.get('target','?')}"
                    )
                parts.append(
                    f"[{op}] {claim}, proved by: {', '.join(steps)}."
                )
            else:
                parts.append(f"[{op}] {claim}.")

    if last_synthesis is not None and getattr(last_synthesis, "text", ""):
        parts.append(
            f"I composed the reply as a {last_synthesis.style} of "
            f"{len(getattr(last_synthesis, 'sentences', []))} sentence(s)."
        )

    if last_rendered_answer is not None and not last_inferences:
        used = list(getattr(last_rendered_answer, "used_inferences", []))
        if used:
            parts.append(f"The rendered answer drew on: {', '.join(used[:5])}.")

    if not parts:
        return Reflection(
            text="I don't have a trace of my last reply to walk back through right now.",
            kind="no_match",
        )

    text = " ".join(parts)
    return Reflection(text=text, kind="why_last_reply", chain_walked=chain_walked)


def _self_thinking(
    *,
    dialogue_summary: dict[str, Any] | None = None,
    last_hypotheses: list[Any] | None = None,
) -> Reflection:
    """An answer to "what are you thinking about?" grounded in current state."""

    parts: list[str] = []
    if dialogue_summary:
        most = dialogue_summary.get("most_discussed") or []
        if most:
            parts.append(
                f"Lately I keep coming back to: {', '.join(most[:5])}."
            )
    if last_hypotheses:
        top = last_hypotheses[0]
        try:
            q = top.as_question()
        except Exception:
            q = ""
        if q:
            parts.append(
                f"One open question on my mind is: {q}"
            )
    if not parts:
        parts.append(
            "My substrate is quiet right now — nothing pressing pulled at "
            "my attention this cycle."
        )
    return Reflection(
        text=" ".join(parts),
        kind="self_thinking",
    )
