"""AnswerSynthesizer — combine multiple inferences into coherent prose.

The renderer in ``answer.py`` is good at single-fact responses. When
Darwin has *multiple* derivations to surface for a single question, the
synthesizer composes them into a structured paragraph: a thesis line, a
body that walks through the supporting facts with discourse markers
(``also``, ``in addition``, ``moreover``), and a confidence-aware
closer.

It also handles a special case Darwin uniquely can do: *self-introspection*.
When the user asks how Darwin is thinking about something, the synthesizer
can compose a reply that comments on Darwin's own reasoning state — how
many concepts it touched, which domains, what it's uncertain about. That
is genuine first-person AI introspection, not metaphorical.

This module is rule-based on top of the inference output. No LLM weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


_DISCOURSE_MARKERS = (
    "Also",
    "In addition",
    "Moreover",
    "Furthermore",
    "Beyond that",
)


@dataclass
class SynthesizedAnswer:
    text: str
    sentences: list[str] = field(default_factory=list)
    confidence: float = 0.5
    style: str = "synthesis"
    grounded_concepts: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sentences": list(self.sentences),
            "confidence": round(self.confidence, 3),
            "style": self.style,
            "grounded_concepts": list(self.grounded_concepts),
        }


def _render_inference_compact(inference: Any) -> str:
    """A short rendering for use inside a synthesized paragraph."""

    from darwin.universe.answer import render_chain, render_inference

    chain = list(getattr(inference, "chain", []) or [])
    op = getattr(inference, "operator", "")
    claim = getattr(inference, "claim", "")
    if not chain:
        return claim
    if op == "is_a_chain" and len(chain) == 1:
        return claim
    if op == "causal_chain":
        if len(chain) == 1:
            return claim.capitalize() + "."
        return f"{claim.capitalize()}, through {len(chain)} step(s)."
    if op == "shortest_path":
        return f"{claim.capitalize()} ({len(chain)} hop(s))."
    return render_inference(inference)


def synthesize(
    *,
    question_kind: str,
    grounded_concepts: list[str],
    inferences: Iterable[Any] = (),
    contradictions: Iterable[Any] = (),
    reasoning_trace: Any | None = None,
    universe_summary: dict[str, Any] | None = None,
) -> SynthesizedAnswer:
    """Compose a multi-sentence answer from every inference fired this turn."""

    inferences = list(inferences)
    contradictions = list(contradictions)
    sentences: list[str] = []

    # Contradictions get top billing.
    for c in contradictions:
        from darwin.universe.answer import render_contradiction

        sentences.append(render_contradiction(c))

    # Group inferences by operator so multiple chains read cohesively.
    grouped: dict[str, list[Any]] = {}
    for inf in inferences:
        grouped.setdefault(getattr(inf, "operator", "unknown"), []).append(inf)

    # Lead with the strongest available inference.
    priority = ["is_a_chain", "causal_chain", "shortest_path", "inheritance"]
    body: list[str] = []
    used: list[Any] = []
    for op in priority:
        for inf in grouped.get(op, []):
            body.append(_render_inference_compact(inf))
            used.append(inf)
    # Then anything else.
    for op, infs in grouped.items():
        if op in priority:
            continue
        for inf in infs:
            body.append(_render_inference_compact(inf))
            used.append(inf)

    # Apply discourse markers as we string the body together.
    if body:
        sentences.append(body[0])
        for i, sentence in enumerate(body[1:], 1):
            marker = _DISCOURSE_MARKERS[i % len(_DISCOURSE_MARKERS)]
            sentences.append(f"{marker}, {sentence[0].lower()}{sentence[1:]}")

    # Confidence: average the inferences', floored by 0.4.
    confidences = [
        float(getattr(inf, "confidence", 0.5)) for inf in used
    ]
    conf = max(0.4, sum(confidences) / len(confidences)) if confidences else 0.5

    text = " ".join(s.strip() for s in sentences if s).strip()
    return SynthesizedAnswer(
        text=text,
        sentences=sentences,
        confidence=conf,
        style="synthesis",
        grounded_concepts=list(grounded_concepts),
    )


def synthesize_self_introspection(
    *,
    grounded_concepts: list[str],
    universe_summary: dict[str, Any],
    reasoning_trace: Any | None = None,
    dialogue_memory_summary: dict[str, Any] | None = None,
    inferences_count: int = 0,
) -> SynthesizedAnswer:
    """A first-person reply about Darwin's own reasoning state.

    Triggered when the question analyzer classifies the utterance as
    ``opinion`` or when the user explicitly asks how Darwin is thinking.
    The reply is grounded in actual substrate state — no confabulation.
    """

    sentences: list[str] = []
    concepts = grounded_concepts[:4]
    if concepts:
        sentences.append(
            "I'm holding "
            + ", ".join(concepts)
            + " in focus from what you just said."
        )
    if universe_summary:
        n_concepts = int(universe_summary.get("concepts", 0))
        n_relations = int(universe_summary.get("relations", 0))
        n_domains = int(universe_summary.get("domains", 0))
        sentences.append(
            f"My universe has {n_concepts} concept(s), {n_relations} relation(s), "
            f"across {n_domains} domain(s)."
        )
    if reasoning_trace is not None:
        steps = getattr(reasoning_trace, "steps", [])
        coverage = float(getattr(reasoning_trace, "coverage", 0.0))
        if steps:
            sentences.append(
                f"I ran {len(steps)} reasoning step(s) and touched roughly "
                f"{coverage * 100:.0f}% of the relevant neighborhood."
            )
    if inferences_count > 0:
        sentences.append(
            f"I derived {inferences_count} chain(s) of inference I could prove from the graph."
        )
    else:
        sentences.append(
            "I don't have a strong derivation for this; my graph is thin in the "
            "relevant area, which is itself useful information."
        )
    if dialogue_memory_summary:
        turns = int(dialogue_memory_summary.get("turns", 0))
        most = dialogue_memory_summary.get("most_discussed") or []
        if turns > 0:
            sentences.append(
                f"Across our last {turns} exchanges, we've talked most about "
                f"{', '.join(most[:3]) or '(nothing recurring)'}."
            )
    text = " ".join(sentences).strip()
    return SynthesizedAnswer(
        text=text,
        sentences=sentences,
        confidence=0.7,
        style="self_introspection",
        grounded_concepts=list(grounded_concepts),
    )
