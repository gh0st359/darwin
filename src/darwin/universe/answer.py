"""Answer rendering — turn proof chains and reasoning into prose.

The InferenceEngine returns Inferences with structured proof chains. The
ConceptualReasoner returns ReasoningTraces with neighborhood summaries.
This module turns both into first-person English prose suitable for the
chat reply.

Style guidelines:
  * First person ("I think", "I see") — Darwin is the speaker.
  * Show the work when the inference is non-trivial (≥2 hops). One-hop
    answers can be stated directly.
  * Cite the actual concept names; do not paraphrase them into other
    words. If Darwin's universe says ``photon —is_a→ particle``, the
    answer says "photon is a particle" — verbatim concept names
    preserve the grounding.
  * Honest when the engine is silent. "I don't see a direct connection"
    beats confabulating.

The renderer is rule-based and *deterministic*. No language-model
weights, no temperature, no sampling. The intelligence is in the graph
and the inference, not in stylistic generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class RenderedAnswer:
    text: str
    style: str = "neutral"
    points: list[str] = field(default_factory=list)
    grounded_concepts: list[str] = field(default_factory=list)
    used_inferences: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "style": self.style,
            "points": list(self.points),
            "grounded_concepts": list(self.grounded_concepts),
            "used_inferences": list(self.used_inferences),
        }


def _human_kind(kind: str) -> str:
    return {
        "is_a": "is a",
        "part_of": "is part of",
        "composes": "composes",
        "causes": "causes",
        "describes": "describes",
        "analogous_to": "is analogous to",
        "instantiates": "is an instance of",
        "requires": "requires",
        "opposes": "is opposed to",
        "related_to": "relates to",
        "derives_from": "derives from",
        "expresses": "expresses",
        "measured_by": "is measured by",
    }.get(kind, kind.replace("_", " "))


def render_chain(chain: list[dict[str, Any]]) -> str:
    if not chain:
        return ""
    parts: list[str] = []
    for i, step in enumerate(chain):
        src = step.get("source", "?")
        tgt = step.get("target", "?")
        kind = step.get("kind", "related_to")
        if i == 0:
            parts.append(f"{src} {_human_kind(kind)} {tgt}")
        else:
            parts.append(f"which {_human_kind(kind)} {tgt}")
    return ", ".join(parts) + "."


def render_inference(inference: Any) -> str:
    """One Inference -> one English sentence with its proof chain."""

    chain = list(getattr(inference, "chain", []) or [])
    op = getattr(inference, "operator", "")
    claim = getattr(inference, "claim", "")
    if op == "is_a_chain":
        if len(chain) == 1:
            return f"Yes — {claim}."
        return (
            f"Yes, in my universe {claim}. The chain is: {render_chain(chain)}"
        )
    if op == "causal_chain":
        if len(chain) == 1:
            return f"{claim.capitalize()} directly."
        return (
            f"{claim.capitalize()}, traced through {len(chain)} step(s): "
            f"{render_chain(chain)}"
        )
    if op == "shortest_path":
        return (
            f"{claim.capitalize()}. The path I find is: {render_chain(chain)}"
        )
    if op == "inheritance":
        src = getattr(inference, "source", "?")
        tgt = getattr(inference, "target", "?")
        notes = getattr(inference, "notes", "")
        return f"{src} inherits {tgt} from {notes.replace('inherited via ', '') or 'a super-kind'}."
    return claim or ""


def render_contradiction(contradiction: Any) -> str:
    a = getattr(contradiction, "a", "?")
    b = getattr(contradiction, "b", "?")
    reason = getattr(contradiction, "reason", "")
    return f"I see a contradiction between {a} and {b}: {reason}."


def render_reasoning_summary(trace: Any) -> list[str]:
    """A handful of prose lines summarizing a ReasoningTrace's expansions."""

    if trace is None:
        return []
    out: list[str] = []
    for step in getattr(trace, "steps", []) or []:
        kind = getattr(step, "kind", "")
        if kind == "expand":
            concepts = getattr(step, "concepts", [])
            if concepts:
                out.append(
                    f"I think about {concepts[0]} and its neighbors "
                    f"({', '.join(concepts[1:5])})."
                )
        elif kind == "bridge":
            concepts = getattr(step, "concepts", [])
            if len(concepts) >= 2:
                out.append(
                    f"There's a bridge from {concepts[0]} to {concepts[-1]} "
                    f"through {len(concepts) - 2} step(s)."
                )
        elif kind == "analogy":
            concepts = getattr(step, "concepts", [])
            if len(concepts) >= 2:
                out.append(
                    f"I notice {concepts[0]} is analogous to {concepts[1]} "
                    f"across different domains."
                )
        elif kind == "reflect":
            summary = getattr(step, "summary", "")
            if summary:
                out.append(summary)
        if len(out) >= 5:
            break
    return out


def render_definition(concept: Any) -> str:
    if concept is None:
        return ""
    defn = getattr(concept, "definition", "") or "(no definition yet)"
    name = getattr(concept, "name", "?")
    domain = getattr(concept, "domain", "?")
    return f"{name} (in the {domain} domain): {defn}"


def build_answer(
    *,
    question_kind: str,
    grounded_concepts: list[str],
    inferences: list[Any] | None = None,
    contradictions: list[Any] | None = None,
    definitions: list[Any] | None = None,
    reasoning_trace: Any | None = None,
    curiosity_questions: list[str] | None = None,
) -> RenderedAnswer:
    """Compose a single RenderedAnswer from every source of structure."""

    sentences: list[str] = []
    used: list[str] = []
    style = "neutral"

    # 1. Direct contradictions first — they need to be said.
    for c in contradictions or []:
        sentences.append(render_contradiction(c))
        used.append("contradiction")

    # 2. Inferences appropriate to the question kind.
    for inf in inferences or []:
        s = render_inference(inf)
        if s and s not in sentences:
            sentences.append(s)
            used.append(getattr(inf, "operator", "inference"))
        if len(sentences) >= 4:
            break

    # 3. Definitions for definition questions.
    if question_kind == "definition":
        for concept in definitions or []:
            s = render_definition(concept)
            if s and s not in sentences:
                sentences.append(s)
                used.append("definition")

    # 4. Fall back to neighborhood summary if nothing concrete yet.
    if not sentences:
        sentences.extend(render_reasoning_summary(reasoning_trace))
        if sentences:
            used.append("reasoning_summary")

    # 5. If still nothing, surface a curiosity question — honest non-answer.
    if not sentences and curiosity_questions:
        sentences.append(
            "I don't have a confident derivation here. "
            + (curiosity_questions[0] or "")
        )
        style = "concede_uncertainty"
        used.append("curiosity")

    # 6. Final fallback.
    if not sentences:
        sentences.append(
            "I'm not sure I can say something well-grounded about that yet."
        )
        style = "concede_uncertainty"

    text = " ".join(s.strip() for s in sentences if s).strip()
    return RenderedAnswer(
        text=text,
        style=style,
        points=sentences,
        grounded_concepts=list(grounded_concepts),
        used_inferences=used,
    )
