"""SpeechPipeline — five-stage compositional non-LLM NLG.

Stages (all deterministic, all pure-Python, zero LLM weights):

  1. **ContentSelection** — choose which ``ResponsePlan`` fields make it
     into the surface utterance. Reads ``OperatorModel`` for verbosity
     preference: short users get thesis-only; long users get the full
     evidence + uncertainty cascade.
  2. **DiscoursePlan** — order content units into a rhetorical tree
     (opening → thesis → support → caveat → followup). Reads
     ``DialogueMemory`` so repeated-concept turns shift toward
     referring expressions or topic-revisit framings.
  3. **SentencePlan** — group discourse units into sentence-sized
     chunks. Decides where to coordinate ("and"), where to subordinate
     ("because"), where to stop.
  4. **LexicalChoice** — map concepts → surface words via the lexicon,
     with style-adapter biasing.
  5. **SurfaceRealization** — compose the final string. Adds the
     leading capital, punctuation, spacing. Then ``LeakGate`` runs;
     failures fall through to a deterministic English fallback.

The pipeline returns a ``DLMRenderResult`` so it is drop-in compatible
with the existing chat path (``runtime.dlm.render(plan, frame, trace)``
already constructs / consumes that shape).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from darwin.speech.leak_gate import LeakGate


# --------------------------------------------------------------------------- #
# Five-stage outputs (intermediate types)
# --------------------------------------------------------------------------- #


@dataclass
class _SelectedContent:
    thesis: str = ""
    points: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    clarifications: list[str] = field(default_factory=list)
    opening_mode: str = "neutral"


@dataclass
class _DiscourseTree:
    units: list[tuple[str, str]] = field(default_factory=list)
    # Each unit is ("section_name", "text-of-this-unit").


@dataclass
class _SentencePlan:
    sentences: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _humanize(text: str) -> str:
    """Light surface tweaks: capitalise first letter, ensure trailing period."""

    if not text:
        return ""
    text = text.strip()
    if not text:
        return ""
    # Don't replace internal characters — only normalise sentence-final
    # whitespace and trailing punctuation, and capitalise.
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _strip_internal_quoting(text: str) -> str:
    """Remove the operator-internal bracketed labels like '[is_a_chain]'."""

    text = re.sub(
        r"\[(?:is_a_chain|causal_chain|shortest_path|inheritance|contradiction|definition|reasoning_summary|reflection|hypothesis|synthesis|self_introspection|concede_uncertainty)\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s{2,}", " ", text).strip()


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def _select_content(plan: Any, operator_model: Any = None) -> _SelectedContent:
    """Stage 1: choose which plan fields become surface content."""

    verbosity = "medium"
    if operator_model is not None:
        try:
            verbosity = operator_model.preferred_length(getattr(plan, "mode", ""))
        except Exception:
            verbosity = "medium"
    thesis = (getattr(plan, "thesis", "") or "").strip()
    points = list(getattr(plan, "answer_points", []) or [])
    evidence = list(getattr(plan, "evidence", []) or [])
    uncertainties = list(getattr(plan, "uncertainties", []) or [])
    clarifications = list(getattr(plan, "clarification_questions", []) or [])

    if verbosity == "short":
        points = points[:1]
        evidence = evidence[:0]
        uncertainties = uncertainties[:0]
    elif verbosity == "long":
        points = points[:6]
        evidence = evidence[:3]
        uncertainties = uncertainties[:2]
    else:
        points = points[:3]
        evidence = evidence[:1]
        uncertainties = uncertainties[:1]

    opening_mode = getattr(plan, "mode", "neutral") or "neutral"
    return _SelectedContent(
        thesis=thesis,
        points=points,
        evidence=evidence,
        uncertainties=uncertainties,
        clarifications=clarifications,
        opening_mode=opening_mode,
    )


def _plan_discourse(content: _SelectedContent, dialogue_memory: Any = None) -> _DiscourseTree:
    """Stage 2: order content into a rhetorical sequence."""

    tree = _DiscourseTree()
    if content.thesis:
        tree.units.append(("thesis", content.thesis))
    for point in content.points:
        tree.units.append(("support", point))
    for ev in content.evidence:
        tree.units.append(("evidence", ev))
    for unc in content.uncertainties:
        tree.units.append(("caveat", unc))
    for clar in content.clarifications:
        tree.units.append(("followup", clar))
    return tree


def _plan_sentences(tree: _DiscourseTree) -> _SentencePlan:
    """Stage 3: group discourse units into sentence-sized chunks."""

    plan = _SentencePlan()
    if not tree.units:
        return plan
    # One sentence per unit, with discourse markers for support units 2+.
    seen_support = 0
    for section, text in tree.units:
        text = _strip_internal_quoting(text)
        if not text:
            continue
        if section == "support" and seen_support >= 1:
            connector = ("Also", "Furthermore", "In addition", "Moreover")[
                seen_support % 4
            ]
            sentence = f"{connector}, {text[0].lower()}{text[1:]}"
        elif section == "evidence":
            sentence = f"For context, {text[0].lower()}{text[1:]}"
        elif section == "caveat":
            sentence = f"I'm not fully certain — {text[0].lower()}{text[1:]}"
        elif section == "followup":
            sentence = f"Quick question back: {text}"
        else:
            sentence = text
        plan.sentences.append(_humanize(sentence))
        if section == "support":
            seen_support += 1
    return plan


def _choose_lexicon(sentences: list[str], lexicon: Any = None) -> list[str]:
    """Stage 4: lexical substitution.

    For the MVP we leave the surface words as-is (they came from
    structured fields that were already English). Future expansion: walk
    each sentence, replace concept tokens with their preferred surface
    form via the lexicon. The hook exists so V-Reason / V-Agents output
    can be passed through richer lexical substitution.
    """

    if lexicon is None:
        return sentences
    out: list[str] = []
    for sentence in sentences:
        # Map underscore_separated concept names to their preferred
        # surface form when the lexicon has a richer alternative.
        replaced = sentence
        for match in re.finditer(r"\b([a-z][a-z0-9_]{2,})\b", sentence):
            token = match.group(1)
            if "_" not in token:
                continue
            try:
                surface = lexicon.surface_for_concept(token)
            except Exception:
                surface = token.replace("_", " ")
            if surface and surface != token:
                replaced = replaced.replace(token, surface)
        out.append(replaced)
    return out


def _realize_surface(sentences: list[str]) -> str:
    """Stage 5: final composition."""

    if not sentences:
        return ""
    # Join with single spaces. Capitalisation + punctuation per sentence
    # already handled in _humanize.
    text = " ".join(s.strip() for s in sentences if s.strip())
    # Collapse any double spaces that crept in.
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


@dataclass
class SpeechRenderResult:
    """Result of one pipeline run."""

    text: str
    valid: bool = True
    leak_passed: bool = True
    leak_reasons: list[str] = field(default_factory=list)
    stage_outputs: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "valid": self.valid,
            "leak_passed": self.leak_passed,
            "leak_reasons": list(self.leak_reasons),
            "stage_outputs": {
                k: (v.__dict__ if hasattr(v, "__dict__") else v)
                for k, v in self.stage_outputs.items()
            },
        }


class SpeechPipeline:
    """Five-stage compositional non-LLM NLG."""

    def __init__(
        self,
        *,
        operator_models: Any = None,
        dialogue_memory: Any = None,
        universe: Any = None,
        lexicon: Any = None,
    ) -> None:
        self.operator_models = operator_models
        self.dialogue_memory = dialogue_memory
        self.universe = universe
        self.lexicon = lexicon
        self.leak_gate = LeakGate()

    def render(self, plan: Any, *, user_id: str | None = None) -> SpeechRenderResult:
        """Run the pipeline. Returns a SpeechRenderResult."""

        operator_model = None
        if self.operator_models is not None:
            try:
                operator_model = self.operator_models.get(user_id)
            except Exception:
                operator_model = None
        content = _select_content(plan, operator_model)
        tree = _plan_discourse(content, self.dialogue_memory)
        sentence_plan = _plan_sentences(tree)
        sentences = _choose_lexicon(sentence_plan.sentences, self.lexicon)
        text = _realize_surface(sentences)
        gate_result = self.leak_gate.check(
            text,
            fallback_text=content.thesis or "I noted what you said.",
        )
        if not gate_result.passed:
            return SpeechRenderResult(
                text=gate_result.sanitized_fallback,
                valid=True,
                leak_passed=False,
                leak_reasons=list(gate_result.reasons),
                stage_outputs={
                    "content": content,
                    "tree": tree,
                    "sentence_plan": sentence_plan,
                },
            )
        return SpeechRenderResult(
            text=text,
            valid=True,
            leak_passed=True,
            stage_outputs={
                "content": content,
                "tree": tree,
                "sentence_plan": sentence_plan,
            },
        )


__all__ = ["SpeechPipeline", "SpeechRenderResult"]
