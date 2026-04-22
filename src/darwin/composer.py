from __future__ import annotations

import hashlib
from typing import Iterable

from darwin.discourse import ResponsePlan
from darwin.semantics import SemanticFrame
from darwin.thought import ThoughtTrace


class NaturalLanguageComposer:
    """Composes responses from a response plan, not from input-output templates."""

    def compose(self, plan: ResponsePlan, frame: SemanticFrame, trace: ThoughtTrace) -> str:
        sections: list[str] = []
        opener = self._opener(plan, frame)
        if opener:
            sections.append(opener)
        sections.extend(self._body_sentences(plan))
        sections.extend(self._uncertainty_sentences(plan))
        sections.extend(self._next_step_sentences(plan))
        if plan.clarification_questions:
            sections.append(plan.clarification_questions[0])
        text = " ".join(self._clean(sentence) for sentence in sections if sentence and sentence.strip())
        return self._smooth(text)

    def _opener(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        variants = {
            "clarify": [
                "I do not want to pretend I understand that fully yet.",
                "I can see the shape of the message, but the grounding is still thin.",
                "I need to slow down here because the meaning is not grounded enough.",
            ],
            "learn": [
                "I am folding that into semantic memory as structured meaning.",
                "I am treating that as new material for my language model of the world.",
                "I am storing the meaning of that rather than just the sentence.",
            ],
            "answer": [
                "The strongest answer I can ground comes from memory.",
                "I can answer that by pulling on stored meanings rather than guessing.",
                "The best grounded thread I can follow is this.",
            ],
            "memory_summary": [
                "From my semantic memory, the center of gravity is becoming clearer.",
                "The goals and values I can actually retrieve are these.",
                "I can summarize the learned pressure from memory.",
            ],
            "self_report": [
                "Inside the current response cycle, I am tracking a few things.",
                "My current internal state points to this.",
                "The live reasoning thread is not polished, but it is inspectable.",
            ],
            "experiment": [
                "The action-oriented answer is an experiment.",
                "I can ground that in an intervention rather than only in words.",
                "The next useful move is to test a prediction.",
            ],
            "unknown_terms": [
                "The terms that need grounding are starting to stand out.",
                "I can turn repeated unknowns into learning targets.",
                "The unresolved language pressure is concentrated in these terms.",
            ],
            "belief_answer": [
                "The beliefs I can defend are the ones tied to intervention traces.",
                "From causal memory, the strongest beliefs are these.",
                "The grounded beliefs I can report are limited but explicit.",
            ],
        }
        choices = variants.get(plan.mode, ["I am connecting this to what I can retrieve."])
        return self._choose(choices, frame.original_text + plan.mode)

    def _body_sentences(self, plan: ResponsePlan) -> list[str]:
        sentences: list[str] = []
        if plan.thesis:
            sentences.append(plan.thesis)
        for point in plan.answer_points[:5]:
            if not point:
                continue
            sentences.append(self._point_sentence(point))
        if plan.evidence:
            evidence = "; ".join(item for item in plan.evidence[:3] if item)
            if evidence:
                sentences.append(self._evidence_sentence(evidence))
        return sentences

    def _uncertainty_sentences(self, plan: ResponsePlan) -> list[str]:
        uncertainties = [item for item in plan.uncertainties if item]
        if not uncertainties and plan.confidence >= 0.45:
            return []
        if uncertainties:
            return [f"My uncertainty is {', '.join(uncertainties[:3])}."]
        return [f"My confidence is limited at {plan.confidence:.2f}, so I should not overstate this."]

    def _next_step_sentences(self, plan: ResponsePlan) -> list[str]:
        actions = [item for item in plan.next_actions if item]
        if not actions:
            return []
        return [f"The next pressure on my learning is {actions[0].replace('_', ' ')}."]

    def _point_sentence(self, point: str) -> str:
        stripped = point.strip()
        if not stripped:
            return ""
        if stripped.endswith("."):
            return stripped
        if ":" in stripped and len(stripped.split()) < 16:
            return stripped + "."
        return stripped[0].upper() + stripped[1:] + "."

    def _choose(self, choices: list[str], seed_text: str) -> str:
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(choices)
        return choices[index]

    def _evidence_sentence(self, evidence: str) -> str:
        if evidence.startswith("no older memory"):
            return "I did not retrieve older memory that was relevant enough to use."
        return f"The evidence I used was {evidence}."

    def _clean(self, sentence: str) -> str:
        sentence = " ".join(sentence.split())
        for marker in ["act=", "topic=", "intent=", "source=", "confidence="]:
            sentence = sentence.replace(marker, marker[:-1] + " ")
        return sentence

    def _smooth(self, text: str) -> str:
        text = text.replace("..", ".")
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        return text.strip()
