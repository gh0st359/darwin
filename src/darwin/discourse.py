from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.experiments import ExperimentProposal
from darwin.retrieval import RetrievalPacket, RetrievedMemory
from darwin.semantics import SemanticFrame


@dataclass
class ResponsePlan:
    mode: str
    intent: str
    thesis: str
    answer_points: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    retrieved_used: list[RetrievedMemory] = field(default_factory=list)
    confidence: float = 0.5
    should_answer_directly: bool = True

    def to_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "intent": self.intent,
            "thesis": self.thesis,
            "answer_points": self.answer_points,
            "evidence": self.evidence,
            "uncertainties": self.uncertainties,
            "clarification_questions": self.clarification_questions,
            "next_actions": self.next_actions,
            "retrieved_used": [item.to_record() for item in self.retrieved_used],
            "confidence": self.confidence,
            "should_answer_directly": self.should_answer_directly,
        }


class DiscoursePlanner:
    """Chooses what Darwin should try to communicate before wording it."""

    def plan(
        self,
        *,
        frame: SemanticFrame,
        packet: RetrievalPacket,
        darwin: Any,
        adapter: Any,
        goal: Any,
        recent_events: list[Any],
    ) -> ResponsePlan:
        report = darwin.self_report()
        focus_terms = self._focus_terms(frame)
        top_items = packet.top(5)

        if frame.needs_clarification and not top_items:
            return self._clarification_plan(frame, packet)

        if frame.speech_act in {"teaching", "goal", "hypothesis", "correction"}:
            return self._learning_plan(frame, packet, report)

        if frame.speech_act == "question":
            return self._question_plan(frame, packet, darwin, adapter, goal, recent_events, focus_terms)

        return self._conversation_plan(frame, packet, report)

    def _question_plan(
        self,
        frame: SemanticFrame,
        packet: RetrievalPacket,
        darwin: Any,
        adapter: Any,
        goal: Any,
        recent_events: list[Any],
        focus_terms: set[str],
    ) -> ResponsePlan:
        top_items = packet.top(5)

        if focus_terms & {"thinking", "mind", "thought", "thoughts", "reason", "reasoning"}:
            report = darwin.self_report()
            recent_points = self._recent_cognition_points(recent_events)
            return ResponsePlan(
                mode="self_report",
                intent="expose current cognition",
                thesis="I should describe the current reasoning process without dumping parser notation.",
                answer_points=[
                    f"I am tracking {report.observations} grounded transitions.",
                    f"My current learning priority is {report.learning_priority}.",
                    *recent_points,
                ],
                evidence=[self._retrieval_evidence_summary(packet)],
                retrieved_used=packet.top(3),
                confidence=max(0.35, frame.confidence),
            )

        if focus_terms & {"belief", "beliefs", "believe", "know", "knowing"}:
            beliefs = darwin.causal_model.beliefs(limit=4)
            points = [
                (
                    f"Under {belief.condition}, {belief.action} changes "
                    f"{belief.variable} as {belief.effect}."
                )
                for belief in beliefs
            ]
            return ResponsePlan(
                mode="belief_answer",
                intent="answer from causal beliefs",
                thesis="The strongest answer should come from learned intervention traces.",
                answer_points=points or ["I do not have enough causal evidence for a strong belief yet."],
                evidence=[f"{belief.samples} samples for {belief.action}" for belief in beliefs],
                retrieved_used=packet.top(4),
                confidence=0.6 if beliefs else 0.25,
            )

        if focus_terms & {"experiment", "experiments", "test", "uncertain", "uncertainty"}:
            proposals = darwin.experiment_engine.propose(
                adapter.observe(),
                adapter.possible_actions(),
                goal=goal,
                limit=2,
            )
            return self._experiment_plan(proposals, packet, frame)

        if focus_terms & {"goal", "goals", "value", "values", "important", "learned"}:
            points = []
            if packet.active_goals:
                points.append(
                    "Active goals: "
                    + ", ".join(f"{key}={value!r}" for key, value in packet.active_goals.items())
                )
            if packet.values:
                points.append(
                    "Strong values: "
                    + ", ".join(f"{key}:{value}" for key, value in list(packet.values.items())[:5])
                )
            if not points:
                points.append("I have not consolidated strong goals or values yet.")
            return ResponsePlan(
                mode="memory_summary",
                intent="summarize goals and values from semantic memory",
                thesis="The answer should come from accumulated semantic memory, not from the current sentence alone.",
                answer_points=points,
                evidence=[self._retrieval_evidence_summary(packet)],
                retrieved_used=packet.top(5),
                confidence=0.55 if packet.active_goals or packet.values else 0.28,
            )

        if focus_terms & {"unknown", "terms", "term", "words"}:
            unknowns = list(packet.unknown_terms.items())[:8]
            points = [
                f"{term} has appeared {count} times without enough grounding."
                for term, count in unknowns
            ]
            return ResponsePlan(
                mode="unknown_terms",
                intent="identify terms that need grounding",
                thesis="I should turn repeated unknown terms into learning targets.",
                answer_points=points or ["I do not have a stable unknown-term target yet."],
                evidence=[self._retrieval_evidence_summary(packet)],
                retrieved_used=packet.top(5),
                confidence=0.5 if points else 0.25,
            )

        if top_items:
            answer_points = [item.content for item in top_items[:4]]
            if "why" in focus_terms:
                answer_points.insert(0, self._reason_from_memory(top_items[0].content))
            return ResponsePlan(
                mode="answer",
                intent="answer using retrieved semantic memory",
                thesis="The answer should be built from the retrieved meaning and its consequences.",
                answer_points=answer_points,
                evidence=[self._evidence_phrase(item) for item in top_items[:4]],
                retrieved_used=top_items[:4],
                confidence=min(0.8, 0.25 + sum(item.score for item in top_items[:3]) / 5.0),
            )

        return self._clarification_plan(frame, packet)

    def _learning_plan(self, frame: SemanticFrame, packet: RetrievalPacket, report: Any) -> ResponsePlan:
        points: list[str] = []
        if frame.propositions:
            points.extend(
                f"{item.subject} {item.relation} {item.object}" for item in frame.propositions[:4]
            )
        if frame.goals:
            points.append(
                "goal update: " + ", ".join(f"{key}={value!r}" for key, value in frame.goals.items())
            )
        if frame.values:
            points.append(
                "value update: " + ", ".join(f"{key}:{value:.2f}" for key, value in frame.values.items())
            )
        if frame.corrections:
            points.extend(f"correction: {item}" for item in frame.corrections[:3])
        if not points:
            points.append("I can store this as conversational evidence, but I need more grounding.")

        return ResponsePlan(
            mode="learn",
            intent="acknowledge and organize new semantic material",
            thesis="I should absorb the meaning and connect it to future interpretation.",
            answer_points=points,
            evidence=[self._retrieval_evidence_summary(packet)],
            uncertainties=[
                f"unknown terms: {', '.join(frame.unknown_terms[:5])}"
                if frame.unknown_terms
                else ""
            ],
            next_actions=[report.learning_priority],
            retrieved_used=packet.top(5),
            confidence=max(0.35, frame.confidence),
        )

    def _conversation_plan(self, frame: SemanticFrame, packet: RetrievalPacket, report: Any) -> ResponsePlan:
        items = packet.top(4)
        points = [item.content for item in items] or [
            f"I understood this as a {frame.speech_act} about {frame.topic}."
        ]
        return ResponsePlan(
            mode="conversation",
            intent="respond from memory while preserving uncertainty",
            thesis="I should connect the current message to stored meanings and current learning pressure.",
            answer_points=points,
            evidence=[self._retrieval_evidence_summary(packet)],
            uncertainties=[
                f"low semantic confidence {frame.confidence:.2f}"
                if frame.confidence < 0.45
                else ""
            ],
            next_actions=[report.learning_priority],
            retrieved_used=items,
            confidence=max(0.3, min(0.75, frame.confidence + 0.15)),
        )

    def _clarification_plan(self, frame: SemanticFrame, packet: RetrievalPacket) -> ResponsePlan:
        unknown = frame.unknown_terms[:5] or list(packet.unknown_terms)[:5]
        question = (
            "Can you ground "
            + ", ".join(unknown)
            + " as a claim, goal, correction, or cause/effect relation?"
            if unknown
            else "Can you say the core idea as a claim, goal, correction, or cause/effect relation?"
        )
        return ResponsePlan(
            mode="clarify",
            intent="ask for grounding instead of pretending certainty",
            thesis="I do not have enough grounded structure to answer cleanly yet.",
            answer_points=[
                f"I parsed the message as {frame.speech_act} about {frame.topic}.",
                f"My semantic confidence is {frame.confidence:.2f}.",
            ],
            uncertainties=[f"unresolved terms: {', '.join(unknown)}" if unknown else "insufficient grounding"],
            clarification_questions=[question],
            retrieved_used=packet.top(3),
            confidence=frame.confidence,
            should_answer_directly=False,
        )

    def _experiment_plan(
        self,
        proposals: list[ExperimentProposal],
        packet: RetrievalPacket,
        frame: SemanticFrame,
    ) -> ResponsePlan:
        if not proposals:
            return ResponsePlan(
                mode="experiment",
                intent="explain lack of available experiment",
                thesis="I do not have a useful experiment proposal from the current state.",
            answer_points=[],
            retrieved_used=packet.top(3),
            confidence=0.25,
        )
        primary = proposals[0]
        points = [
            f"test {primary.action.name}: {primary.question}",
            f"prediction: {primary.predicted_state}",
            f"expected reward: {primary.expected_reward:.2f}; uncertainty: {primary.uncertainty:.2f}",
        ]
        if len(proposals) > 1:
            points.append(f"compare against {proposals[1].action.name}")
        return ResponsePlan(
            mode="experiment",
            intent="propose an uncertainty-reducing intervention",
            thesis="The best next experiment is the one that buys information about uncertain consequences.",
            answer_points=points,
            evidence=[primary.rationale, self._retrieval_evidence_summary(packet)],
            retrieved_used=packet.top(3),
            confidence=max(0.35, frame.confidence),
        )

    def _has_strong_memory(self, items: list[RetrievedMemory]) -> bool:
        return bool(items and items[0].score >= 0.55)

    def _focus_terms(self, frame: SemanticFrame) -> set[str]:
        terms = {term.lower() for term in frame.tokens if len(term) > 2}
        terms.update(grounding.text.lower() for grounding in frame.groundings)
        return terms

    def _reason_from_memory(self, content: str) -> str:
        if not content:
            return "The reason is not yet grounded in a strong memory."
        return (
            f"I can ground the reason in this learned relation: {content}. "
            "That matters because surface repetition can preserve wording while losing the "
            "connection between words, meaning, and consequence"
        )

    def _evidence_phrase(self, item: RetrievedMemory) -> str:
        source = item.kind.replace("_", " ")
        title = item.title.replace("/", " about ")
        if item.score >= 0.55:
            strength = "strong"
        elif item.score >= 0.35:
            strength = "partial"
        else:
            strength = "weak"
        return f"{strength} {source} memory: {title}"

    def _retrieval_evidence_summary(self, packet: RetrievalPacket) -> str:
        top_items = packet.top(3)
        if not top_items:
            return "no older memory was relevant enough to retrieve"
        strongest = self._evidence_phrase(top_items[0])
        if len(top_items) == 1:
            return strongest
        return f"{strongest}; {len(top_items) - 1} additional memory links were considered"

    def _recent_cognition_points(self, recent_events: list[Any]) -> list[str]:
        for event in reversed(recent_events):
            if getattr(event, "kind", "") != "thought":
                continue
            trace = getattr(event, "payload", {}).get("trace", {})
            steps = trace.get("steps", [])
            if not steps:
                continue
            points: list[str] = []
            final_mode = trace.get("final_mode")
            final_confidence = trace.get("final_confidence")
            if final_mode:
                confidence_text = (
                    f" at confidence {float(final_confidence):.2f}"
                    if isinstance(final_confidence, (int, float))
                    else ""
                )
                points.append(f"The previous reasoning cycle ended in {final_mode}{confidence_text}.")
            names = [str(step.get("name", "")).replace("_", " ") for step in steps if step.get("name")]
            if names:
                points.append("Its main stages were " + ", ".join(names[:5]) + ".")
            return points
        return ["I do not have a recent reasoning trace to summarize yet."]
