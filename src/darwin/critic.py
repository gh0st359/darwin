from __future__ import annotations

from dataclasses import dataclass, field

from darwin.discourse import ResponsePlan
from darwin.retrieval import RetrievalPacket
from darwin.semantics import SemanticFrame


@dataclass
class Critique:
    passed: bool
    issues: list[str] = field(default_factory=list)
    revisions: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "revisions": self.revisions,
        }

    def summary(self) -> str:
        if self.passed:
            return "passed"
        return "; ".join(self.issues)


class ResponseCritic:
    """Checks whether a planned response respects Darwin's own constraints."""

    def evaluate(self, plan: ResponsePlan, draft: str, frame: SemanticFrame, packet: RetrievalPacket) -> Critique:
        issues: list[str] = []
        revisions: list[str] = []
        lower = draft.lower()

        notation_markers = [
            "act=",
            "topic=",
            "intent=",
            "source=",
            "confidence=",
            "groundings=",
            "propositions=",
            "score=",
            "semantic:",
        ]
        if any(marker in draft for marker in notation_markers):
            issues.append("response leaked parser notation")
            revisions.append("replace parser notation with natural language")

        if frame.speech_act == "question" and plan.mode not in {"clarify", "self_report"}:
            if not plan.answer_points:
                issues.append("question did not receive answer points")
                revisions.append("use retrieved memory or ask a targeted clarification")

        if packet.items and not plan.retrieved_used and plan.mode != "clarify":
            issues.append("retrieved memory was ignored")
            revisions.append("ground the response in retrieved memory")

        if plan.confidence < 0.4 and not plan.uncertainties and plan.mode != "clarify":
            issues.append("low confidence was not disclosed")
            revisions.append("add uncertainty before speaking")

        if any(word in lower for word in ["certainly", "definitely", "obviously"]) and plan.confidence < 0.7:
            issues.append("overconfident language with weak support")
            revisions.append("soften certainty")

        if len(draft.split()) < 8 and plan.mode != "clarify":
            issues.append("response too thin")
            revisions.append("include a grounded point or evidence")

        return Critique(passed=not issues, issues=issues, revisions=revisions)

    def revise(self, plan: ResponsePlan, critique: Critique, frame: SemanticFrame, packet: RetrievalPacket) -> ResponsePlan:
        revised = ResponsePlan(
            mode=plan.mode,
            intent=plan.intent,
            thesis=plan.thesis,
            answer_points=list(plan.answer_points),
            evidence=list(plan.evidence),
            uncertainties=list(plan.uncertainties),
            clarification_questions=list(plan.clarification_questions),
            next_actions=list(plan.next_actions),
            retrieved_used=list(plan.retrieved_used),
            confidence=plan.confidence,
            should_answer_directly=plan.should_answer_directly,
        )

        if "response leaked parser notation" in critique.issues:
            revised.thesis = revised.thesis.replace("act=", "speech act ").replace("topic=", "topic ")

        if "retrieved memory was ignored" in critique.issues:
            revised.retrieved_used = packet.top(3)
            revised.answer_points.extend(item.content for item in packet.top(2))

        if "low confidence was not disclosed" in critique.issues:
            revised.uncertainties.append(f"semantic confidence {frame.confidence:.2f}")

        if "question did not receive answer points" in critique.issues:
            if packet.items:
                revised.answer_points.append(packet.items[0].content)
                revised.retrieved_used.append(packet.items[0])
            else:
                revised.mode = "clarify"
                revised.should_answer_directly = False
                revised.clarification_questions.append(
                    "Can you give me one grounded claim or goal that should anchor the answer?"
                )

        if "response too thin" in critique.issues and packet.items:
            revised.answer_points.append(packet.items[0].content)
            revised.retrieved_used.append(packet.items[0])

        return revised
