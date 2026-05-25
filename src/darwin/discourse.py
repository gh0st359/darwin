from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from darwin.experiments import ExperimentProposal
from darwin.retrieval import RetrievalPacket, RetrievedMemory
from darwin.semantics import SemanticFrame


@dataclass
class CausalClaim:
    """A single causal assertion that the renderer must preserve verbatim."""

    action: str
    variable: str
    effect: str
    confidence: float
    samples: int
    condition: str = "always"

    def to_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "variable": self.variable,
            "effect": self.effect,
            "confidence": self.confidence,
            "samples": self.samples,
            "condition": self.condition,
        }


@dataclass
class ReferencedExperience:
    """A memory the response is grounded in."""

    kind: str
    title: str
    summary: str
    score: float
    timestamp: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "score": self.score,
            "timestamp": self.timestamp,
        }


@dataclass
class UncertaintyLevel:
    """An explicit uncertainty marker. The renderer MUST surface this."""

    target: str
    level: float
    reason: str = ""

    def to_record(self) -> dict[str, Any]:
        return {"target": self.target, "level": self.level, "reason": self.reason}


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
    causal_claims: list[CausalClaim] = field(default_factory=list)
    referenced_experiences: list[ReferencedExperience] = field(default_factory=list)
    uncertainty_levels: list[UncertaintyLevel] = field(default_factory=list)
    self_reflection: list[str] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tone: str = "neutral"
    target_length: str = "medium"

    def to_record(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
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
            "causal_claims": [item.to_record() for item in self.causal_claims],
            "referenced_experiences": [item.to_record() for item in self.referenced_experiences],
            "uncertainty_levels": [item.to_record() for item in self.uncertainty_levels],
            "self_reflection": self.self_reflection,
            "tone": self.tone,
            "target_length": self.target_length,
        }

    def to_dlm_payload(self) -> dict[str, Any]:
        """A strictly-shaped payload for the Darwin Language Module to render.

        Only this view is shown to an external renderer. No Darwin internals
        leak through, and every claim, uncertainty, and reference is explicit
        so the renderer can be validated against it.
        """

        return {
            "mode": self.mode,
            "intent": self.intent,
            "thesis": self.thesis,
            "answer_points": list(self.answer_points),
            "clarification_questions": list(self.clarification_questions),
            "next_actions": list(self.next_actions),
            "causal_claims": [claim.to_record() for claim in self.causal_claims],
            "referenced_experiences": [item.to_record() for item in self.referenced_experiences],
            "uncertainty_levels": [item.to_record() for item in self.uncertainty_levels],
            "self_reflection": list(self.self_reflection),
            "confidence": self.confidence,
            "tone": self.tone,
            "target_length": self.target_length,
        }


class DiscoursePlanner:
    """Chooses what Darwin should try to communicate before wording it."""

    SOCIAL_MODES = {"greeting", "farewell", "small_talk", "identity"}
    DOMAIN_TERMS = {
        "math": {"math", "number", "numbers", "arithmetic", "addition", "subtract", "subtraction", "multiply", "zero"},
        "space": {"space", "spatial", "block", "blocks", "motion", "move", "moving", "position", "push", "lift", "drop", "gravity"},
        "room": {"room", "curtain", "curtains", "light", "bright", "brightness", "fuse", "switch", "circuit", "daylight"},
        "time": {"time", "wait", "tick", "ticks", "pause"},
    }

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

        meta_focus_terms = {
            "belief", "beliefs", "believe", "know", "knowing", "knew",
            "thinking", "mind", "thought", "thoughts", "reason", "reasoning",
            "experiment", "experiments", "test", "uncertain", "uncertainty",
            "goal", "goals", "value", "values", "learned",
            "unknown", "terms", "term", "words",
        }
        is_meta_question = (
            frame.speech_act == "question"
            and bool(focus_terms & meta_focus_terms)
        )

        if frame.speech_act == "greeting":
            plan = self._greeting_plan(frame)
        elif frame.speech_act == "farewell":
            plan = self._farewell_plan(frame)
        elif frame.speech_act == "small_talk":
            plan = self._small_talk_plan(frame)
        elif frame.speech_act == "identity_question":
            plan = self._identity_plan(frame, darwin, report)
        elif is_meta_question:
            # A question about Darwin's own beliefs/knowledge/experiments
            # should answer from internal state even if the surface
            # phrasing was sparse.
            plan = self._question_plan(frame, packet, darwin, adapter, goal, recent_events, focus_terms)
        elif frame.speech_act in {"teaching", "goal", "hypothesis", "correction"}:
            # Explicit teaching never gets bumped to clarify — if the
            # user said "learn this", absorb it.
            plan = self._learning_plan(frame, packet, report)
        elif frame.needs_clarification and not top_items:
            plan = self._clarification_plan(frame, packet)
        elif frame.speech_act == "question":
            plan = self._question_plan(frame, packet, darwin, adapter, goal, recent_events, focus_terms)
        else:
            plan = self._conversation_plan(frame, packet, report)

        return self._enrich_plan(plan, frame, packet, darwin, report)

    def _greeting_plan(self, frame: SemanticFrame) -> ResponsePlan:
        # Cognitive layer only labels the intent. The renderer (composer
        # fallback or DLM) decides what English words to use — that is
        # the language layer's job, not the mind's.
        return ResponsePlan(
            mode="greeting",
            intent="acknowledge the user has just greeted me",
            thesis="A greeting from the user is a contact-establishment signal; respond at the same level.",
            answer_points=[],
            confidence=0.9,
            tone="neutral",
            target_length="short",
        )

    def _farewell_plan(self, frame: SemanticFrame) -> ResponsePlan:
        return ResponsePlan(
            mode="farewell",
            intent="acknowledge the user is ending the conversation",
            thesis="A farewell is a contact-termination signal; respond at the same level.",
            answer_points=[],
            confidence=0.9,
            tone="neutral",
            target_length="short",
        )

    def _small_talk_plan(self, frame: SemanticFrame) -> ResponsePlan:
        # Classify the SUB-INTENT structurally; the renderer maps each
        # sub-intent to natural English. No response text is fixed here.
        text = frame.normalized_text
        if "thank" in text or text.strip(" .!?,") in {"thanks", "thx", "ty"}:
            sub_intent = "acknowledge_gratitude"
        elif any(p in text for p in ("how are you", "how's it going", "you good", "are you good", "how do you feel", "how are things")):
            sub_intent = "report_state_briefly"
        elif any(p in text for p in ("you there", "are you there", "anyone home")):
            sub_intent = "confirm_presence"
        else:
            sub_intent = "minimal_acknowledgement"
        return ResponsePlan(
            mode="small_talk",
            intent=sub_intent,
            thesis="A short social signal deserves a short response at the same level.",
            answer_points=[],
            confidence=0.85,
            tone="neutral",
            target_length="short",
        )

    def _identity_plan(self, frame: SemanticFrame, darwin: Any, report: Any) -> ResponsePlan:
        # Identity is derived from CURRENT INTERNAL STATE, not a fixed
        # bio string. The renderer receives the structured facts via
        # self_reflection and composes an honest self-description.
        beliefs = darwin.causal_model.beliefs(limit=3)
        strongest = beliefs[0].action if beliefs else None
        reflection = [
            f"name: Darwin",
            f"observations: {report.observations}",
            f"known_actions: {report.known_actions}",
            f"known_variables: {report.known_variables}",
            f"strongest_action: {strongest or 'none yet'}",
            f"learning_priority: {report.learning_priority}",
        ]
        return ResponsePlan(
            mode="identity",
            intent="describe self from current internal state",
            thesis="Describe what I am using actual observation counts and current learning posture.",
            answer_points=[],
            self_reflection=reflection,
            confidence=0.7,
            tone="neutral",
            target_length="medium",
        )

    def _enrich_plan(
        self,
        plan: ResponsePlan,
        frame: SemanticFrame,
        packet: RetrievalPacket,
        darwin: Any,
        report: Any,
    ) -> ResponsePlan:
        # Social modes are intentionally NOT enriched with causal claims,
        # retrieved memories, or surfaced uncertainty dumps. A "Hello"
        # should produce a "Hi", not a 5-belief memory report. But we
        # DO preserve any self_reflection the planner already attached
        # (identity mode needs it; small_talk's report_state_briefly
        # uses it). Likewise, attach self_reflection to small_talk's
        # state-report sub-intent if the planner did not.
        if plan.mode in self.SOCIAL_MODES:
            plan.causal_claims = []
            plan.referenced_experiences = []
            plan.uncertainty_levels = []
            if plan.mode == "small_talk" and plan.intent == "report_state_briefly" and not plan.self_reflection:
                plan.self_reflection = [
                    f"observations: {report.observations}",
                    f"learning_priority: {report.learning_priority}",
                ]
            return plan

        # Only modes that actually consult Darwin's causal beliefs get
        # them on the plan. Casual conversation does not.
        claim_modes = {"belief_answer", "answer", "experiment", "self_report", "learn"}
        causal_claims: list[CausalClaim] = []
        if plan.mode in claim_modes:
            domain = self._domain_from_plan_or_frame(plan, frame)
            beliefs = (
                self._beliefs_for_domain(darwin, domain, limit=4)
                if domain
                else darwin.causal_model.beliefs(limit=4)
            )
            for belief in beliefs:
                causal_claims.append(
                    CausalClaim(
                        action=belief.action,
                        variable=belief.variable,
                        effect=belief.effect,
                        confidence=belief.confidence,
                        samples=belief.samples,
                        condition=belief.condition,
                    )
                )
        plan.causal_claims = causal_claims

        plan.referenced_experiences = [
            ReferencedExperience(
                kind=item.kind,
                title=item.title,
                summary=item.content,
                score=item.score,
            )
            for item in plan.retrieved_used[:4]
        ]

        levels: list[UncertaintyLevel] = []
        if frame.confidence < 0.45 and plan.mode not in {"conversation", "clarify"}:
            levels.append(
                UncertaintyLevel(
                    target="interpretation",
                    level=1.0 - frame.confidence,
                    reason="semantic parse is weak",
                )
            )
        if plan.confidence < 0.45 and plan.mode in claim_modes:
            levels.append(
                UncertaintyLevel(
                    target="answer",
                    level=1.0 - plan.confidence,
                    reason="grounded memory was thin",
                )
            )
        # Only the *single* lowest-confidence relevant claim surfaces as
        # an uncertainty line, not every belief under 0.6.
        weak_claims = [claim for claim in causal_claims if claim.confidence < 0.55]
        if weak_claims and plan.mode in claim_modes:
            claim = weak_claims[0]
            levels.append(
                UncertaintyLevel(
                    target=f"belief:{claim.action}->{claim.variable}",
                    level=1.0 - claim.confidence,
                    reason=f"only {claim.samples} samples",
                )
            )
        plan.uncertainty_levels = levels

        plan.self_reflection = (
            [f"current learning priority: {report.learning_priority}"]
            if plan.mode in claim_modes
            else []
        )
        if plan.confidence >= 0.7:
            plan.tone = "confident"
        elif plan.confidence <= 0.4:
            plan.tone = "tentative"
        else:
            plan.tone = "neutral"
        if len(plan.answer_points) <= 1:
            plan.target_length = "short"
        elif len(plan.answer_points) >= 4:
            plan.target_length = "long"
        else:
            plan.target_length = "medium"
        return plan

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
        domain = self._domain_from_focus_terms(focus_terms)

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

        if domain and focus_terms & {"belief", "beliefs", "believe", "know", "knowing", "learned", "learn"}:
            beliefs = self._beliefs_for_domain(darwin, domain, limit=4)
            # Do NOT pre-render beliefs into answer_points; the composer
            # will build prose from the causal_claims that _enrich_plan
            # attaches. That keeps belief_answer output free of robotic
            # "Under always, X changes Y as None -> True" lines.
            return ResponsePlan(
                mode="belief_answer",
                intent=f"answer from {domain} causal beliefs",
                thesis="The strongest answer should come from learned intervention traces.",
                answer_points=[] if beliefs else ["I don't have enough direct experience yet to say."],
                evidence=[f"domain::{domain}"],
                retrieved_used=packet.top(2),
                confidence=0.6 if beliefs else 0.25,
                target_length="medium",
            )

        if focus_terms & {"belief", "beliefs", "believe", "know", "knowing", "learned", "learn"}:
            beliefs = darwin.causal_model.beliefs(limit=4)
            return ResponsePlan(
                mode="belief_answer",
                intent="answer from causal beliefs",
                thesis="The strongest answer should come from learned intervention traces.",
                answer_points=[] if beliefs else ["I don't have enough direct experience yet to say."],
                retrieved_used=packet.top(2),
                confidence=0.6 if beliefs else 0.25,
                target_length="medium",
            )

        if focus_terms & {"experiment", "experiments", "test", "uncertain", "uncertainty"}:
            proposals = darwin.experiment_engine.propose(
                adapter.observe(),
                adapter.possible_actions(),
                goal=goal,
                limit=2,
                variable_filter_for_action=self._variable_filter_for_adapter_action(adapter),
            )
            return self._experiment_plan(proposals, packet, frame)

        if focus_terms & {"goal", "goals", "value", "values", "important"}:
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
        # Pass STRUCTURED signals through evidence (snippet of what the
        # user said, plus the parsed propositions / goals / corrections).
        # The renderer composes the acknowledgement sentences. No
        # pre-formed English lives in the cognitive layer.
        snippet = self._teaching_snippet(frame)
        evidence: list[str] = [f"snippet::{snippet}"]
        if frame.propositions:
            prop = frame.propositions[0]
            evidence.append(
                f"proposition::{prop.subject.strip()}|{prop.relation}|{prop.object.strip()}"
            )
        for key, value in list(frame.goals.items())[:3]:
            evidence.append(f"goal::{key}|{value!r}")
        if frame.corrections:
            evidence.append(f"correction::{frame.corrections[0].strip()}")
        return ResponsePlan(
            mode="learn",
            intent="acknowledge what was just taught",
            thesis="Confirm what I've stored, derived from what the user actually said.",
            answer_points=[],
            evidence=evidence,
            retrieved_used=[],
            confidence=max(0.5, frame.confidence + 0.1),
            tone="neutral",
            target_length="medium",
        )

    def _teaching_snippet(self, frame: SemanticFrame) -> str:
        text = frame.original_text.strip().rstrip(".!?")
        for cue in ("learn this:", "learn this,", "remember that", "remember:", "teach me that"):
            idx = text.lower().find(cue)
            if idx >= 0:
                text = text[idx + len(cue):].strip(" :,-")
                break
        if text.lower().startswith("darwin,"):
            text = text[len("darwin,"):].strip()
        # Cap length to keep the echo conversational.
        if len(text) > 140:
            text = text[:137].rstrip() + "..."
        return text

    def _goals_as_prose(self, goals: dict[str, Any]) -> str:
        pieces: list[str] = []
        for key, value in goals.items():
            human_key = key.replace("_", " ")
            if isinstance(value, bool):
                pieces.append(f"keep {human_key} {'true' if value else 'false'}")
            elif value == "increase":
                pieces.append(f"increase {human_key}")
            elif value == "attend":
                pieces.append(f"pay attention to {human_key}")
            else:
                pieces.append(f"target {human_key} = {value!r}")
        if not pieces:
            return ""
        return "Stored goal: " + ", ".join(pieces) + "."

    def _conversation_plan(self, frame: SemanticFrame, packet: RetrievalPacket, report: Any) -> ResponsePlan:
        # A general statement is not a request for a memory dump. If the
        # user said something with no clear question, no clear teaching
        # signal, and we have nothing strongly relevant retrieved, just
        # acknowledge briefly and invite them to continue.
        strong_items = [item for item in packet.top(4) if item.score >= 0.45]
        if not strong_items:
            return ResponsePlan(
                mode="conversation",
                intent="acknowledge briefly and invite the user to continue",
                thesis="A short acknowledgement is more honest than a memory dump.",
                answer_points=["Got it.", "What would you like to talk about?"],
                retrieved_used=[],
                confidence=max(0.4, min(0.7, frame.confidence + 0.2)),
                tone="neutral",
                target_length="short",
            )

        # Only when retrieval was actually strong do we connect what the
        # user said to existing memory — and we cap to ONE point so the
        # reply stays conversational, not encyclopedic.
        top = strong_items[0]
        return ResponsePlan(
            mode="conversation",
            intent="connect the user's message to one relevant memory",
            thesis="The reply should be a short connection, not a memory listing.",
            answer_points=[f"This reminds me of {top.content}."],
            retrieved_used=[top],
            confidence=max(0.4, min(0.75, frame.confidence + 0.2)),
            tone="neutral",
            target_length="short",
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
        for grounding in frame.groundings:
            terms.add(grounding.name.lower())
            terms.update(
                part
                for part in grounding.name.replace("/", " ").replace(".", " ").replace("_", " ").lower().split()
                if len(part) > 2
            )
        return terms

    def _domain_from_focus_terms(self, focus_terms: set[str]) -> str | None:
        for domain, terms in self.DOMAIN_TERMS.items():
            if domain in focus_terms or focus_terms & terms:
                return domain
        return None

    def _domain_from_plan_or_frame(self, plan: ResponsePlan, frame: SemanticFrame) -> str | None:
        for item in plan.evidence:
            if item.startswith("domain::"):
                domain = item.split("::", 1)[1].strip()
                if domain:
                    return domain
        return self._domain_from_focus_terms(self._focus_terms(frame))

    def _beliefs_for_domain(self, darwin: Any, domain: str, limit: int = 4) -> list[Any]:
        prefix = f"{domain}."
        action_prefix = f"{domain}/"
        beliefs = []
        seen: set[tuple[str, str]] = set()
        for belief in darwin.causal_model.beliefs(limit=80):
            if not (belief.action.startswith(action_prefix) or belief.variable.startswith(prefix)):
                continue
            key = (belief.action, belief.variable)
            if key in seen:
                continue
            seen.add(key)
            beliefs.append(belief)
            if len(beliefs) >= limit:
                break
        return beliefs

    def _variable_filter_for_adapter_action(self, adapter: Any) -> Any:
        variable_finder = getattr(adapter, "variables_for_domain", None)
        if not callable(variable_finder):
            return None

        def include(action: Any, variable: str) -> bool:
            domain = str(getattr(action, "metadata", {}).get("domain", ""))
            if not domain:
                return True
            variables = set(variable_finder(domain))
            return variable in variables if variables else True

        return include

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
