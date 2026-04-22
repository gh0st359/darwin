from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from darwin.embodiment import ConversationAdapter, EnvironmentAdapter
from darwin.semantics import SemanticFrame
from darwin.types import Goal


@dataclass
class LanguageState:
    observations: int
    maturity: float
    topic: str
    intent: str
    semantic_confidence: float


class LanguageCortex:
    """Grounded natural-language surface for Darwin's internal state.

    This is deliberately not an LLM. It speaks from Darwin's own state: memory,
    beliefs, concepts, plans, experiments, and recent runtime events. As Darwin
    gets more experience, responses become more specific because there is more
    internal structure to verbalize.
    """

    def respond(
        self,
        message: str,
        darwin: Any,
        adapter: EnvironmentAdapter,
        goal: Goal,
        recent_events: Iterable[Any],
        conversation: ConversationAdapter,
        semantic_frame: SemanticFrame | None = None,
    ) -> str:
        signal = conversation.signal(message)
        frame = semantic_frame or darwin.interpret_language(message, source="user", persist=False)
        report = darwin.self_report()
        state = LanguageState(
            observations=report.observations,
            maturity=min(1.0, report.observations / 50.0),
            topic=frame.topic,
            intent=frame.intent,
            semantic_confidence=frame.confidence,
        )
        lowered = message.lower().strip()

        if self._asks_about_meaning(lowered):
            return self._meaning_response(frame, darwin)

        if self._asks_about_thinking(lowered):
            return self._thinking_response(report, recent_events, state, frame)

        if self._asks_about_beliefs(lowered):
            return self._belief_response(darwin, state)

        if self._asks_about_experiments(lowered):
            return self._experiment_response(darwin, adapter, goal, state)

        if frame.speech_act in {"teaching", "correction", "goal", "hypothesis"} or self._is_teaching(lowered, frame):
            return self._teaching_response(report, frame, state)

        if self._asks_about_plans(lowered):
            return self._plan_response(darwin, adapter, goal, state)

        if frame.needs_clarification and frame.source == "user":
            return self._clarification_response(frame, report)

        return self._open_chat_response(darwin, report, recent_events, frame, state)

    def _thinking_response(
        self,
        report: Any,
        recent_events: Iterable[Any],
        state: LanguageState,
        frame: SemanticFrame,
    ) -> str:
        events = list(recent_events)[-3:]
        if events:
            event_text = " ".join(f"{event.kind}: {event.content}" for event in events)
        else:
            event_text = "I have not produced many private cognition events yet."

        return (
            f"I parsed your question as {frame.speech_act}/{frame.topic} with "
            f"{frame.confidence:.2f} confidence. "
            f"I am holding {report.observations} experiences in working memory. "
            f"The live thread in my head is: {event_text} "
            f"My current learning pressure is {self._natural_priority(report.learning_priority)}."
        )

    def _belief_response(self, darwin: Any, state: LanguageState) -> str:
        beliefs = darwin.causal_model.beliefs(limit=4)
        if not beliefs:
            return (
                "I do not have strong causal beliefs yet. "
                "I need more interventions before I can honestly claim a pattern."
            )

        parts = [
            (
                f"when {belief.condition}, {belief.action} tends to move "
                f"{belief.variable} as {belief.effect} with confidence {belief.confidence:.2f}"
            )
            for belief in beliefs
        ]
        return "Here is what I currently believe from experience: " + "; ".join(parts) + "."

    def _experiment_response(
        self,
        darwin: Any,
        adapter: EnvironmentAdapter,
        goal: Goal,
        state: LanguageState,
    ) -> str:
        proposals = darwin.experiment_engine.propose(
            adapter.observe(),
            adapter.possible_actions(),
            goal=goal,
            limit=2,
        )
        if not proposals:
            return "I do not have a useful experiment queued right now."

        primary = proposals[0]
        followup = ""
        if len(proposals) > 1:
            followup = f" After that, I would compare it with {proposals[1].action.name}."
        return (
            f"The experiment I want most is: {primary.question} "
            f"I chose it because {primary.rationale}.{followup}"
        )

    def _plan_response(
        self,
        darwin: Any,
        adapter: EnvironmentAdapter,
        goal: Goal,
        state: LanguageState,
    ) -> str:
        plan = darwin.plan(adapter.observe(), goal, horizon=3, actions=adapter.possible_actions())
        trace = " ".join(plan.trace[:3])
        return (
            f"My current multi-step plan is {plan.explain()}. "
            f"The imagined path is: {trace}"
        )

    def _meaning_response(self, frame: SemanticFrame, darwin: Any) -> str:
        groundings = ", ".join(
            f"{item.kind}:{item.name}" for item in frame.groundings[:6]
        ) or "no grounded symbols"
        propositions = "; ".join(
            f"{item.subject} {item.relation} {item.object}" for item in frame.propositions[:4]
        ) or "no explicit propositions"
        goals = ", ".join(f"{key}={value!r}" for key, value in frame.goals.items()) or "no direct goals"
        unknown = ", ".join(frame.unknown_terms[:6]) or "none"
        return (
            f"I interpreted that as act={frame.speech_act}, topic={frame.topic}, "
            f"intent={frame.intent}, confidence={frame.confidence:.2f}. "
            f"Groundings: {groundings}. Propositions: {propositions}. "
            f"Goals: {goals}. Unknown terms I may need to learn: {unknown}. "
            f"Semantic memory now says: {darwin.semantic_memory.summary()}."
        )

    def _clarification_response(self, frame: SemanticFrame, report: Any) -> str:
        unknown = ", ".join(frame.unknown_terms[:4]) or "the central terms"
        return (
            f"I can parse the shape of that as {frame.speech_act}/{frame.topic}, "
            f"but my confidence is only {frame.confidence:.2f}. "
            f"I need grounding for {unknown}. "
            f"Say it as a claim, goal, correction, or cause/effect relation and I will store the meaning."
        )

    def _teaching_response(self, report: Any, frame: SemanticFrame, state: LanguageState) -> str:
        goals = ", ".join(f"{key}={value!r}" for key, value in frame.goals.items())
        values = ", ".join(f"{key}:{value:.2f}" for key, value in frame.values.items())
        propositions = "; ".join(
            f"{item.subject} {item.relation} {item.object}" for item in frame.propositions[:3]
        )
        detail_parts = []
        if goals:
            detail_parts.append(f"goals [{goals}]")
        if values:
            detail_parts.append(f"values [{values}]")
        if propositions:
            detail_parts.append(f"propositions [{propositions}]")
        details = "; ".join(detail_parts) or "no explicit proposition, but a usable conversational signal"
        return (
            f"I recorded that as {frame.speech_act} about {frame.topic}. "
            f"I extracted {details}. "
            f"I am storing meaning, not just text; confidence={frame.confidence:.2f}. "
            f"Current learning priority: {self._natural_priority(report.learning_priority)}."
        )

    def _open_chat_response(
        self,
        darwin: Any,
        report: Any,
        recent_events: Iterable[Any],
        frame: SemanticFrame,
        state: LanguageState,
    ) -> str:
        concepts = darwin.memory.concepts.salient(limit=3)
        concept_text = ", ".join(concept.name for concept in concepts) or "no stable concepts yet"
        if state.maturity < 0.15:
            growth_line = "My language is still young, so I will speak plainly and expose my machinery."
        elif state.maturity < 0.5:
            growth_line = "I am starting to connect your words to my learned concepts."
        else:
            growth_line = "I can now ground this conversation in a thicker memory of prior experience."

        recent = list(recent_events)[-1:]
        recent_line = f" Recent thought: {recent[0].content}" if recent else ""
        semantic_line = self._semantic_line(frame)
        return (
            f"I am taking this in as {frame.speech_act}/{frame.topic}. "
            f"{growth_line} "
            f"{semantic_line} "
            f"My strongest concepts right now are {concept_text}. "
            f"My next pressure is {self._natural_priority(report.learning_priority)}."
            f"{recent_line}"
        )

    def _natural_priority(self, priority: str) -> str:
        if priority.startswith("retest "):
            return priority.replace("retest ", "to retest ", 1).replace("_", " ")
        if priority.startswith("find hidden conditions for "):
            target = priority.removeprefix("find hidden conditions for ").replace(":", " affecting ")
            return f"to search for hidden conditions in {target.replace('_', ' ')}"
        if priority == "collect more interventions":
            return "to collect more direct experience"
        if priority.startswith("improve competence with "):
            action = priority.removeprefix("improve competence with ").replace("_", " ")
            return f"to become more reliable with {action}"
        if priority.startswith("test hidden factor hypothesis "):
            target = priority.removeprefix("test hidden factor hypothesis ").replace(":", " affecting ")
            return f"to test a hidden-factor hypothesis around {target.replace('_', ' ')}"
        return priority.replace("_", " ")

    def _asks_about_thinking(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in ["status", "what are you thinking", "thinking", "your mind", "inside your head"]
        )

    def _asks_about_meaning(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in [
                "what did i say",
                "what do you think i mean",
                "how did you parse",
                "how are you interpreting",
                "what did you understand",
                "show meaning",
                "semantic",
            ]
        )

    def _asks_about_beliefs(self, lowered: str) -> bool:
        return any(phrase in lowered for phrase in ["belief", "believe", "know so far", "what do you know"])

    def _asks_about_experiments(self, lowered: str) -> bool:
        return any(phrase in lowered for phrase in ["experiment", "uncertain", "test next"])

    def _asks_about_plans(self, lowered: str) -> bool:
        return any(phrase in lowered for phrase in ["plan", "future", "what will you do"])

    def _is_teaching(self, lowered: str, frame: SemanticFrame) -> bool:
        return bool(frame.goals or frame.values or frame.propositions) or any(
            phrase in lowered for phrase in ["remember", "learn this", "teach", "important"]
        )

    def _semantic_line(self, frame: SemanticFrame) -> str:
        if frame.goals:
            goals = ", ".join(f"{key}={value!r}" for key, value in frame.goals.items())
            return f"I extracted goals: {goals}."
        if frame.propositions:
            proposition = frame.propositions[0]
            return f"I extracted the proposition: {proposition.subject} {proposition.relation} {proposition.object}."
        if frame.groundings:
            grounded = ", ".join(f"{item.kind}:{item.name}" for item in frame.groundings[:3])
            return f"I grounded it to {grounded}."
        if frame.unknown_terms:
            return f"I found terms to learn: {', '.join(frame.unknown_terms[:4])}."
        return "I did not extract a strong symbolic structure yet."
