from __future__ import annotations

import hashlib

from darwin.discourse import ResponsePlan
from darwin.semantics import SemanticFrame
from darwin.thought import ThoughtTrace


class NaturalLanguageComposer:
    """Composes responses from a structured ResponsePlan.

    The composer is faithful by construction — every sentence is derived
    from a field on the plan, never invented. It is intentionally
    conversational rather than report-like. The DLM (gemma-3-270m) can
    paraphrase the composer's output more fluently when it's available,
    but the composer's own text is the baseline contract for what
    "natural Darwin" sounds like.
    """

    def compose(self, plan: ResponsePlan, frame: SemanticFrame, trace: ThoughtTrace) -> str:
        if plan.mode == "greeting":
            return self._render_greeting(plan, frame)
        if plan.mode == "farewell":
            return self._render_farewell(plan, frame)
        if plan.mode == "small_talk":
            return self._render_small_talk(plan, frame)
        if plan.mode == "identity":
            return self._render_identity(plan, frame)
        if plan.mode == "learn":
            return self._render_learn(plan, frame)
        if plan.mode == "clarify":
            return self._render_clarify(plan, frame)
        return self._render_substantive(plan, frame)

    def _render_learn(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        # Reads the structured 'evidence' the planner attached and
        # composes a natural acknowledgement. No fixed English in the
        # planner; only renderer-level templates here.
        signals = self._learn_signals(plan)
        parts: list[str] = []
        snippet = signals.get("snippet")
        if snippet:
            parts.append(f"Noted: {snippet}.")
        prop = signals.get("proposition")
        if prop:
            subject, relation, obj = prop
            parts.append(f"I'll remember that {subject} {relation} {obj}.")
        goals = signals.get("goals", [])
        if goals and not prop:
            for key, value in goals[:1]:
                key_h = key.replace("_", " ")
                if value == "True" or value == True:
                    parts.append(f"Stored goal: keep {key_h} true.")
                elif value == "False" or value == False:
                    parts.append(f"Stored goal: keep {key_h} false.")
                elif value == "'increase'":
                    parts.append(f"Stored goal: increase {key_h}.")
                else:
                    parts.append(f"Stored goal: target {key_h} = {value}.")
        correction = signals.get("correction")
        if correction:
            parts.append(f"I'll update my view: {correction}.")
        if not parts:
            return "Noted."
        return self._smooth(" ".join(parts))

    def _learn_signals(self, plan: ResponsePlan) -> dict:
        out: dict = {"goals": []}
        for entry in plan.evidence:
            if "::" not in entry:
                continue
            kind, body = entry.split("::", 1)
            kind = kind.strip()
            body = body.strip()
            if kind == "snippet":
                out["snippet"] = body
            elif kind == "proposition" and body.count("|") >= 2:
                subj, rel, obj = body.split("|", 2)
                out["proposition"] = (subj, rel, obj)
            elif kind == "goal" and "|" in body:
                key, value = body.split("|", 1)
                out["goals"].append((key, value))
            elif kind == "correction":
                out["correction"] = body
        return out

    # -- social modes ---------------------------------------------------
    #
    # These renderings live in the LANGUAGE layer on purpose. The
    # cognitive layer only labels the intent ('greeting', 'farewell',
    # 'small_talk' with a sub_intent, 'identity' with self_reflection
    # carrying actual state). The composer's job is to turn that
    # structured intent into English, the same way it does for all
    # other modes. Templates here are renderer primitives, not
    # hardcoded cognition. The DLM (gemma-3-270m) will replace these
    # with fluent paraphrases when wired in.

    def _render_greeting(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        choices = ["Hi.", "Hello.", "Hey there."]
        return self._choose(choices, frame.original_text)

    def _render_farewell(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        choices = ["Talk to you later.", "See you.", "Goodbye."]
        return self._choose(choices, frame.original_text)

    def _render_small_talk(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        sub_intent = plan.intent
        if sub_intent == "acknowledge_gratitude":
            return self._choose(["You're welcome.", "Sure.", "Anytime."], frame.original_text)
        if sub_intent == "confirm_presence":
            return self._choose(["Yes, I'm here.", "Here.", "Still here."], frame.original_text)
        if sub_intent == "report_state_briefly":
            # Pull from self_reflection if present; otherwise generic.
            reflection = self._reflection_dict(plan)
            observations = reflection.get("observations")
            priority = reflection.get("learning_priority")
            if observations and priority:
                return (
                    f"I'm running. {observations} observations so far, "
                    f"currently focused on {self._humanize_priority(priority)}."
                )
            if observations:
                return f"I'm running. {observations} observations so far."
            return self._choose(["I'm running.", "Here, thinking."], frame.original_text)
        return self._choose(["Okay.", "Got it.", "Noted."], frame.original_text)

    def _render_identity(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        # Pure derivation from the structured self-state on the plan.
        # No fixed bio. If self_reflection is empty, the composer falls
        # back to the smallest honest claim it can make.
        reflection = self._reflection_dict(plan)
        if not reflection:
            return "I don't have enough self-state yet to introduce myself."
        name = reflection.get("name", "Darwin")
        observations = reflection.get("observations", "0")
        known_actions = reflection.get("known_actions", "0")
        known_variables = reflection.get("known_variables", "0")
        strongest_action = reflection.get("strongest_action", "none yet")
        priority = reflection.get("learning_priority", "")

        parts = [
            f"I'm {name}.",
            (
                f"So far I've made {observations} direct observations across "
                f"{known_actions} actions and {known_variables} world variables."
            ),
        ]
        if strongest_action and strongest_action != "none yet":
            parts.append(
                f"The action I have the strongest causal beliefs about is {strongest_action.replace('_', ' ')}."
            )
        if priority:
            parts.append(f"Right now my learning is focused on {self._humanize_priority(priority)}.")
        return " ".join(parts)

    def _reflection_dict(self, plan: ResponsePlan) -> dict[str, str]:
        out: dict[str, str] = {}
        for entry in plan.self_reflection:
            if ":" not in entry:
                continue
            key, value = entry.split(":", 1)
            out[key.strip()] = value.strip()
        return out

    def _humanize_priority(self, priority: str) -> str:
        text = priority.strip()
        if text.startswith("retest "):
            return "retesting " + text[len("retest "):].replace("_", " ")
        if text.startswith("find hidden conditions for "):
            return "finding hidden conditions for " + text[len("find hidden conditions for "):].replace("_", " ").replace(":", " affecting ")
        if text.startswith("improve competence with "):
            return "getting better at " + text[len("improve competence with "):].replace("_", " ")
        if text.startswith("test hidden factor hypothesis "):
            return "testing a hidden-factor hypothesis"
        if text == "collect more interventions":
            return "collecting more direct experience"
        if text == "expand the environment with new actions and variables":
            return "exploring new actions and variables"
        return text.replace("_", " ")

    def _render_clarify(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        question = plan.clarification_questions[0] if plan.clarification_questions else (
            "Could you say a bit more about what you mean?"
        )
        return question

    # -- substantive responses -----------------------------------------

    def _render_substantive(self, plan: ResponsePlan, frame: SemanticFrame) -> str:
        parts: list[str] = []

        if plan.mode == "belief_answer":
            parts.append(self._belief_summary(plan))
        else:
            body = self._body_paragraph(plan)
            if body:
                parts.append(body)
            # Only modes that actually answer a question/explain something
            # are allowed to surface an extra "and by the way I learned X"
            # causal sentence. Acknowledging teaching (mode=learn) and
            # casual conversation should not get spontaneous belief drops.
            if plan.mode in {"answer", "self_report", "experiment", "memory_summary"}:
                causal = self._causal_sentence(plan)
                if causal:
                    parts.append(causal)

        uncertainty = self._uncertainty_sentence(plan)
        if uncertainty:
            parts.append(uncertainty)

        if plan.clarification_questions:
            parts.append(plan.clarification_questions[0])

        text = " ".join(part for part in parts if part)
        if not text:
            text = "I'm not sure what to say to that yet."
        return self._smooth(text)

    def _belief_summary(self, plan: ResponsePlan) -> str:
        # Filter to the strongest, well-supported beliefs. Drop only the
        # very thinly-supported ones; humanize the rest. Deduplicate by
        # (action, variable) so we don't repeat "X makes Y true" twice
        # because both a global and a conditioned belief exist.
        seen: set[tuple[str, str]] = set()
        usable = []
        for claim in plan.causal_claims:
            if claim.confidence < 0.4 or claim.samples < 1:
                continue
            key = (claim.action, claim.variable)
            if key in seen:
                continue
            seen.add(key)
            usable.append(claim)

        if not usable:
            return "I don't have enough direct experience yet to say."

        humanized = []
        for claim in usable[:3]:
            condition = "" if claim.condition == "always" else f" when {claim.condition.replace('_', ' ')},"
            humanized.append(
                f"{claim.action.replace('_', ' ')} makes "
                f"{claim.variable.replace('_', ' ')} {self._humanize_effect(claim.effect)}"
                f"{condition} (seen {claim.samples} {self._noun('time', claim.samples)})"
            )

        if len(humanized) == 1:
            return f"What I've learned is that {humanized[0]}."
        if len(humanized) == 2:
            return f"What I've learned is that {humanized[0]}, and that {humanized[1]}."
        return (
            f"What I've learned is that {humanized[0]}, that {humanized[1]}, "
            f"and that {humanized[2]}."
        )

    def _body_paragraph(self, plan: ResponsePlan) -> str:
        points = [point.strip() for point in plan.answer_points if point and point.strip()]
        if not points:
            return ""

        # Cap to the first three points so substantive responses don't
        # turn into bullet dumps. The DLM can fluff this out, but even
        # the composer should read like a paragraph, not a report.
        chosen = points[:3]
        sentences = [self._point_sentence(point) for point in chosen]
        return " ".join(sentences)

    def _causal_sentence(self, plan: ResponsePlan) -> str:
        # Surface at most ONE causal claim in prose. Multiple claims will
        # appear as separate plan_payload entries for the DLM if the user
        # actually asked for a belief dump (mode == "belief_answer").
        if plan.mode == "belief_answer":
            relevant = [claim for claim in plan.causal_claims if claim.confidence >= 0.55]
            if not relevant:
                return ""
            top = relevant[0]
            condition = "" if top.condition == "always" else f" when {top.condition},"
            human_effect = self._humanize_effect(top.effect)
            return (
                f"From {top.samples} direct {self._noun('observation', top.samples)},"
                f"{condition} {top.action.replace('_', ' ')} "
                f"makes {top.variable.replace('_', ' ')} {human_effect}."
            )
        # In other modes, only mention a causal claim if it scored very
        # high — otherwise we conflate "I noticed something" with the
        # actual answer to whatever was asked.
        for claim in plan.causal_claims:
            if claim.confidence >= 0.75 and claim.samples >= 3:
                human_effect = self._humanize_effect(claim.effect)
                return (
                    f"What I'm most sure of is that {claim.action.replace('_', ' ')} "
                    f"makes {claim.variable.replace('_', ' ')} {human_effect}."
                )
        return ""

    def _uncertainty_sentence(self, plan: ResponsePlan) -> str:
        if not plan.uncertainty_levels:
            return ""
        strong = [level for level in plan.uncertainty_levels if level.level >= 0.55]
        if not strong:
            return ""
        target = strong[0]
        target_text = target.target.replace("_", " ").replace(":", " — ")
        if target.reason:
            return f"I'm not very sure about {target_text} ({target.reason})."
        return f"I'm not very sure about {target_text}."

    # -- helpers --------------------------------------------------------

    def _humanize_effect(self, effect: str) -> str:
        # Map raw before->after notation to natural English.
        cleaned = effect.strip()
        if cleaned == "False -> True":
            return "true"
        if cleaned == "True -> False":
            return "false"
        if cleaned.startswith("None -> "):
            target = cleaned[len("None -> "):].strip().strip("'\"")
            if target.lower() in {"true", "false"}:
                return target.lower()
            return f"become {target}"
        if " -> " in cleaned:
            before, after = (part.strip().strip("'\"") for part in cleaned.split(" -> ", 1))
            if before.lower() == "none":
                return f"become {after}"
            return f"change from {before} to {after}"
        if cleaned.startswith("+="):
            try:
                delta = float(cleaned[2:].strip())
            except ValueError:
                return cleaned
            if delta < 0:
                return f"drop by {abs(delta):g}"
            if delta > 0:
                return f"rise by {delta:g}"
            return "stay the same"
        return cleaned

    def _noun(self, base: str, count: int) -> str:
        return base if count == 1 else f"{base}s"

    def _point_sentence(self, point: str) -> str:
        stripped = point.strip()
        if not stripped:
            return ""
        if stripped.endswith((".", "?", "!")):
            return stripped[0].upper() + stripped[1:]
        return stripped[0].upper() + stripped[1:] + "."

    def _choose(self, choices: list[str], seed_text: str) -> str:
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return choices[int(digest[:8], 16) % len(choices)]

    def _smooth(self, text: str) -> str:
        text = " ".join(text.split())
        text = text.replace(" .", ".").replace(" ,", ",")
        text = text.replace("..", ".").replace(" ?", "?").replace(" !", "!")
        return text.strip()
