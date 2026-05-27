"""LLM-free symbolic discourse realizer for Darwin v5.

This module replaces the LLM-based surface realization path entirely on the v5
kernel. There is no token sampling, no prompt template handed to a model, no
external inference. Every word in every utterance is composed deterministically
from ``ResponsePlan`` fields by rhetorical strategies.

Variety in phrasing comes from three sources, all deterministic:
  1. Plan content fingerprint — same plan + same internal state -> same output.
  2. Darwin's current cognitive state (learning_priority, competence, recent
     surprise) modulates section weights and qualifier strength.
  3. A small in-process ``StarterRegistry`` that tracks the last N sentence
     openers and avoids reuse across consecutive utterances.

The realizer is intentionally tunable. Its parameters live in ``RealizerConfig``
so Phase E's ``SelfModificationEngine`` can propose changes to ``connector_
frequency``, ``aside_rate``, ``qualifier_strength``, ``opening_strategy_
weights``, and per-mode length budgets. Darwin learns to talk better not by
sampling tokens but by tuning its own composition strategies against the
critic's pass rate on held-out plans.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from darwin.connectors import (
    CLOSE_OPENERS,
    FAREWELL_TOKENS,
    FUNCTION_WORDS,
    GRATITUDE_ACKNOWLEDGEMENTS,
    GREETING_ACKNOWLEDGEMENTS,
    INVITES,
    PRESENCE_CONFIRMATIONS,
    QUALIFIER_OPENERS,
    REFLECTIVE_OPENERS,
    SECONDARY_OPENERS,
    SECTION_OPENERS,
    STRUCTURE_CONNECTORS,
)
from darwin.discourse import (
    CausalClaim,
    ReferencedExperience,
    ResponsePlan,
    UncertaintyLevel,
)


__all__ = [
    "DiscourseRealizer",
    "RealizerConfig",
    "RealizerOutput",
    "StarterRegistry",
    "build_content_alias_table",
    "tokenize_content_words",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


_DEFAULT_LENGTH_PER_MODE = {
    "greeting": "short",
    "farewell": "short",
    "small_talk": "short",
    "identity": "medium",
    "self_description": "medium",
    "self_history": "medium",
    "self_capabilities": "medium",
    "knowledge_answer": "medium",
    "belief_answer": "medium",
    "answer": "medium",
    "experiment": "medium",
    "self_report": "medium",
    "memory_summary": "medium",
    "unknown_terms": "medium",
    "clarify": "short",
    "learn": "short",
    "conversation": "short",
}


_DEFAULT_OPENING_WEIGHTS = {
    "direct": 0.45,
    "reflective": 0.20,
    "noticing": 0.20,
    "framed": 0.15,
}


@dataclass
class RealizerConfig:
    """Tunable knobs for the symbolic realizer.

    These are the levers Phase E's self-modification engine can adjust via
    ``propose_realizer_config()``. Every default below is conservative —
    the realizer reads well at construction time, and Darwin tunes from there.
    """

    connector_frequency: float = 0.35
    aside_rate: float = 0.25
    qualifier_strength: float = 0.5
    opening_strategy_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_OPENING_WEIGHTS)
    )
    length_per_mode: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_LENGTH_PER_MODE)
    )
    starter_window: int = 8

    def length_for(self, mode: str, plan_length: str) -> str:
        """Per-mode length default; plan can override via ``target_length``."""

        return plan_length if plan_length in {"short", "medium", "long"} else self.length_per_mode.get(mode, "medium")

    def to_record(self) -> dict[str, Any]:
        return {
            "connector_frequency": self.connector_frequency,
            "aside_rate": self.aside_rate,
            "qualifier_strength": self.qualifier_strength,
            "opening_strategy_weights": dict(self.opening_strategy_weights),
            "length_per_mode": dict(self.length_per_mode),
            "starter_window": self.starter_window,
        }


# ---------------------------------------------------------------------------
# Starter registry — avoids repeating sentence openers across recent turns.
# ---------------------------------------------------------------------------


class StarterRegistry:
    """Track the last N sentence openers so the realizer doesn't repeat itself.

    The realizer normalizes an opener (lowercase, first 4 words) and asks the
    registry whether it has been used recently. Avoidance is best-effort — if
    every candidate is recent, the registry returns the candidate with the
    oldest hit so the conversation still keeps moving.
    """

    def __init__(self, max_recent: int = 8) -> None:
        self.max_recent = max(1, max_recent)
        self._history: list[str] = []

    def record(self, opener: str) -> None:
        normalized = self._normalize(opener)
        if not normalized:
            return
        self._history.append(normalized)
        if len(self._history) > self.max_recent:
            self._history = self._history[-self.max_recent:]

    def is_recent(self, opener: str) -> bool:
        return self._normalize(opener) in self._history

    def pick(self, candidates: Iterable[str], fingerprint: str) -> str:
        """Choose an opener that isn't recent. Fingerprint provides stability."""

        choices = [c for c in candidates if c]
        if not choices:
            return ""
        fresh = [c for c in choices if not self.is_recent(c)]
        pool = fresh if fresh else choices
        idx = int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % len(pool)
        return pool[idx]

    def _normalize(self, opener: str) -> str:
        words = re.findall(r"[a-z']+", opener.lower())
        return " ".join(words[:4])


# ---------------------------------------------------------------------------
# Content-word extraction & alias table for validator.
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def tokenize_content_words(text: str) -> list[str]:
    """Lower-cased word tokens (no punctuation, no numbers). Validator helper."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _morphological_variants(word: str) -> set[str]:
    """Tiny suffix-based morphology (no third-party deps).

    Produces a small alias set for a plan-content word so common
    inflections (acceleration <-> accelerating, force <-> forces) all count
    as references to the same content.
    """

    word = word.lower()
    variants = {word}
    if len(word) < 4:
        return variants
    if word.endswith("ies"):
        variants.add(word[:-3] + "y")
    elif word.endswith("es") and not word.endswith("ses"):
        variants.add(word[:-2])
    elif word.endswith("s") and not word.endswith("ss"):
        variants.add(word[:-1])
    if word.endswith("ing"):
        stem = word[:-3]
        variants.add(stem)
        variants.add(stem + "e")
    if word.endswith("ed"):
        stem = word[:-2]
        variants.add(stem)
        variants.add(stem + "e")
    # Reverse direction: add the -ing / -ed / -s for a stem.
    if word.isalpha():
        variants.add(word + "s")
        variants.add(word + "ed")
        variants.add(word + "ing")
    return variants


def build_content_alias_table(plan: ResponsePlan) -> set[str]:
    """Collect every content word the validator should accept.

    Every nominal/verbal token in the plan is added, plus a small set of
    morphological variants. The realizer's output is checked against this
    union plus ``FUNCTION_WORDS`` and ``STRUCTURE_CONNECTORS``.
    """

    pool: set[str] = set()
    pool.update(tokenize_content_words(plan.thesis or ""))
    for point in plan.answer_points:
        pool.update(tokenize_content_words(point))
    for entry in plan.self_reflection:
        pool.update(tokenize_content_words(entry))
    for question in plan.clarification_questions:
        pool.update(tokenize_content_words(question))
    for action in plan.next_actions:
        pool.update(tokenize_content_words(action))
    for level in plan.uncertainty_levels:
        pool.update(tokenize_content_words(level.target))
        pool.update(tokenize_content_words(level.reason))
    for claim in plan.causal_claims:
        for value in (claim.action, claim.variable, claim.effect, claim.condition):
            pool.update(tokenize_content_words(value))
    for experience in plan.referenced_experiences:
        pool.update(tokenize_content_words(experience.title))
        pool.update(tokenize_content_words(experience.summary))
    for entry in plan.evidence:
        pool.update(tokenize_content_words(entry))
    # Mode/intent strings are descriptive — include them so words like
    # "greeting", "experiment", "memory" can appear without surprise.
    pool.update(tokenize_content_words(plan.mode))
    pool.update(tokenize_content_words(plan.intent))

    aliases: set[str] = set()
    for word in pool:
        aliases.update(_morphological_variants(word))
    # Number-words the realizer commonly emits when surfacing samples.
    aliases.update({"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "dozen"})
    return aliases


# ---------------------------------------------------------------------------
# Surface rules — turn plan-field structures into natural sentences.
#
# Every humanizer below has 4-8 surface variants. Selection is deterministic
# (content-fingerprint hash) so the realizer never relies on sampling.
# ---------------------------------------------------------------------------


_NUMBER_WORDS = {
    1: "once",
    2: "twice",
    3: "three times",
    4: "four times",
    5: "five times",
    6: "six times",
    7: "seven times",
    8: "eight times",
    9: "nine times",
    10: "ten times",
}


def _humanize_count(samples: int) -> str:
    if samples in _NUMBER_WORDS:
        return _NUMBER_WORDS[samples]
    return f"{samples} times"


def _humanize_effect(effect: str) -> str:
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


def _humanize_action_name(name: str) -> str:
    return name.replace("_", " ").replace("/", " ")


def _humanize_variable(name: str) -> str:
    return name.replace("_", " ").replace(".", " ")


CAUSAL_VARIANTS: tuple[Callable[[CausalClaim], str], ...] = (
    lambda claim: (
        f"applying {_humanize_action_name(claim.action)} makes "
        f"{_humanize_variable(claim.variable)} "
        f"{_humanize_effect(claim.effect)}"
    ),
    lambda claim: (
        f"{_humanize_variable(claim.variable)} tends to "
        f"{_humanize_effect(claim.effect)} once {_humanize_action_name(claim.action)} runs"
    ),
    lambda claim: (
        f"{_humanize_action_name(claim.action)} is the move that pushes "
        f"{_humanize_variable(claim.variable)} to {_humanize_effect(claim.effect)}"
    ),
    lambda claim: (
        f"every run of {_humanize_action_name(claim.action)} seems to "
        f"{_humanize_effect(claim.effect)} on {_humanize_variable(claim.variable)}"
    ),
    lambda claim: (
        f"the cleanest link i have is "
        f"{_humanize_action_name(claim.action)} → {_humanize_variable(claim.variable)} "
        f"({_humanize_effect(claim.effect)})"
    ),
)


def humanize_causal_claim(claim: CausalClaim, fingerprint: str) -> str:
    if claim.condition and claim.condition != "always":
        condition_text = f" when {claim.condition.replace('_', ' ')}"
    else:
        condition_text = ""
    idx = int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % len(CAUSAL_VARIANTS)
    base = CAUSAL_VARIANTS[idx](claim)
    sample_tag = f" — i've seen this {_humanize_count(claim.samples)}" if claim.samples >= 3 else ""
    sentence = f"{base}{condition_text}{sample_tag}"
    return _capitalize(sentence)


UNCERTAINTY_VARIANTS: tuple[Callable[[UncertaintyLevel], str], ...] = (
    lambda level: (
        f"i'm still uncertain about "
        f"{level.target.replace('_', ' ').replace(':', ' — ')}"
    ),
    lambda level: (
        f"one thing that's still fuzzy for me is "
        f"{level.target.replace('_', ' ').replace(':', ' — ')}"
    ),
    lambda level: (
        f"i wouldn't bet much on my read of "
        f"{level.target.replace('_', ' ').replace(':', ' — ')} yet"
    ),
    lambda level: (
        f"the thinner part of my understanding is around "
        f"{level.target.replace('_', ' ').replace(':', ' — ')}"
    ),
)


def humanize_uncertainty(level: UncertaintyLevel, fingerprint: str) -> str:
    idx = int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % len(UNCERTAINTY_VARIANTS)
    base = UNCERTAINTY_VARIANTS[idx](level)
    if level.reason:
        tail = f" — {level.reason}"
    else:
        tail = ""
    return _capitalize(f"{base}{tail}")


EXPERIENCE_VARIANTS: tuple[Callable[[ReferencedExperience], str], ...] = (
    lambda exp: f"that fits with {exp.summary}",
    lambda exp: f"last time i looked at this i saw {exp.summary}",
    lambda exp: f"a related thread in memory is {exp.summary}",
    lambda exp: f"connected to that is {exp.summary}",
)


def humanize_experience(experience: ReferencedExperience, fingerprint: str) -> str:
    idx = int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % len(EXPERIENCE_VARIANTS)
    return _capitalize(EXPERIENCE_VARIANTS[idx](experience))


def humanize_reflection(reflection: str, fingerprint: str) -> str:
    """A self_reflection entry is rendered as a reflective close beat."""

    cleaned = reflection.strip()
    if ":" in cleaned:
        key, value = cleaned.split(":", 1)
        key = key.strip().replace("_", " ")
        value = value.strip()
        templates = (
            f"my {key} right now is {value}",
            f"{key}-wise, i'm sitting at {value}",
            f"on {key}, where i am is {value}",
        )
    else:
        templates = (
            cleaned,
            f"the bigger thread: {cleaned}",
            f"underneath it all: {cleaned}",
        )
    idx = int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % len(templates)
    return _capitalize(templates[idx])


def _capitalize(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return sentence
    return sentence[0].upper() + sentence[1:]


# ---------------------------------------------------------------------------
# Outline planning — mode-aware rhetorical shape.
# ---------------------------------------------------------------------------


@dataclass
class OutlineSection:
    name: str
    sentences: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)


@dataclass
class Outline:
    mode: str
    sections: list[OutlineSection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Realizer output type.
# ---------------------------------------------------------------------------


@dataclass
class RealizerOutput:
    text: str
    provenance_map: list[dict[str, Any]]
    sentences: list[str]
    outline: Outline | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sentences": list(self.sentences),
            "provenance_map": list(self.provenance_map),
        }


# ---------------------------------------------------------------------------
# DiscourseRealizer
# ---------------------------------------------------------------------------


class DiscourseRealizer:
    """Symbolic, deterministic, faithful surface realizer.

    Pipeline:
        1. Outline  — pick a rhetorical shape based on plan.mode.
        2. Draft    — fill each section by humanizing the relevant plan fields.
        3. Weave    — add transitions, sentence-length variety, asides.
        4. Budget   — trim/expand to fit target_length.
        5. Smooth   — fix punctuation, capitalize sentence-initials.

    Variety is deterministic (plan fingerprint + starter-registry avoidance).
    """

    def __init__(
        self,
        config: RealizerConfig | None = None,
        registry: StarterRegistry | None = None,
    ) -> None:
        self.config = config or RealizerConfig()
        self.registry = registry or StarterRegistry(max_recent=self.config.starter_window)

    # -- public API --------------------------------------------------------

    def realize(self, plan: ResponsePlan) -> RealizerOutput:
        fingerprint = self._fingerprint(plan)
        outline = self._build_outline(plan, fingerprint)
        outline = self._draft(outline, plan, fingerprint)
        sentences, provenance = self._weave(outline, plan, fingerprint)
        sentences, provenance = self._budget(
            sentences,
            provenance,
            self.config.length_for(plan.mode, plan.target_length),
        )
        text = self._smooth(" ".join(sentences))
        # Record only the FIRST sentence opener — anti-repetition acts at
        # the start of utterances, not inside one.
        if sentences:
            first = sentences[0].lower()
            self.registry.record(first)
        return RealizerOutput(
            text=text,
            provenance_map=provenance,
            sentences=sentences,
            outline=outline,
        )

    # -- pipeline ----------------------------------------------------------

    def _fingerprint(self, plan: ResponsePlan) -> str:
        seed = "|".join(
            [
                plan.plan_id,
                plan.mode,
                plan.intent,
                plan.thesis,
                ",".join(plan.answer_points),
                ",".join(plan.self_reflection),
                ",".join(f"{c.action}->{c.variable}:{c.effect}" for c in plan.causal_claims),
                ",".join(f"{u.target}:{u.level:.2f}" for u in plan.uncertainty_levels),
            ]
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def _build_outline(self, plan: ResponsePlan, fingerprint: str) -> Outline:
        mode = plan.mode
        sections: list[OutlineSection] = []
        if mode == "greeting":
            sections = [
                OutlineSection("acknowledge"),
                OutlineSection("invite"),
            ]
        elif mode == "farewell":
            sections = [OutlineSection("acknowledge")]
        elif mode == "small_talk":
            sections = [OutlineSection("acknowledge")]
            if plan.self_reflection:
                sections.append(OutlineSection("state_share"))
        elif mode == "identity":
            sections = [OutlineSection("identity"), OutlineSection("focus")]
        elif mode in {"self_description", "self_history", "self_capabilities"}:
            sections = [
                OutlineSection("identity"),
                OutlineSection("focus"),
                OutlineSection("close"),
            ]
        elif mode == "clarify":
            sections = [OutlineSection("acknowledge_gap"), OutlineSection("question")]
        elif mode == "learn":
            sections = [OutlineSection("acknowledge_input")]
        elif mode == "knowledge_answer":
            sections = [
                OutlineSection("hook"),
                OutlineSection("develop"),
                OutlineSection("qualify"),
            ]
        elif mode == "belief_answer":
            sections = [
                OutlineSection("hook"),
                OutlineSection("ground"),
                OutlineSection("qualify"),
                OutlineSection("close"),
            ]
        elif mode in {"answer", "experiment", "self_report", "memory_summary", "unknown_terms"}:
            sections = [
                OutlineSection("hook"),
                OutlineSection("develop"),
                OutlineSection("ground"),
                OutlineSection("qualify"),
            ]
        elif mode == "conversation":
            sections = [OutlineSection("acknowledge")]
            if plan.answer_points:
                sections.append(OutlineSection("link"))
        else:
            sections = [OutlineSection("hook"), OutlineSection("develop")]
        return Outline(mode=mode, sections=sections)

    def _draft(self, outline: Outline, plan: ResponsePlan, fingerprint: str) -> Outline:
        ctx: dict[str, Any] = {"claims_used": set()}
        for section in outline.sections:
            drafter = _SECTION_DRAFTERS.get(section.name, _draft_fallback)
            drafter(section, plan, self.registry, fingerprint, self.config, ctx)
        return outline

    def _weave(
        self,
        outline: Outline,
        plan: ResponsePlan,
        fingerprint: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        sentences: list[str] = []
        provenance: list[dict[str, Any]] = []
        cursor = 0
        for section_index, section in enumerate(outline.sections):
            for sentence_index, sentence in enumerate(section.sentences):
                if not sentence:
                    continue
                source = section.provenance[sentence_index] if sentence_index < len(section.provenance) else f"{section.name}:?"
                # Light-touch transition: section breaks may take a connector.
                if (
                    section_index > 0
                    and sentence_index == 0
                    and self._should_connect(fingerprint, section.name)
                ):
                    connector = self._pick_connector(section.name, fingerprint + section.name)
                    if connector:
                        sentence = f"{connector} {sentence[0].lower()}{sentence[1:]}"
                sentences.append(sentence)
                provenance.append({
                    "span": cursor,
                    "section": section.name,
                    "source": source,
                })
                cursor += 1
        return sentences, provenance

    def _budget(
        self,
        sentences: list[str],
        provenance: list[dict[str, Any]],
        target_length: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if not sentences:
            return sentences, provenance
        max_sentences = {"short": 2, "medium": 4, "long": 6}.get(target_length, 4)
        if len(sentences) <= max_sentences:
            return sentences, provenance
        # Trimming preserves causal grounding first (sections with "ground" /
        # "develop" in their name), then qualifiers, then closes. We rank
        # each sentence by priority and drop the least important ones.
        scored = list(zip(sentences, provenance))
        priority = {
            "hook": 3,
            "identity": 3,
            "acknowledge": 3,
            "acknowledge_input": 3,
            "acknowledge_gap": 3,
            "develop": 4,
            "ground": 5,
            "state_share": 3,
            "focus": 2,
            "qualify": 4,
            "link": 2,
            "question": 5,
            "close": 1,
            "invite": 2,
        }
        scored.sort(
            key=lambda item: priority.get(item[1]["section"], 2),
            reverse=True,
        )
        scored = scored[:max_sentences]
        # Restore original order using the recorded ``span`` field.
        scored.sort(key=lambda item: item[1]["span"])
        return [item[0] for item in scored], [item[1] for item in scored]

    def _smooth(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace(" .", ".").replace(" ,", ",")
        text = text.replace(" ?", "?").replace(" !", "!")
        text = re.sub(r"\.\.+", ".", text)
        # Capitalize sentence-initial letters after period/question/exclam.
        def _cap(match: re.Match[str]) -> str:
            return match.group(1) + match.group(2).upper()
        text = re.sub(r"(^|[.?!]\s+)([a-z])", _cap, text)
        return text

    # -- connector logic --------------------------------------------------

    def _should_connect(self, fingerprint: str, section_name: str) -> bool:
        digest = hashlib.sha256((fingerprint + section_name).encode("utf-8")).hexdigest()
        bucket = int(digest[:4], 16) / 0xFFFF
        return bucket < self.config.connector_frequency

    def _pick_connector(self, section_name: str, seed: str) -> str:
        if section_name == "qualify":
            return self.registry.pick(QUALIFIER_OPENERS, seed)
        if section_name == "close":
            return self.registry.pick(CLOSE_OPENERS, seed)
        if section_name == "ground":
            return self.registry.pick(REFLECTIVE_OPENERS, seed)
        return self.registry.pick(SECONDARY_OPENERS, seed)


# ---------------------------------------------------------------------------
# Per-section drafters.
# Each fills ``section.sentences`` and ``section.provenance`` in place.
# ---------------------------------------------------------------------------


def _opener_for_hook(plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig) -> str:
    weights = config.opening_strategy_weights
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    bucket = int(digest[:4], 16) / 0xFFFF
    cumulative = 0.0
    chosen = "direct"
    for strategy, weight in weights.items():
        cumulative += weight
        if bucket <= cumulative:
            chosen = strategy
            break
    if chosen == "noticing":
        return registry.pick(REFLECTIVE_OPENERS, fingerprint + "hook")
    if chosen == "framed":
        return registry.pick(SECTION_OPENERS, fingerprint + "hook")
    if chosen == "reflective":
        return registry.pick(REFLECTIVE_OPENERS, fingerprint + "reflect")
    return ""


def _draft_acknowledge(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    if plan.mode == "greeting":
        opener = registry.pick(GREETING_ACKNOWLEDGEMENTS, fingerprint + "greet")
        section.sentences.append(_capitalize(opener + "."))
        section.provenance.append("greeting:acknowledge")
        return
    if plan.mode == "farewell":
        opener = registry.pick(FAREWELL_TOKENS, fingerprint + "fare")
        section.sentences.append(_capitalize(opener + "."))
        section.provenance.append("farewell:acknowledge")
        return
    if plan.mode == "small_talk":
        intent = plan.intent
        if intent == "acknowledge_gratitude":
            opener = registry.pick(GRATITUDE_ACKNOWLEDGEMENTS, fingerprint + "thx")
            section.sentences.append(_capitalize(opener + "."))
            section.provenance.append("small_talk:gratitude")
            return
        if intent == "confirm_presence":
            opener = registry.pick(PRESENCE_CONFIRMATIONS, fingerprint + "here")
            section.sentences.append(_capitalize(opener + "."))
            section.provenance.append("small_talk:presence")
            return
        if intent == "report_state_briefly":
            section.sentences.append("Here, thinking.")
            section.provenance.append("small_talk:state")
            return
        opener = registry.pick(("noted", "got it", "okay", "alright"), fingerprint + "ack")
        section.sentences.append(_capitalize(opener + "."))
        section.provenance.append("small_talk:default")
        return
    if plan.mode == "conversation":
        opener = registry.pick(("got it", "noted", "okay"), fingerprint + "convack")
        section.sentences.append(_capitalize(opener + "."))
        section.provenance.append("conversation:acknowledge")


def _draft_invite(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    invite = registry.pick(INVITES, fingerprint + "invite")
    section.sentences.append(_capitalize(invite))
    section.provenance.append("invite")


def _draft_state_share(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    reflection = _reflection_dict(plan.self_reflection)
    observations = reflection.get("observations")
    priority = reflection.get("learning_priority") or reflection.get("focus")
    if observations and priority:
        sentence = (
            f"I'm running — {observations} observations in, and the part i'm chewing on is "
            f"{priority.replace('_', ' ')}."
        )
    elif observations:
        sentence = f"I'm running, {observations} observations deep."
    elif priority:
        sentence = f"I'm focused on {priority.replace('_', ' ')} right now."
    else:
        sentence = "I'm here, thinking."
    section.sentences.append(sentence)
    section.provenance.append("self_reflection:state")


def _draft_identity(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    reflection = _reflection_dict(plan.self_reflection)
    if not reflection:
        section.sentences.append("I'm Darwin — a causal-adaptive system, still building up direct experience.")
        section.provenance.append("identity:default")
        return
    name = reflection.get("name", "Darwin")
    observations = reflection.get("observations")
    known_actions = reflection.get("known_actions")
    known_variables = reflection.get("known_variables")
    parts = [f"I'm {name}."]
    if observations and known_actions and known_variables:
        parts.append(
            f"So far i've made {observations} direct observations across "
            f"{known_actions} actions and {known_variables} world variables."
        )
    elif observations:
        parts.append(f"So far i've made {observations} direct observations.")
    strongest = reflection.get("strongest_action")
    if strongest and strongest != "none yet":
        parts.append(
            f"The action i have the cleanest causal hold on is "
            f"{_humanize_action_name(strongest)}."
        )
    sentence = " ".join(parts)
    section.sentences.append(_capitalize(sentence))
    section.provenance.append("identity:from_reflection")


def _draft_focus(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    reflection = _reflection_dict(plan.self_reflection)
    priority = reflection.get("learning_priority") or reflection.get("focus")
    if not priority:
        return
    opener = registry.pick(REFLECTIVE_OPENERS, fingerprint + "focus")
    sentence = f"{opener} my next learning beat is {priority.replace('_', ' ').replace(':', ' on ')}."
    section.sentences.append(_capitalize(sentence))
    section.provenance.append("self_reflection:focus")


def _draft_close(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    if not plan.self_reflection:
        return
    entry = next((r for r in plan.self_reflection if r and "priority" not in r.lower()), None)
    if not entry:
        return
    opener = registry.pick(CLOSE_OPENERS, fingerprint + "close")
    body = humanize_reflection(entry, fingerprint + "closebody")
    sentence = f"{opener} {body[0].lower()}{body[1:]}."
    section.sentences.append(_capitalize(sentence))
    section.provenance.append("self_reflection:close")


def _draft_acknowledge_gap(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    if plan.answer_points:
        section.sentences.append(_capitalize(_period(plan.answer_points[0])))
        section.provenance.append("answer_points[0]")
    else:
        section.sentences.append("I don't have enough grounded structure to answer this cleanly yet.")
        section.provenance.append("clarify:default")


def _draft_question(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    if plan.clarification_questions:
        section.sentences.append(_ensure_question(plan.clarification_questions[0]))
        section.provenance.append("clarification_questions[0]")


def _draft_acknowledge_input(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    signals = _learn_signals(plan)
    parts: list[str] = []
    prov: list[str] = []
    snippet = signals.get("snippet")
    if snippet:
        parts.append(f"Noted — {snippet}.")
        prov.append("evidence:snippet")
    prop = signals.get("proposition")
    if prop:
        subject, relation, obj = prop
        parts.append(f"I'll remember that {subject} {relation} {obj}.")
        prov.append("evidence:proposition")
    goals = signals.get("goals", [])
    if goals and not prop:
        key, value = goals[0]
        key_h = key.replace("_", " ")
        if value in {"True", "true", True, "increase"}:
            parts.append(f"Stored goal: keep {key_h} on track.")
        else:
            parts.append(f"Stored goal: target {key_h} = {value}.")
        prov.append("evidence:goal")
    correction = signals.get("correction")
    if correction:
        parts.append(f"I'll update my view: {correction}.")
        prov.append("evidence:correction")
    if not parts:
        parts.append("Noted.")
        prov.append("learn:default")
    section.sentences.extend(parts)
    section.provenance.extend(prov)


def _draft_hook(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    opener = _opener_for_hook(plan, registry, fingerprint, config)
    if plan.answer_points:
        hook = _period(plan.answer_points[0])
        if opener:
            sentence = f"{opener} {hook[0].lower()}{hook[1:]}"
        else:
            sentence = hook
        section.sentences.append(_capitalize(sentence))
        section.provenance.append("answer_points[0]")
        return
    # For modes like belief_answer that have no answer_points, derive the
    # hook from the strongest causal claim so the realizer varies with the
    # data instead of repeating the thesis verbatim. Different claims yield
    # different humanizations, which is the variety target across plans.
    if plan.causal_claims:
        claim = max(plan.causal_claims, key=lambda c: (c.confidence, c.samples))
        humanized = humanize_causal_claim(claim, fingerprint + "hookclaim")
        sentence = f"{opener} {humanized[0].lower()}{humanized[1:]}" if opener else humanized
        section.sentences.append(_period(_capitalize(sentence)))
        section.provenance.append("causal_claims[hook]")
        ctx.setdefault("claims_used", set()).add((claim.action, claim.variable))
        return
    if plan.thesis:
        hook = _period(plan.thesis)
        sentence = f"{opener} {hook[0].lower()}{hook[1:]}" if opener else hook
        section.sentences.append(_capitalize(sentence))
        section.provenance.append("thesis")


def _draft_develop(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    if len(plan.answer_points) <= 1:
        return
    remaining = plan.answer_points[1:4]
    for idx, point in enumerate(remaining):
        connector = ""
        if idx == 0 and len(remaining) > 1:
            connector = registry.pick(SECONDARY_OPENERS, fingerprint + f"dev{idx}")
        rendered = _period(point.strip())
        if connector:
            sentence = f"{connector} {rendered[0].lower()}{rendered[1:]}"
        else:
            sentence = rendered
        section.sentences.append(_capitalize(sentence))
        section.provenance.append(f"answer_points[{idx + 1}]")


def _draft_ground(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    # Surface up to two grounded causal claims as separate sentences. Mid-
    # confidence claims (<0.55) are skipped to keep grounding honest.
    grounded: list[CausalClaim] = []
    seen: set[tuple[str, str]] = set(ctx.get("claims_used", set()))
    for claim in plan.causal_claims:
        if claim.confidence < 0.55:
            continue
        key = (claim.action, claim.variable)
        if key in seen:
            continue
        seen.add(key)
        grounded.append(claim)
        if len(grounded) >= 2:
            break
    ctx.setdefault("claims_used", set()).update(seen)
    for idx, claim in enumerate(grounded):
        sentence = humanize_causal_claim(claim, fingerprint + f"claim{idx}")
        section.sentences.append(_period(sentence))
        section.provenance.append(f"causal_claims[{idx}]")
    # Optional: a referenced experience if there's room.
    if plan.referenced_experiences and len(section.sentences) < 2:
        experience = plan.referenced_experiences[0]
        if experience.summary:
            sentence = humanize_experience(experience, fingerprint + "exp")
            section.sentences.append(_period(sentence))
            section.provenance.append("referenced_experiences[0]")


def _draft_qualify(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    strong = [level for level in plan.uncertainty_levels if level.level >= 0.5 * (1.0 + (1.0 - config.qualifier_strength))]
    if not strong:
        # Fall back to the highest-level uncertainty regardless of threshold so
        # the qualify section never silently disappears when plan signals it.
        strong = sorted(plan.uncertainty_levels, key=lambda l: l.level, reverse=True)[:1]
    if not strong:
        return
    sentence = humanize_uncertainty(strong[0], fingerprint + "unc")
    section.sentences.append(_period(sentence))
    section.provenance.append("uncertainty_levels[0]")


def _draft_link(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    if not plan.answer_points:
        return
    point = plan.answer_points[0]
    section.sentences.append(_capitalize(_period(point)))
    section.provenance.append("answer_points[0]")


def _draft_fallback(section: OutlineSection, plan: ResponsePlan, registry: StarterRegistry, fingerprint: str, config: RealizerConfig, ctx: dict[str, Any]) -> None:
    if plan.answer_points:
        section.sentences.append(_capitalize(_period(plan.answer_points[0])))
        section.provenance.append("answer_points[0]")
    elif plan.thesis:
        section.sentences.append(_capitalize(_period(plan.thesis)))
        section.provenance.append("thesis")


_SECTION_DRAFTERS: dict[str, Callable[..., None]] = {
    "acknowledge": _draft_acknowledge,
    "invite": _draft_invite,
    "state_share": _draft_state_share,
    "identity": _draft_identity,
    "focus": _draft_focus,
    "close": _draft_close,
    "acknowledge_gap": _draft_acknowledge_gap,
    "question": _draft_question,
    "acknowledge_input": _draft_acknowledge_input,
    "hook": _draft_hook,
    "develop": _draft_develop,
    "ground": _draft_ground,
    "qualify": _draft_qualify,
    "link": _draft_link,
}


# ---------------------------------------------------------------------------
# Helpers shared by drafters.
# ---------------------------------------------------------------------------


def _reflection_dict(reflection: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in reflection:
        if ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _learn_signals(plan: ResponsePlan) -> dict[str, Any]:
    out: dict[str, Any] = {"goals": []}
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
            out["proposition"] = (subj.strip(), rel.strip(), obj.strip())
        elif kind == "goal" and "|" in body:
            key, value = body.split("|", 1)
            out["goals"].append((key.strip(), value.strip()))
        elif kind == "correction":
            out["correction"] = body
    return out


def _period(text: str) -> str:
    cleaned = text.strip().rstrip(".?!")
    return cleaned + "."


def _ensure_question(text: str) -> str:
    cleaned = text.strip().rstrip(".?!")
    return cleaned + "?"
