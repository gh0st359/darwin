from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from darwin.types import Action


STOPWORDS = {
    "a",
    "about",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "not",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "then",
    "this",
    "to",
    "us",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class Grounding:
    kind: str
    name: str
    text: str
    confidence: float = 1.0

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SemanticProposition:
    kind: str
    subject: str
    relation: str
    object: str
    polarity: bool = True
    confidence: float = 0.5
    evidence: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "polarity": self.polarity,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class SemanticFrame:
    source: str
    original_text: str
    normalized_text: str
    tokens: list[str]
    speech_act: str
    topic: str
    intent: str
    confidence: float
    uncertainty: float
    groundings: list[Grounding] = field(default_factory=list)
    propositions: list[SemanticProposition] = field(default_factory=list)
    goals: dict[str, Any] = field(default_factory=dict)
    instructions: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=dict)
    hypotheses: list[SemanticProposition] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.confidence < 0.42 or (not self.groundings and not self.propositions and len(self.tokens) > 4)

    def summary(self) -> str:
        grounded = ", ".join(f"{item.kind}:{item.name}" for item in self.groundings[:4]) or "none"
        propositions = "; ".join(
            f"{item.subject} {item.relation} {item.object}" for item in self.propositions[:3]
        ) or "none"
        return (
            f"source={self.source} act={self.speech_act} topic={self.topic} "
            f"confidence={self.confidence:.2f} groundings={grounded} propositions={propositions}"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "tokens": self.tokens,
            "speech_act": self.speech_act,
            "topic": self.topic,
            "intent": self.intent,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "groundings": [item.to_record() for item in self.groundings],
            "propositions": [item.to_record() for item in self.propositions],
            "goals": self.goals,
            "instructions": self.instructions,
            "questions": self.questions,
            "corrections": self.corrections,
            "values": self.values,
            "hypotheses": [item.to_record() for item in self.hypotheses],
            "unknown_terms": self.unknown_terms,
            "needs_clarification": self.needs_clarification,
            "summary": self.summary(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SemanticFrame":
        return cls(
            source=str(record.get("source", "unknown")),
            original_text=str(record.get("original_text", "")),
            normalized_text=str(record.get("normalized_text", "")),
            tokens=list(record.get("tokens", [])),
            speech_act=str(record.get("speech_act", "statement")),
            topic=str(record.get("topic", "general")),
            intent=str(record.get("intent", "conversation")),
            confidence=float(record.get("confidence", 0.0)),
            uncertainty=float(record.get("uncertainty", 1.0)),
            groundings=[
                Grounding(
                    kind=str(item.get("kind", "")),
                    name=str(item.get("name", "")),
                    text=str(item.get("text", "")),
                    confidence=float(item.get("confidence", 0.0)),
                )
                for item in record.get("groundings", [])
            ],
            propositions=[
                SemanticProposition(
                    kind=str(item.get("kind", "claim")),
                    subject=str(item.get("subject", "")),
                    relation=str(item.get("relation", "")),
                    object=str(item.get("object", "")),
                    polarity=bool(item.get("polarity", True)),
                    confidence=float(item.get("confidence", 0.0)),
                    evidence=str(item.get("evidence", "")),
                )
                for item in record.get("propositions", [])
            ],
            goals=dict(record.get("goals", {})),
            instructions=list(record.get("instructions", [])),
            questions=list(record.get("questions", [])),
            corrections=list(record.get("corrections", [])),
            values=dict(record.get("values", {})),
            hypotheses=[
                SemanticProposition(
                    kind=str(item.get("kind", "hypothesis")),
                    subject=str(item.get("subject", "")),
                    relation=str(item.get("relation", "")),
                    object=str(item.get("object", "")),
                    polarity=bool(item.get("polarity", True)),
                    confidence=float(item.get("confidence", 0.0)),
                    evidence=str(item.get("evidence", "")),
                )
                for item in record.get("hypotheses", [])
            ],
            unknown_terms=list(record.get("unknown_terms", [])),
        )


class SemanticParser:
    """Rule-based parser that grounds language into Darwin's internal symbols."""

    def __init__(self) -> None:
        self.action_aliases: dict[str, set[str]] = {
            "open_curtains": {"open curtains", "open the curtains", "let light in", "let daylight in"},
            "close_curtains": {"close curtains", "close the curtains", "block daylight", "make it dark"},
            "toggle_switch": {"toggle switch", "flip switch", "turn on light", "turn off light", "switch light"},
            "replace_fuse": {"replace fuse", "fix fuse", "restore fuse", "repair fuse"},
            "overload_circuit": {"overload circuit", "break fuse", "stress circuit"},
            "wait": {"wait", "do nothing", "pause"},
            "chat_with_user": {"chat", "talk", "speak", "conversation"},
        }
        self.variable_aliases: dict[str, set[str]] = {
            "room_bright": {"room bright", "bright room", "brightness", "light", "darkness", "dark"},
            "fuse_intact": {"fuse intact", "fuse", "circuit intact", "circuit"},
            "curtains_open": {"curtains open", "curtains", "curtain"},
            "switch_on": {"switch on", "switch", "light switch"},
            "battery_charge": {"battery", "battery charge", "charge", "power"},
            "daylight": {"daylight", "sunlight", "day"},
            "language_understanding": {
                "natural language",
                "language",
                "understanding",
                "understand",
                "meaning",
                "semantics",
                "symbolic",
                "conceptual",
            },
            "self_model": {"self model", "metacognition", "self awareness", "introspection"},
            "world_model": {"world model", "reality", "world", "cause and effect", "causality"},
        }
        self.topic_keywords: dict[str, set[str]] = {
            "language": {
                "language",
                "meaning",
                "understand",
                "semantics",
                "parser",
                "word",
                "sentence",
                "grounding",
                "grounded",
                "regurgitation",
                "parroting",
                "imitation",
                "repeating",
            },
            "learning": {"learn", "training", "teach", "memory", "dataset", "data"},
            "planning": {"plan", "goal", "future", "strategy", "simulate"},
            "experiments": {"experiment", "test", "uncertain", "hypothesis", "prove"},
            "self": {"self", "mind", "thinking", "aware", "metacognition", "consciousness"},
            "causality": {"cause", "effect", "because", "consequence", "causal"},
            "vision": {"agi", "asi", "intelligence", "darwin", "frontier", "brain"},
            "tools": {"tool", "web", "scrape", "huggingface", "dataset", "api"},
        }

    def parse(
        self,
        text: str,
        *,
        source: str = "user",
        actions: Iterable[Action] = (),
        known_concepts: Iterable[str] = (),
        known_variables: Iterable[str] = (),
    ) -> SemanticFrame:
        normalized = self._normalize(text)
        tokens = self._tokens(normalized)
        groundings = self._ground(normalized, tokens, actions, known_concepts, known_variables)
        propositions = self._extract_propositions(normalized)
        goals = self._extract_goals(normalized, groundings)
        instructions = self._extract_instructions(normalized, groundings)
        questions = self._extract_questions(text, normalized)
        corrections = self._extract_corrections(normalized)
        values = self._extract_values(normalized)
        hypotheses = [item for item in propositions if item.kind == "hypothesis"]
        speech_act = self._speech_act(normalized, questions, corrections, goals, instructions, hypotheses)
        topic = self._topic(tokens, groundings)
        intent = self._intent(speech_act, normalized)
        unknown_terms = self._unknown_terms(tokens, groundings)
        confidence = self._confidence(
            tokens=tokens,
            groundings=groundings,
            propositions=propositions,
            goals=goals,
            instructions=instructions,
            questions=questions,
            corrections=corrections,
            values=values,
        )

        return SemanticFrame(
            source=source,
            original_text=text,
            normalized_text=normalized,
            tokens=tokens,
            speech_act=speech_act,
            topic=topic,
            intent=intent,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            groundings=groundings,
            propositions=propositions,
            goals=goals,
            instructions=instructions,
            questions=questions,
            corrections=corrections,
            values=values,
            hypotheses=hypotheses,
            unknown_terms=unknown_terms,
        )

    def _ground(
        self,
        normalized: str,
        tokens: list[str],
        actions: Iterable[Action],
        known_concepts: Iterable[str],
        known_variables: Iterable[str],
    ) -> list[Grounding]:
        groundings: list[Grounding] = []
        seen: set[tuple[str, str]] = set()

        action_names = {action.name for action in actions}
        for action_name in action_names | set(self.action_aliases):
            aliases = set(self.action_aliases.get(action_name, set()))
            aliases.add(action_name.replace("_", " "))
            self._append_matches(groundings, seen, normalized, "action", action_name, aliases)

        variable_names = set(known_variables) | set(self.variable_aliases)
        for variable in variable_names:
            aliases = set(self.variable_aliases.get(variable, set()))
            aliases.add(variable.replace("_", " "))
            self._append_matches(groundings, seen, normalized, "variable", variable, aliases)

        for concept in known_concepts:
            cleaned = concept.replace(":", " ").replace("_", " ").replace("=", " ").replace("'", "")
            words = cleaned.split()
            if 1 <= len(words) <= 8 and all(word not in STOPWORDS for word in words[:1]):
                self._append_matches(groundings, seen, normalized, "concept", concept, {cleaned})

        for word in {"agi", "asi", "darwin", "brain", "consciousness", "dataset", "huggingface"}:
            if word in tokens and ("concept", word) not in seen:
                groundings.append(Grounding(kind="concept", name=word, text=word, confidence=0.8))
                seen.add(("concept", word))

        return groundings

    def _append_matches(
        self,
        groundings: list[Grounding],
        seen: set[tuple[str, str]],
        normalized: str,
        kind: str,
        name: str,
        aliases: Iterable[str],
    ) -> None:
        for alias in sorted(aliases, key=lambda value: (-len(value), value)):
            alias = alias.strip().lower()
            if not alias:
                continue
            pattern = r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])"
            if re.search(pattern, normalized) and (kind, name) not in seen:
                confidence = 1.0 if alias == name.replace("_", " ") else 0.85
                groundings.append(Grounding(kind=kind, name=name, text=alias, confidence=confidence))
                seen.add((kind, name))
                return

    def _extract_propositions(self, normalized: str) -> list[SemanticProposition]:
        propositions: list[SemanticProposition] = []

        for match in re.finditer(r"\bif (?P<subject>.+?) then (?P<object>[^.?!;]+)", normalized):
            propositions.append(
                SemanticProposition(
                    kind="hypothesis",
                    subject=match.group("subject").strip(),
                    relation="implies",
                    object=match.group("object").strip(),
                    confidence=0.72,
                    evidence=match.group(0),
                )
            )

        causal_patterns = [
            (r"(?P<subject>[^.?!;]+?)\bcauses?\b(?P<object>[^.?!;]+)", "causes"),
            (r"(?P<subject>[^.?!;]+?)\bleads? to\b(?P<object>[^.?!;]+)", "causes"),
            (r"(?P<object>[^.?!;]+?)\bbecause\b(?P<subject>[^.?!;]+)", "explains"),
            (r"(?P<subject>[^.?!;]+?)\bprevents?\b(?P<object>[^.?!;]+)", "prevents"),
        ]
        for pattern, relation in causal_patterns:
            for match in re.finditer(pattern, normalized):
                propositions.append(
                    SemanticProposition(
                        kind="hypothesis" if relation in {"causes", "prevents", "explains"} else "claim",
                        subject=match.group("subject").strip(),
                        relation=relation,
                        object=match.group("object").strip(),
                        confidence=0.68,
                        evidence=match.group(0),
                    )
                )

        definitional_patterns = [
            (r"(?P<subject>[a-z0-9_ -]{2,80})\bmeans\b(?P<object>[^.?!;]+)", "means"),
            (r"(?P<subject>[a-z0-9_ -]{2,80})\bis\b(?P<object>[^.?!;]+)", "is"),
            (r"(?P<subject>[a-z0-9_ -]{2,80})\bare\b(?P<object>[^.?!;]+)", "are"),
        ]
        for pattern, relation in definitional_patterns:
            for match in re.finditer(pattern, normalized):
                subject = match.group("subject").strip()
                obj = match.group("object").strip()
                if subject and obj and subject not in STOPWORDS:
                    propositions.append(
                        SemanticProposition(
                            kind="definition" if relation == "means" else "claim",
                            subject=subject,
                            relation=relation,
                            object=obj,
                            confidence=0.58,
                            evidence=match.group(0),
                        )
                    )

        return self._dedupe_propositions(propositions)

    def _extract_goals(self, normalized: str, groundings: list[Grounding]) -> dict[str, Any]:
        goals: dict[str, Any] = {}
        goal_cues = (
            "i want",
            "we want",
            "i need",
            "we need",
            "goal is",
            "the goal",
            "my goal",
            "our goal",
            "must",
            "should",
            "make sure",
        )
        if not any(cue in normalized for cue in goal_cues):
            return goals

        if any(word in normalized for word in ["bright", "brightness", "light", "not dark"]):
            goals["room_bright"] = True
        if "dark" in normalized and "not dark" not in normalized and "don't want dark" not in normalized:
            goals["room_bright"] = False
        if any(phrase in normalized for phrase in ["fuse intact", "fix fuse", "working circuit"]):
            goals["fuse_intact"] = True
        if any(phrase in normalized for phrase in ["natural language", "understand language", "meaning"]):
            goals["language_understanding"] = "increase"
        if "always running" in normalized or "24/7" in normalized:
            goals["continuous_cognition"] = True
        if "not an llm" in normalized or "don't want an llm" in normalized or "do not want an llm" in normalized:
            goals["llm_dependency"] = False

        for grounding in groundings:
            if grounding.kind == "variable" and grounding.name not in goals:
                goals[grounding.name] = "attend"
        return goals

    def _extract_instructions(self, normalized: str, groundings: list[Grounding]) -> list[str]:
        instructions: list[str] = []
        verbs = [
            "build",
            "implement",
            "run",
            "test",
            "remember",
            "learn",
            "teach",
            "explain",
            "show",
            "think",
            "plan",
            "parse",
        ]
        for verb in verbs:
            if re.search(rf"\b{verb}\b", normalized):
                instructions.append(verb)
        for grounding in groundings:
            if grounding.kind == "action":
                instructions.append(f"consider_action:{grounding.name}")
        return sorted(set(instructions))

    def _extract_questions(self, original: str, normalized: str) -> list[str]:
        questions: list[str] = []
        if "?" in original:
            questions.extend(part.strip() for part in re.split(r"\?", original) if part.strip())
        if re.match(r"^(what|why|how|when|where|who|can|could|should|would|is|are|do|does)\b", normalized):
            questions.append(original.strip())
        return list(dict.fromkeys(questions))

    def _extract_corrections(self, normalized: str) -> list[str]:
        corrections: list[str] = []
        cues = ["actually", "no ", "wrong", "instead", "rather than"]
        if any(cue in normalized for cue in cues):
            for clause in re.split(r"[.;]", normalized):
                if any(cue in clause for cue in cues):
                    corrections.append(clause.strip())
        return corrections

    def _extract_values(self, normalized: str) -> dict[str, float]:
        values: dict[str, float] = {}
        value_patterns = {
            "importance": ["important", "essential", "must", "need", "critical", "huge"],
            "preference": ["prefer", "want", "like"],
            "rejection": ["do not want", "don't want", "not an llm", "not just"],
            "trust": ["truth", "truthfully", "honest", "actual", "true"],
            "autonomy": ["always running", "proactive", "self-adapt", "self learn", "self-evolve"],
        }
        for value, cues in value_patterns.items():
            score = sum(1 for cue in cues if cue in normalized)
            if score:
                values[value] = min(1.0, 0.35 + 0.2 * score)
        return values

    def _speech_act(
        self,
        normalized: str,
        questions: list[str],
        corrections: list[str],
        goals: dict[str, Any],
        instructions: list[str],
        hypotheses: list[SemanticProposition],
    ) -> str:
        if questions:
            return "question"
        if hypotheses:
            return "hypothesis"
        if goals:
            return "goal"
        if any(item in instructions for item in ["remember", "learn", "teach"]):
            return "teaching"
        if corrections:
            return "correction"
        if instructions:
            return "directive"
        if any(phrase in normalized for phrase in ["i think", "i believe", "my view", "truth is"]):
            return "claim"
        return "statement"

    def _topic(self, tokens: list[str], groundings: list[Grounding]) -> str:
        token_set = set(tokens)
        scored: Counter[str] = Counter()
        for topic, keywords in self.topic_keywords.items():
            scored[topic] += len(token_set & keywords)
        for grounding in groundings:
            if grounding.name in {"language_understanding"}:
                scored["language"] += 3
            elif grounding.name in {"world_model"}:
                scored["causality"] += 2
            elif grounding.kind == "action":
                scored["experiments"] += 1
        if not scored:
            return "general"
        return scored.most_common(1)[0][0]

    def _intent(self, speech_act: str, normalized: str) -> str:
        if speech_act in {"question", "correction", "goal", "teaching", "directive", "hypothesis"}:
            return speech_act
        if any(phrase in normalized for phrase in ["i am proud", "good job", "doing great"]):
            return "encouragement"
        return "conversation"

    def _unknown_terms(self, tokens: list[str], groundings: list[Grounding]) -> list[str]:
        grounded_words = set()
        for grounding in groundings:
            grounded_words.update(self._tokens(grounding.text))

        terms = []
        for token in tokens:
            if len(token) < 5 or token in STOPWORDS or token in grounded_words:
                continue
            if token.isdigit():
                continue
            terms.append(token)
        return sorted(set(terms))[:12]

    def _confidence(
        self,
        *,
        tokens: list[str],
        groundings: list[Grounding],
        propositions: list[SemanticProposition],
        goals: dict[str, Any],
        instructions: list[str],
        questions: list[str],
        corrections: list[str],
        values: dict[str, float],
    ) -> float:
        if not tokens:
            return 0.0
        score = 0.18
        score += min(0.28, 0.08 * len(groundings))
        score += min(0.22, 0.07 * len(propositions))
        score += min(0.14, 0.06 * len(goals))
        score += min(0.12, 0.04 * len(instructions))
        score += 0.06 if questions else 0.0
        score += 0.05 if corrections else 0.0
        score += min(0.08, 0.03 * len(values))
        if len(tokens) <= 3 and not groundings:
            score -= 0.12
        return max(0.0, min(0.98, score))

    def _normalize(self, text: str) -> str:
        normalized = text.lower().strip()
        replacements = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "i'm": "i am",
            "it's": "it is",
            "we're": "we are",
        }
        for original, replacement in replacements.items():
            normalized = normalized.replace(original, replacement)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*|\d+(?:\.\d+)?", text.lower())

    def _dedupe_propositions(self, propositions: list[SemanticProposition]) -> list[SemanticProposition]:
        seen: set[tuple[str, str, str, bool]] = set()
        unique: list[SemanticProposition] = []
        for proposition in propositions:
            key = (
                proposition.subject,
                proposition.relation,
                proposition.object,
                proposition.polarity,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(proposition)
        return unique


class SemanticMemory:
    """Stores parsed meaning, not just raw conversation text."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self.frames: list[SemanticFrame] = []
        self.propositions: Counter[tuple[str, str, str]] = Counter()
        self.goals: dict[str, Any] = {}
        self.values: Counter[str] = Counter()
        self.unknown_terms: Counter[str] = Counter()
        self.by_topic: dict[str, list[SemanticFrame]] = defaultdict(list)
        self.by_act: dict[str, list[SemanticFrame]] = defaultdict(list)

    def learn(self, frame: SemanticFrame) -> None:
        self.frames.append(frame)
        if len(self.frames) > self.capacity:
            self.frames = self.frames[-self.capacity :]

        self.by_topic[frame.topic].append(frame)
        self.by_act[frame.speech_act].append(frame)
        for proposition in frame.propositions:
            self.propositions[(proposition.subject, proposition.relation, proposition.object)] += 1
        for key, value in frame.goals.items():
            self.goals[key] = value
        for key, score in frame.values.items():
            self.values[key] += max(1, int(score * 10))
        for term in frame.unknown_terms:
            self.unknown_terms[term] += 1

    def recent(self, limit: int = 10, source: str | None = None) -> list[SemanticFrame]:
        frames = self.frames if source is None else [frame for frame in self.frames if frame.source == source]
        return frames[-limit:]

    def load_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            self.learn(SemanticFrame.from_record(record))

    def summary(self) -> str:
        top_topic = self._top_key(self.by_topic)
        top_act = self._top_key(self.by_act)
        top_value = self.values.most_common(1)[0][0] if self.values else "none"
        top_unknown = self.unknown_terms.most_common(1)[0][0] if self.unknown_terms else "none"
        return (
            f"semantic_frames={len(self.frames)} top_topic={top_topic} "
            f"top_act={top_act} top_value={top_value} top_unknown={top_unknown}"
        )

    def active_goals(self) -> dict[str, Any]:
        return dict(self.goals)

    def meaning_records(self, limit: int = 10) -> list[dict[str, Any]]:
        return [frame.to_record() for frame in self.recent(limit)]

    def _top_key(self, mapping: Mapping[str, list[SemanticFrame]]) -> str:
        if not mapping:
            return "none"
        return max(mapping.items(), key=lambda item: len(item[1]))[0]
