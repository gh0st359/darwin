"""ConceptFusion — Darwin integrates statements from chat into its universe.

When the operator says "rain causes flooding", "a photon is a particle", or
"entropy opposes order", that isn't just text. It is a *declarative
statement* — an assertion of a typed relation between two concepts. The
fusion layer parses such statements out of natural-language utterances and
folds them into the concept universe as real edges. Darwin therefore
*learns from chat in real time*, the same way it learns from observation.

Statement patterns recognized:
  * ``X is a Y`` / ``X are Y``                                  → is_a
  * ``X are a kind of Y`` / ``X is a kind of Y``                → is_a
  * ``X causes Y``  / ``X cause Y``                             → causes
  * ``X requires Y`` / ``X needs Y`` / ``X depends on Y``       → requires
  * ``X composes Y`` / ``X is part of Y`` / ``X is made of Y``  → part_of
  * ``X opposes Y`` / ``X contradicts Y`` / ``X is the opposite of Y`` → opposes
  * ``X relates to Y`` / ``X is related to Y`` / ``X and Y are related`` → related_to
  * ``X describes Y``                                            → describes
  * ``X is analogous to Y`` / ``X is like Y``                    → analogous_to

The parser is rule-based and conservative — it only fires on clear
patterns. False negative is preferred over false positive (adding a wrong
edge corrupts reasoning; missing a fuzzy one just means Darwin needs
another mention).

Every fusion is *attributed* — the resulting Relation's notes record that
the edge came from a chat statement, when it happened, and which
utterance produced it. The meta-gate can use this when deciding whether
to roll back a derivation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from darwin.universe.concept_universe import ConceptUniverse


# Compiled patterns. Each (regex, relation_kind, swap_direction).
# swap_direction True means the regex captures (target, source).
# Order matters: more-specific patterns come FIRST so e.g. "is part of"
# wins over the looser "is a".
_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    # is_a forms — most specific first
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+(?:a|an)\s+(?:kind|type|sort|form|instance)\s+of\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "is_a", False),
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:instantiates|is\s+an\s+instance\s+of)\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "instantiates", False),
    # composition / part_of — MUST come before the looser "is a".
    # Article-eating subpattern (?:(?:a|an|the)\s+)? requires the article
    # to be followed by whitespace, preventing 'an' from being mis-captured
    # as 'a' + 'n' of 'norganism'.
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+part\s+of\s+(?:(?:a|an|the)\s+)?(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "part_of", False),
    # "X is made/composed of Y" — semantically Y is part of X, so swap=True.
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+(?:made|composed)\s+of\s+(?:(?:a|an|the)\s+)?(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "part_of", True),
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+composes\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "composes", False),
    # opposition — also more specific than "is a"
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+the\s+opposite\s+of\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "opposes", False),
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:opposes|contradicts)\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "opposes", False),
    # related_to — more specific than "is a"
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+related\s+to\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "related_to", False),
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+relates\s+to\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "related_to", False),
    # analogy — more specific than "is a"
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+analogous\s+to\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "analogous_to", False),
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+like\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "analogous_to", False),
    # generic "X is a Y" — comes AFTER all the specific "is/are" forms
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:is|are)\s+(?:a|an)\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "is_a", False),
    # causation
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:causes|cause)\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "causes", False),
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:leads to|results in|produces)\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "causes", False),
    # requires / dependency
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+(?:requires|needs|depends on)\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "requires", False),
    # describes
    (re.compile(r"\b(?P<src>[a-z][a-z_\- ]{1,40})\s+describes\s+(?P<tgt>[a-z][a-z_\- ]{1,40})\b"), "describes", False),
]


# Words that should never become "concepts" even though they often appear
# as left-hand-side captures (subject pronouns, auxiliaries, common
# function words). A real concept noun is rarely one of these.
_NEVER_CONCEPT = frozenset({
    "it", "this", "that", "these", "those", "i", "you", "we", "they",
    "he", "she", "him", "her", "them", "us", "me",
    "the", "a", "an", "some", "any", "many", "few", "one", "two",
    "there", "here", "now", "then",
    "yes", "no", "ok",
})


@dataclass
class FusedRelation:
    """A relation added to the universe from a chat statement."""

    source: str
    target: str
    kind: str
    surface: str          # the actual text fragment that produced it
    confidence: float = 0.6
    created_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "surface": self.surface,
            "confidence": round(self.confidence, 3),
            "created_at": self.created_at,
        }


@dataclass
class FusionResult:
    text: str
    added: list[FusedRelation] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "added": [r.to_record() for r in self.added],
            "rejected": list(self.rejected),
        }


_LEADING_ARTICLE = re.compile(r"^\s*(?:a|an|the|some|any)\s+", re.IGNORECASE)


def _normalize_phrase(phrase: str) -> str:
    """Lowercase, strip leading articles, collapse whitespace, snake_case."""

    # Strip leading article so "a sparrow" -> "sparrow".
    stripped = _LEADING_ARTICLE.sub("", phrase.strip().lower())
    cleaned = re.sub(r"\s+", "_", stripped)
    return re.sub(r"_+", "_", cleaned).strip("_")


def _is_acceptable_concept_name(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 48:
        return False
    if name in _NEVER_CONCEPT:
        return False
    # Reject anything that's all auxiliaries / numbers / punctuation.
    if not re.match(r"[a-z]", name):
        return False
    return True


class ConceptFusion:
    """Parses declarative statements from chat into typed graph edges."""

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        default_confidence: float = 0.6,
        new_domain: str = "fused",
        bus: Any = None,
    ) -> None:
        self.universe = universe
        self.default_confidence = default_confidence
        self.new_domain = new_domain
        self.bus = bus
        self._fused: list[FusedRelation] = []

    def fuse(self, text: str) -> FusionResult:
        """Scan text for declarative statements and integrate them.

        Questions are skipped: a question is not an assertion. Fusion runs
        only on text that looks declarative.
        """

        result = FusionResult(text=text or "")
        if not text:
            return result
        stripped = text.strip()
        # Questions never assert relations.
        if stripped.endswith("?"):
            return result
        lowered = stripped.lower()
        # Interrogative openers signal a question even without a "?".
        interrogative_openers = (
            "is ", "are ", "was ", "were ", "do ", "does ", "did ",
            "can ", "could ", "would ", "should ", "may ", "might ",
            "what ", "why ", "how ", "when ", "where ", "which ", "who ",
        )
        if any(lowered.startswith(opener) for opener in interrogative_openers):
            return result
        # Strip leading question words so "Tell me that X is a Y" still parses.
        lowered = re.sub(
            r"^\s*(?:tell me that|tell me|did you know|note that|remember that|fyi|btw)\s+",
            "",
            lowered,
        )
        for pattern, kind, swap in _PATTERNS:
            for match in pattern.finditer(lowered):
                src_raw = match.group("src")
                tgt_raw = match.group("tgt")
                if swap:
                    src_raw, tgt_raw = tgt_raw, src_raw
                src = _normalize_phrase(src_raw)
                tgt = _normalize_phrase(tgt_raw)
                if not (_is_acceptable_concept_name(src) and _is_acceptable_concept_name(tgt)):
                    result.rejected.append(f"{src or '_'} —{kind}→ {tgt or '_'}")
                    continue
                if src == tgt:
                    result.rejected.append(f"self-loop {src} —{kind}→ {tgt}")
                    continue
                # Skip if the edge already exists.
                already = any(
                    rel.target == tgt and rel.kind == kind
                    for rel in self.universe.neighbors(src) if self.universe.has(src)
                )
                if already:
                    continue
                surface = match.group(0)
                try:
                    # ensure_concepts=True so brand-new entities get registered.
                    self.universe.add_concept(src, domain=self.new_domain)
                    self.universe.add_concept(tgt, domain=self.new_domain)
                    self.universe.add_relation(
                        src, tgt, kind,
                        weight=self.default_confidence,
                        notes=f"fused from chat: {surface!r}",
                    )
                except Exception:
                    result.rejected.append(f"{src} —{kind}→ {tgt}")
                    continue
                fused = FusedRelation(
                    source=src, target=tgt, kind=kind, surface=surface,
                    confidence=self.default_confidence,
                )
                self._fused.append(fused)
                result.added.append(fused)
        self._publish(result)
        return result

    def fused_count(self) -> int:
        return len(self._fused)

    def recent(self, limit: int = 20) -> list[FusedRelation]:
        return self._fused[-limit:]

    def _publish(self, result: FusionResult) -> None:
        if not result.added or self.bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            self.bus.publish(
                BusTopic.TRANSITIONS,
                {
                    "kind": "concept_fusion",
                    "added": [r.to_record() for r in result.added],
                    "source_text": result.text[:200],
                },
                source="concept_fusion",
            )
        except Exception:
            pass

    def summary(self) -> dict[str, Any]:
        from collections import Counter

        return {
            "total_fused": len(self._fused),
            "by_kind": dict(Counter(f.kind for f in self._fused)),
            "recent": [
                {"source": f.source, "kind": f.kind, "target": f.target}
                for f in self._fused[-8:]
            ],
        }
