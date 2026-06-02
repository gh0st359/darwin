"""Hand-rolled pure-Python NL parser for ingestion.

Tokenizer → POS tagger (rule-based with a frequency-weighted fallback)
→ a lightweight dependency-style relation extractor → a basic NER pass
on capitalised proper nouns. No LLM. No external dependencies.

The grammar is small but tractable. It handles the high-value sentence
shapes the operator's text corpora contain:

  * "X is/are a/the Y."         → (X, is_a, Y)
  * "X is/are part of Y."       → (X, part_of, Y)
  * "X causes/leads to/produces Y." → (X, causes, Y)
  * "X requires/needs Y."        → (X, requires, Y)
  * "X opposes/contradicts Y."   → (X, opposes, Y)
  * "X relates to / is related to Y." → (X, related_to, Y)
  * "X describes Y."             → (X, describes, Y)
  * "X is composed/made of Y."   → (X, has_part, Y)  [swap]
  * "X is an instance of Y."     → (X, instantiates, Y)

False negatives are preferred over false positives — a missed
extraction is fine; a wrong fact corrupts the universe.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


# Tokenization handles punctuation, contractions, and hyphenated words.
_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z\-']{0,40}|\d+(?:\.\d+)?|[.,;!?:]",
)

# POS lexicon — a small seed of high-confidence tags. Anything not in
# here is tagged by suffix heuristics or defaulted to NN.
_POS_LEXICON: dict[str, str] = {
    # Determiners
    "the": "DT", "a": "DT", "an": "DT", "some": "DT", "any": "DT",
    "this": "DT", "that": "DT", "these": "DT", "those": "DT",
    # Auxiliaries / copulas
    "is": "VBZ", "are": "VBP", "was": "VBD", "were": "VBD",
    "be": "VB", "been": "VBN", "being": "VBG",
    "has": "VBZ", "have": "VBP", "had": "VBD",
    "do": "VBP", "does": "VBZ", "did": "VBD",
    # Common verbs (the ones we relate over)
    "causes": "VBZ", "cause": "VBP", "caused": "VBD",
    "produces": "VBZ", "produce": "VBP", "produced": "VBD",
    "leads": "VBZ", "lead": "VBP", "led": "VBD",
    "requires": "VBZ", "require": "VBP", "required": "VBD",
    "needs": "VBZ", "need": "VBP", "needed": "VBD",
    "opposes": "VBZ", "oppose": "VBP",
    "contradicts": "VBZ", "contradict": "VBP",
    "relates": "VBZ", "relate": "VBP",
    "describes": "VBZ", "describe": "VBP",
    "composes": "VBZ", "compose": "VBP",
    "consists": "VBZ", "consist": "VBP", "consisted": "VBD",
    "depends": "VBZ", "depend": "VBP", "depended": "VBD",
    # Prepositions
    "of": "IN", "in": "IN", "on": "IN", "at": "IN", "by": "IN",
    "for": "IN", "with": "IN", "from": "IN", "to": "IN", "as": "IN",
    "about": "IN", "into": "IN", "onto": "IN", "between": "IN",
    # Conjunctions
    "and": "CC", "or": "CC", "but": "CC", "yet": "CC", "so": "CC",
    # Pronouns
    "i": "PRP", "you": "PRP", "he": "PRP", "she": "PRP", "it": "PRP",
    "we": "PRP", "they": "PRP", "them": "PRP", "us": "PRP",
    # Negation / modifiers
    "not": "RB", "very": "RB", "also": "RB", "actually": "RB",
    "really": "RB", "much": "RB", "more": "RB", "less": "RB",
    # Punctuation
    ".": ".", ",": ",", ";": ";", "!": ".", "?": ".", ":": ":",
}


# Verbs that map directly to relation kinds.
_VERB_TO_RELATION: dict[str, str] = {
    "is": "is_a",
    "are": "is_a",
    "was": "is_a",
    "were": "is_a",
    "causes": "causes", "cause": "causes", "caused": "causes",
    "produces": "causes", "produce": "causes", "produced": "causes",
    "leads": "causes", "lead": "causes", "led": "causes",
    "requires": "requires", "require": "requires", "required": "requires",
    "needs": "requires", "need": "requires", "needed": "requires",
    "depends": "requires", "depend": "requires", "depended": "requires",
    "consists": "has_part", "consist": "has_part",
    "opposes": "opposes", "oppose": "opposes",
    "contradicts": "opposes", "contradict": "opposes",
    "relates": "related_to", "relate": "related_to",
    "describes": "describes", "describe": "describes",
    "composes": "composes", "compose": "composes",
}


@dataclass
class Token:
    surface: str
    pos: str = ""
    is_capitalised: bool = False


def tokenize(text: str) -> list[Token]:
    """Return a flat list of tokens. Splits on whitespace + punctuation."""

    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(text or ""):
        raw = match.group(0)
        tokens.append(Token(
            surface=raw,
            is_capitalised=raw[0].isupper() if raw[:1].isalpha() else False,
        ))
    return tokens


def _suffix_pos(surface: str) -> str:
    """Heuristic POS by suffix when the lexicon doesn't know."""

    lower = surface.lower()
    if lower in (".", ",", "!", "?", ";", ":"):
        return "."
    if lower.endswith("ly"):
        return "RB"
    if lower.endswith(("ing", "ed")):
        return "VBG" if lower.endswith("ing") else "VBN"
    if lower.endswith(("ness", "tion", "ity", "ment", "ence", "ance")):
        return "NN"
    if lower.endswith(("ous", "ful", "ive", "able", "ible", "ic", "al")):
        return "JJ"
    if lower.endswith("'s"):
        return "POS"
    return "NN"


def pos_tag(tokens: Iterable[Token]) -> list[Token]:
    """Annotate each token with a POS in-place. Returns the list."""

    out = list(tokens)
    for token in out:
        lower = token.surface.lower()
        if lower in _POS_LEXICON:
            token.pos = _POS_LEXICON[lower]
        else:
            token.pos = _suffix_pos(token.surface)
    return out


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks. Conservative."""

    if not text:
        return []
    # Normalise whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    # Sentence boundary: punctuation followed by whitespace then capital.
    parts = _SENT_END.split(text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# NER — capitalisation + gazetteer placeholder
# --------------------------------------------------------------------------- #


_LOWERCASE_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "of", "in", "on", "at", "by", "for", "with", "from", "to", "as",
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "be", "been", "being", "has", "have", "had", "not", "very",
})


def named_entities(tokens: Iterable[Token]) -> list[str]:
    """Identify named entities by capitalisation + non-function-word filter."""

    out: list[str] = []
    current: list[str] = []
    for token in tokens:
        if (
            token.is_capitalised
            and token.surface.lower() not in _LOWERCASE_FUNCTION_WORDS
            and token.pos in ("NN", "NNP", "NNS")
        ):
            current.append(token.surface)
        else:
            if current:
                out.append(" ".join(current))
                current = []
    if current:
        out.append(" ".join(current))
    return out


# --------------------------------------------------------------------------- #
# Fact extraction
# --------------------------------------------------------------------------- #


@dataclass
class Fact:
    """A (subject, predicate, object) triple extracted from text."""

    subject: str
    predicate: str
    object: str
    confidence: float = 0.7
    source_sentence: str = ""

    def to_record(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": round(self.confidence, 3),
            "source_sentence": self.source_sentence[:200],
        }


_LEADING_DETERMINER = re.compile(
    r"^\s*(?:a|an|the|some|any|this|that|these|those)\s+",
    re.IGNORECASE,
)


def _normalise_phrase(words: list[str]) -> str:
    """Lowercase, strip leading determiner, snake_case."""

    if not words:
        return ""
    raw = " ".join(words).strip().lower()
    raw = _LEADING_DETERMINER.sub("", raw)
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return cleaned


def _find_main_verb(tokens: list[Token]) -> int:
    """Index of the first relation-mapping verb, or -1."""

    for i, t in enumerate(tokens):
        if t.surface.lower() in _VERB_TO_RELATION:
            return i
    return -1


def _extract_predicate(tokens: list[Token], verb_idx: int) -> tuple[str, int]:
    """Return (relation_kind, index just after the predicate)."""

    verb = tokens[verb_idx].surface.lower()
    base_relation = _VERB_TO_RELATION.get(verb, "related_to")
    # Recognise multi-word verbs that change the relation kind.
    if verb in ("is", "are", "was", "were") and verb_idx + 1 < len(tokens):
        next_tok = tokens[verb_idx + 1].surface.lower()
        if next_tok == "part" and verb_idx + 2 < len(tokens) \
                and tokens[verb_idx + 2].surface.lower() == "of":
            return "part_of", verb_idx + 3
        if next_tok in ("made", "composed") and verb_idx + 2 < len(tokens) \
                and tokens[verb_idx + 2].surface.lower() == "of":
            return "has_part", verb_idx + 3  # swap direction at caller
        if next_tok == "an" and verb_idx + 2 < len(tokens) \
                and tokens[verb_idx + 2].surface.lower() == "instance" \
                and verb_idx + 3 < len(tokens) \
                and tokens[verb_idx + 3].surface.lower() == "of":
            return "instantiates", verb_idx + 4
        if next_tok in ("related", "analogous", "opposed") \
                and verb_idx + 2 < len(tokens) \
                and tokens[verb_idx + 2].surface.lower() == "to":
            kind = {
                "related": "related_to",
                "analogous": "analogous_to",
                "opposed": "opposes",
            }[next_tok]
            return kind, verb_idx + 3
        if next_tok in ("the", "a", "an"):
            kind = "is_a"
            if verb_idx + 2 < len(tokens) \
                    and tokens[verb_idx + 2].surface.lower() == "opposite" \
                    and verb_idx + 3 < len(tokens) \
                    and tokens[verb_idx + 3].surface.lower() == "of":
                return "opposes", verb_idx + 4
            return kind, verb_idx + 1
    if verb in ("leads",) and verb_idx + 1 < len(tokens) \
            and tokens[verb_idx + 1].surface.lower() == "to":
        return "causes", verb_idx + 2
    if verb == "consists" and verb_idx + 1 < len(tokens) \
            and tokens[verb_idx + 1].surface.lower() == "of":
        return "has_part", verb_idx + 2
    if verb == "depends" and verb_idx + 1 < len(tokens) \
            and tokens[verb_idx + 1].surface.lower() == "on":
        return "requires", verb_idx + 2
    return base_relation, verb_idx + 1


def _collect_noun_phrase(
    tokens: list[Token], start: int, end_exclusive: int | None = None,
) -> tuple[list[str], int]:
    """Walk forward from ``start`` collecting noun-phrase tokens.

    Returns (words, index just past the last consumed token).
    """

    end = end_exclusive if end_exclusive is not None else len(tokens)
    words: list[str] = []
    i = start
    # Skip leading determiners by consuming them silently — they survive
    # _normalise_phrase's stripping anyway.
    while i < end:
        t = tokens[i]
        if t.pos == "." or t.surface in (",", ";"):
            break
        if t.pos in ("CC",):
            # "and" / "or" — break the phrase here so we don't accidentally
            # eat the second clause.
            break
        if t.pos in ("VBZ", "VBP", "VBD", "VB", "VBN", "VBG") and words:
            break
        if t.surface.lower() in _VERB_TO_RELATION and words:
            break
        # Keep determiners, adjectives, nouns, hyphenated forms.
        if t.pos in ("DT", "JJ", "NN", "NNS", "NNP", "PRP", "POS", "RB"):
            words.append(t.surface)
        else:
            # Unknown POS — keep going only if we haven't started yet.
            if not words:
                words.append(t.surface)
            else:
                break
        i += 1
    return words, i


def extract_facts(sentence: str) -> list[Fact]:
    """Extract zero or more Facts from one sentence."""

    if not sentence:
        return []
    tokens = pos_tag(tokenize(sentence))
    # Drop trailing punctuation token for parsing convenience.
    while tokens and tokens[-1].pos == ".":
        tokens.pop()
    if len(tokens) < 3:
        return []
    verb_idx = _find_main_verb(tokens)
    if verb_idx == -1:
        return []
    # Subject is everything before the verb.
    subj_words, _ = _collect_noun_phrase(tokens, 0, verb_idx)
    if not subj_words:
        return []
    # Predicate (relation kind) + the index where the object begins.
    relation_kind, obj_start = _extract_predicate(tokens, verb_idx)
    obj_words, _ = _collect_noun_phrase(tokens, obj_start)
    if not obj_words:
        return []
    subj = _normalise_phrase(subj_words)
    obj = _normalise_phrase(obj_words)
    if not (subj and obj) or subj == obj:
        return []
    # has_part is the surface predicate's swapped form of part_of.
    if relation_kind == "has_part":
        subj, obj = obj, subj
        relation_kind = "part_of"
    return [Fact(
        subject=subj,
        predicate=relation_kind,
        object=obj,
        confidence=0.7,
        source_sentence=sentence,
    )]


class NLParser:
    """Pipeline orchestrator wrapping the stand-alone functions."""

    def __init__(self) -> None:
        self.sentence_count = 0
        self.fact_count = 0

    def parse(self, text: str) -> list[Fact]:
        """Parse text → list of Facts."""

        out: list[Fact] = []
        for sentence in sentences(text):
            self.sentence_count += 1
            facts = extract_facts(sentence)
            self.fact_count += len(facts)
            out.extend(facts)
        return out


__all__ = [
    "Fact",
    "NLParser",
    "Token",
    "extract_facts",
    "named_entities",
    "pos_tag",
    "sentences",
    "tokenize",
]
