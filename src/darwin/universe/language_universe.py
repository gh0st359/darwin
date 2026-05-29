"""Language grounding: mapping the words Darwin reads to concept nodes.

The grounder is the bridge between raw chat text and the concept universe.
When the user types "tell me about gravity", the grounder maps "gravity"
to the ``gravity`` concept node so the conceptual reasoner can expand its
neighborhood for the response. When the user types a word Darwin has never
seen, the grounder *creates* a concept under a sensible domain so it has a
place in the universe — Darwin's vocabulary grows from conversation.

Grounding strategies (in order):
  1. Exact match on normalized name.
  2. Alias match (concepts can declare alternative spellings / synonyms).
  3. Substring/contains match against existing concept names.
  4. Embedding-space cosine if a CausalEmbeddingSpace is attached.
  5. Fallback: register a fresh concept under the configured ``new_domain``.

The grounder is intentionally additive. It never *removes* concepts. Bad
groundings (a wrong domain, an unhelpful alias) are corrected by the
meta-gate via the same proposal grammar everything else uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.universe.concept_universe import ConceptUniverse


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as", "that",
    "this", "these", "those", "it", "its", "you", "i", "we", "they", "them",
    "us", "what", "how", "why", "when", "where", "do", "does", "did", "can",
    "could", "would", "should", "will", "may", "might", "shall", "have",
    "has", "had", "not", "no", "yes", "so", "if", "then", "else", "than",
    "about", "tell", "me", "more", "very", "much", "really", "just",
})


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']{1,}")


def tokenize(text: str) -> list[str]:
    """Lowercase, alpha-only tokens; punctuation stripped."""

    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def content_words(text: str) -> list[str]:
    """Tokens minus stopwords."""

    return [tok for tok in tokenize(text) if tok not in _STOPWORDS and len(tok) > 2]


@dataclass
class GroundedTerm:
    surface: str            # the word as typed by the user
    concept_name: str       # the canonical concept name it grounded to
    domain: str
    method: str             # how we grounded it: exact / alias / fuzzy / embedding / new
    confidence: float = 0.5
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "concept": self.concept_name,
            "domain": self.domain,
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


@dataclass
class GroundingResult:
    text: str
    grounded: list[GroundedTerm] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)

    @property
    def concept_names(self) -> list[str]:
        # Dedup while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for term in self.grounded:
            if term.concept_name not in seen:
                seen.add(term.concept_name)
                out.append(term.concept_name)
        return out

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "grounded": [g.to_record() for g in self.grounded],
            "unrecognized": list(self.unrecognized),
            "concepts": self.concept_names,
        }


class LanguageGrounder:
    """Maps words in user text to concepts in Darwin's universe.

    Optionally attached to a CausalEmbeddingSpace; when present, fuzzy
    grounding uses cosine similarity over concept-name embeddings. The
    embedding space is *not required* — pure-string matching still works.
    """

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        embedding_space: Any = None,
        new_domain: str = "general",
        auto_register: bool = True,
        fuzzy_threshold: float = 0.55,
    ) -> None:
        self.universe = universe
        self.embedding_space = embedding_space
        self.new_domain = new_domain
        self.auto_register = auto_register
        self.fuzzy_threshold = fuzzy_threshold
        self._alias_index: dict[str, str] = {}
        self._rebuild_alias_index()

    def _rebuild_alias_index(self) -> None:
        self._alias_index = {}
        for concept in self.universe.all_concepts():
            self._alias_index[concept.name] = concept.name
            for alias in concept.aliases:
                key = ConceptUniverse._normalize(alias)
                self._alias_index[key] = concept.name

    def ground(self, text: str) -> GroundingResult:
        """Ground every content word in a piece of text."""

        result = GroundingResult(text=text)
        words = content_words(text)
        seen: set[str] = set()
        for word in words:
            if word in seen:
                continue
            seen.add(word)
            term = self._ground_one(word)
            if term is None:
                result.unrecognized.append(word)
            else:
                result.grounded.append(term)
        return result

    def _ground_one(self, word: str) -> GroundedTerm | None:
        normalized = ConceptUniverse._normalize(word)

        # 1. Exact concept name.
        concept = self.universe.get(normalized)
        if concept is not None:
            return GroundedTerm(
                surface=word, concept_name=concept.name, domain=concept.domain,
                method="exact", confidence=0.95,
            )

        # 2. Alias match.
        target = self._alias_index.get(normalized)
        if target is not None:
            concept = self.universe.get(target)
            if concept is not None:
                return GroundedTerm(
                    surface=word, concept_name=concept.name, domain=concept.domain,
                    method="alias", confidence=0.85,
                )

        # 3. Substring/contains match. Both directions: the word is contained
        # in a concept name (e.g. "phys" → "physics_concept"), or a concept
        # name is contained in the word (e.g. "gravitational" → "gravity"
        # ... only if the contained name is at least 4 chars).
        for candidate in self.universe.all_concepts():
            cn = candidate.name
            if len(normalized) >= 4 and normalized in cn:
                return GroundedTerm(
                    surface=word, concept_name=cn, domain=candidate.domain,
                    method="substring", confidence=0.6,
                )
            if len(cn) >= 4 and cn in normalized:
                return GroundedTerm(
                    surface=word, concept_name=cn, domain=candidate.domain,
                    method="substring", confidence=0.6,
                )

        # 4. Embedding-space fuzzy match (optional).
        if self.embedding_space is not None:
            try:
                fuzzy = self._embedding_match(normalized)
                if fuzzy is not None:
                    return fuzzy
            except Exception:
                pass

        # 5. Fallback: register a new concept in the configured domain.
        if self.auto_register:
            new_concept = self.universe.add_concept(
                normalized,
                domain=self.new_domain,
                definition=f"Concept derived from conversation: {word!r}.",
                salience=0.7,
            )
            self._alias_index[new_concept.name] = new_concept.name
            return GroundedTerm(
                surface=word, concept_name=new_concept.name,
                domain=new_concept.domain, method="new", confidence=0.4,
                notes="registered during grounding",
            )
        return None

    def _embedding_match(self, normalized: str) -> GroundedTerm | None:
        space = self.embedding_space
        from darwin.mysterio.embeddings import cosine

        try:
            qvec = space.embed(f"concept:{normalized}")
        except Exception:
            return None
        best_name: str | None = None
        best_score: float = -1.0
        for concept in self.universe.all_concepts():
            try:
                kvec = space.embed(f"concept:{concept.name}")
            except Exception:
                continue
            score = cosine(qvec, kvec)
            if score > best_score:
                best_score = score
                best_name = concept.name
        if best_name is None or best_score < self.fuzzy_threshold:
            return None
        concept = self.universe.expect(best_name)
        return GroundedTerm(
            surface=normalized, concept_name=concept.name, domain=concept.domain,
            method="embedding", confidence=float(best_score),
        )

    def refresh(self) -> None:
        """Re-read the universe's aliases (call after bulk additions)."""

        self._rebuild_alias_index()
