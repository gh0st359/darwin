"""CCGLexicon — a growing inventory of word-to-category mappings.

Each ``LexicalEntry`` ties a surface word form to a CCG category and the
concept(s) it can refer to. The lexicon grows by observation: as the
LanguageGrounder, ConceptFusion, and (later) IngestPipeline encounter
words, they call ``register(concept, surface, category)``. Persisted to
``data_dir() / "darwin_lexicon.json"`` so vocabulary survives restarts.

The lexicon also seeds a small core of structural function words (the,
a, is, are, of, in, etc.) so the surface realizer has something to work
with from turn 1.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darwin.speech.ccg import CCGCategory, N, NP, S


@dataclass
class LexicalEntry:
    """One (surface form, CCG category, concept) tuple in the lexicon."""

    surface: str
    category: str             # serialised CCG category (e.g. "N", "S\\NP/NP")
    concept: str = ""          # the universe concept this entry refers to
    pos: str = ""              # part-of-speech tag, e.g. "NN" / "VB" / "JJ"
    frequency: int = 0
    last_used_at: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "category": self.category,
            "concept": self.concept,
            "pos": self.pos,
            "frequency": self.frequency,
            "last_used_at": self.last_used_at,
            "created_at": self.created_at,
        }


# Core function-word seed. Tiny — just enough for the surface realizer
# to glue thoughts together. Concept names left empty: these aren't
# concepts.
_CORE_FUNCTION_WORDS: list[tuple[str, str, str]] = [
    # determiners
    ("the", "NP/N", "DT"),
    ("a", "NP/N", "DT"),
    ("an", "NP/N", "DT"),
    # copulas
    ("is", "S\\NP/NP", "VBZ"),
    ("are", "S\\NP/NP", "VBP"),
    ("was", "S\\NP/NP", "VBD"),
    # prepositions
    ("of", "PREP", "IN"),
    ("in", "PREP", "IN"),
    ("by", "PREP", "IN"),
    ("on", "PREP", "IN"),
    ("at", "PREP", "IN"),
    # conjunctions
    ("and", "S\\S/S", "CC"),
    ("but", "S\\S/S", "CC"),
    # negation
    ("not", "S/S", "RB"),
    # discourse markers (the synthesizer uses these)
    ("also", "S/S", "RB"),
    ("furthermore", "S/S", "RB"),
    ("moreover", "S/S", "RB"),
]


class CCGLexicon:
    """A growing collection of LexicalEntries keyed by surface form."""

    def __init__(self) -> None:
        self._by_surface: dict[str, list[LexicalEntry]] = {}
        self._by_concept: dict[str, list[LexicalEntry]] = {}
        self._seeded = False
        self._seed_core()

    def _seed_core(self) -> None:
        for surface, category, pos in _CORE_FUNCTION_WORDS:
            self.register(
                surface=surface, category=category, pos=pos,
            )
        self._seeded = True

    def register(
        self,
        *,
        surface: str,
        category: str | CCGCategory = "N",
        concept: str = "",
        pos: str = "",
    ) -> LexicalEntry:
        """Add or update a lexical entry. Returns the entry."""

        category_str = str(category)
        surface_norm = surface.strip().lower()
        if not surface_norm:
            raise ValueError("surface must be non-empty")
        bucket = self._by_surface.setdefault(surface_norm, [])
        # Update existing entry if category + concept already present.
        for entry in bucket:
            if entry.category == category_str and entry.concept == concept:
                entry.frequency += 1
                entry.last_used_at = time.time()
                return entry
        entry = LexicalEntry(
            surface=surface_norm,
            category=category_str,
            concept=concept,
            pos=pos,
            frequency=1,
            last_used_at=time.time(),
        )
        bucket.append(entry)
        if concept:
            self._by_concept.setdefault(concept, []).append(entry)
        return entry

    def lookup(self, surface: str) -> list[LexicalEntry]:
        return list(self._by_surface.get(surface.strip().lower(), ()))

    def lookup_concept(self, concept: str) -> list[LexicalEntry]:
        return list(self._by_concept.get(concept, ()))

    def surface_for_concept(self, concept: str) -> str:
        """Return the surface form to use for ``concept``.

        Strategy: prefer the most-recently-used entry that names the
        concept; otherwise fall back to the concept's name with
        underscores → spaces.
        """

        entries = self._by_concept.get(concept, [])
        if entries:
            entries.sort(key=lambda e: (e.last_used_at, e.frequency), reverse=True)
            entry = entries[0]
            entry.frequency += 1
            entry.last_used_at = time.time()
            return entry.surface
        # Fallback: humanise the concept name.
        return concept.replace("_", " ")

    def total_entries(self) -> int:
        return sum(len(v) for v in self._by_surface.values())

    def total_surfaces(self) -> int:
        return len(self._by_surface)

    def total_concepts(self) -> int:
        return len(self._by_concept)

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> bool:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "entries": [
                    entry.to_record()
                    for bucket in self._by_surface.values()
                    for entry in bucket
                ],
            }
            fd, tmp = tempfile.mkstemp(
                prefix="lex_", suffix=".json", dir=str(target.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as h:
                json.dump(payload, h, separators=(",", ":"))
            os.replace(tmp, target)
            return True
        except OSError:
            return False

    def load(self, path: str | Path) -> int:
        source = Path(path)
        if not source.exists():
            return 0
        try:
            with source.open("r", encoding="utf-8") as h:
                payload = json.load(h)
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(payload, dict):
            return 0
        added = 0
        for record in payload.get("entries", []) or []:
            try:
                surface = record["surface"]
                # Skip if this surface+category+concept already exists.
                bucket = self._by_surface.get(surface, [])
                key = (record.get("category", ""), record.get("concept", ""))
                if any((e.category, e.concept) == key for e in bucket):
                    continue
                entry = LexicalEntry(
                    surface=surface,
                    category=record.get("category", "N"),
                    concept=record.get("concept", ""),
                    pos=record.get("pos", ""),
                    frequency=int(record.get("frequency", 0) or 0),
                    last_used_at=float(record.get("last_used_at", 0.0) or 0.0),
                    created_at=float(record.get("created_at", time.time()) or time.time()),
                )
                self._by_surface.setdefault(surface, []).append(entry)
                if entry.concept:
                    self._by_concept.setdefault(entry.concept, []).append(entry)
                added += 1
            except Exception:
                continue
        return added


def default_lexicon_path() -> Path:
    from darwin.paths import data_dir

    return data_dir() / "darwin_lexicon.json"


__all__ = ["CCGLexicon", "LexicalEntry", "default_lexicon_path"]
