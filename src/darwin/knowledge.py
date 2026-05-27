from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Provenance:
    source_type: str
    source_id: str
    extractor: str
    confidence: float
    captured_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "extractor": self.extractor,
            "confidence": self.confidence,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Provenance":
        return cls(
            source_type=str(record.get("source_type", "unknown")),
            source_id=str(record.get("source_id", "")),
            extractor=str(record.get("extractor", "unknown")),
            confidence=float(record.get("confidence", 0.0)),
            captured_at=float(record.get("captured_at", time.time())),
        )


@dataclass
class KnowledgeAtom:
    kind: str
    subject: str
    relation: str
    object: str
    text: str
    provenance: Provenance
    confidence: float = 0.5
    promoted: bool = False
    support_kind: str = "corpus"
    atom_id: str = ""

    def __post_init__(self) -> None:
        if not self.atom_id:
            raw = "|".join(
                [
                    self.kind,
                    self.subject.lower(),
                    self.relation.lower(),
                    self.object.lower(),
                    self.provenance.source_type,
                    self.provenance.source_id,
                ]
            )
            self.atom_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def to_record(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "kind": self.kind,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "text": self.text,
            "confidence": self.confidence,
            "promoted": self.promoted,
            "support_kind": self.support_kind,
            "provenance": self.provenance.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "KnowledgeAtom":
        provenance = record.get("provenance", {})
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        return cls(
            atom_id=str(record.get("atom_id", "")),
            kind=str(record.get("kind", "")),
            subject=str(record.get("subject", "")),
            relation=str(record.get("relation", "")),
            object=str(record.get("object", "")),
            text=str(record.get("text", "")),
            confidence=float(record.get("confidence", 0.0)),
            promoted=bool(record.get("promoted", False)),
            support_kind=str(record.get("support_kind", "corpus")),
            provenance=Provenance.from_record(provenance),
        )


@dataclass
class IngestResult:
    source: str
    source_type: str
    atoms_created: int
    atoms_seen: int


class CorpusIngestor:
    """Deterministic curated-corpus ingestion.

    This is deliberately not an LLM extractor. It turns explicit text
    patterns into provenance-rich atoms that can propose hypotheses but
    cannot become causal beliefs until Darwin tests them.
    """

    def __init__(self, store: Any | None = None) -> None:
        self.store = store

    def ingest(self, path: str | Path, source_type: str = "wikipedia") -> IngestResult:
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        atoms = self.extract(text, source_type=source_type, source_id=str(source))
        created = 0
        if self.store is not None:
            for atom in atoms:
                created += self.store.record_knowledge_atom(atom.to_record())
        return IngestResult(
            source=str(source),
            source_type=source_type,
            atoms_created=created if self.store is not None else len(atoms),
            atoms_seen=len(atoms),
        )

    def extract(self, text: str, *, source_type: str, source_id: str) -> list[KnowledgeAtom]:
        if source_type == "wikidata":
            return self._extract_wikidata(text, source_type=source_type, source_id=source_id)
        return self._extract_text(text, source_type=source_type, source_id=source_id)

    def _extract_text(self, text: str, *, source_type: str, source_id: str) -> list[KnowledgeAtom]:
        atoms: list[KnowledgeAtom] = []
        current_heading = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading = re.match(r"^=+\s*(?P<title>[^=]+?)\s*=+$", line)
            if heading:
                current_heading = heading.group("title").strip()
                continue
            provenance = Provenance(source_type, source_id, "deterministic-text-v1", 0.72)
            atoms.extend(self._atoms_from_sentence(line, current_heading, provenance))
        return self._dedupe(atoms)

    def _extract_wikidata(self, text: str, *, source_type: str, source_id: str) -> list[KnowledgeAtom]:
        atoms: list[KnowledgeAtom] = []
        provenance = Provenance(source_type, source_id, "deterministic-wikidata-v1", 0.8)
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            subject = str(item.get("label") or item.get("id") or "").strip()
            if not subject:
                continue
            description = str(item.get("description", "")).strip()
            if description:
                atoms.append(KnowledgeAtom("definition", subject, "is", description, description, provenance, 0.7))
            aliases = item.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases[:12]:
                    atoms.append(KnowledgeAtom("alias", subject, "alias", str(alias), str(alias), provenance, 0.75))
            claims = item.get("claims", {})
            if isinstance(claims, dict):
                for relation, values in claims.items():
                    if not isinstance(values, list):
                        values = [values]
                    for value in values[:20]:
                        atoms.append(
                            KnowledgeAtom(
                                "relation",
                                subject,
                                str(relation),
                                str(value),
                                f"{subject} {relation} {value}",
                                provenance,
                                0.65,
                            )
                        )
        return self._dedupe(atoms)

    def _atoms_from_sentence(
        self,
        sentence: str,
        heading: str,
        provenance: Provenance,
    ) -> list[KnowledgeAtom]:
        atoms: list[KnowledgeAtom] = []
        subject_hint = heading or ""

        alias_match = re.match(r"aliases?:\s*(?P<aliases>.+)", sentence, flags=re.IGNORECASE)
        if alias_match and subject_hint:
            for alias in re.split(r"[,;]", alias_match.group("aliases")):
                alias = alias.strip()
                if alias:
                    atoms.append(KnowledgeAtom("alias", subject_hint, "alias", alias, sentence, provenance, 0.78))

        definition = re.match(
            r"(?P<subject>[A-Z][A-Za-z0-9 _-]{1,80})\s+(?:is|are)\s+(?P<object>.+?)[.?!]?$",
            sentence,
        )
        is_definition = bool(definition)
        if definition:
            subject = definition.group("subject").strip()
            obj = definition.group("object").strip()
            relation = "is"
            kind = "quantity" if "measured in" in obj.lower() or "quantity" in obj.lower() else "definition"
            atoms.append(KnowledgeAtom(kind, subject, relation, obj, sentence, provenance, 0.7))

        if is_definition:
            return atoms

        causal_patterns = [
            (r"(?P<subject>[A-Z][A-Za-z0-9 _-]{1,80})\s+causes?\s+(?P<object>.+?)[.?!]?$", "causes"),
            (r"(?P<subject>[A-Z][A-Za-z0-9 _-]{1,80})\s+changes?\s+(?P<object>.+?)[.?!]?$", "changes"),
            (r"(?P<subject>[A-Z][A-Za-z0-9 _-]{1,80})\s+affects?\s+(?P<object>.+?)[.?!]?$", "affects"),
            (r"(?P<subject>[A-Z][A-Za-z0-9 _-]{1,80})\s+resists?\s+(?P<object>.+?)[.?!]?$", "resists"),
        ]
        for pattern, relation in causal_patterns:
            match = re.match(pattern, sentence)
            if match:
                atoms.append(
                    KnowledgeAtom(
                        "causal_hypothesis",
                        match.group("subject").strip(),
                        relation,
                        match.group("object").strip(),
                        sentence,
                        provenance,
                        0.62,
                    )
                )
        return atoms

    def _dedupe(self, atoms: Iterable[KnowledgeAtom]) -> list[KnowledgeAtom]:
        unique: dict[str, KnowledgeAtom] = {}
        for atom in atoms:
            unique.setdefault(atom.atom_id, atom)
        return list(unique.values())


class KnowledgeGraph:
    def __init__(self, atoms: Iterable[KnowledgeAtom] = ()) -> None:
        self.atoms = list(atoms)

    @classmethod
    def from_store(cls, store: Any) -> "KnowledgeGraph":
        return cls(KnowledgeAtom.from_record(record) for record in store.load_knowledge_atoms())

    def search(self, query: str, limit: int = 10) -> list[KnowledgeAtom]:
        terms = {term.lower() for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", query) if len(term) > 2}
        if not terms:
            return []
        scored: list[tuple[int, KnowledgeAtom]] = []
        for atom in self.atoms:
            haystack = f"{atom.subject} {atom.relation} {atom.object} {atom.text}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, atom))
        scored.sort(key=lambda item: (item[0], item[1].confidence, item[1].promoted), reverse=True)
        return [atom for _score, atom in scored[:limit]]

    def causal_hypotheses(self) -> list[KnowledgeAtom]:
        return [atom for atom in self.atoms if atom.kind == "causal_hypothesis"]

    def promoted_atoms(self) -> list[KnowledgeAtom]:
        return [atom for atom in self.atoms if atom.promoted]

    def relations_for(self, subject: str) -> list[KnowledgeAtom]:
        """Return ``relation`` atoms (Wikidata-style triples) about a subject."""

        normalized = subject.lower().strip()
        if not normalized:
            return []
        return [
            atom
            for atom in self.atoms
            if atom.kind == "relation" and atom.subject.lower() == normalized
        ]

    def quantities_for(self, subject: str) -> list[KnowledgeAtom]:
        """Return ``quantity`` atoms (definitions describing measurable units)."""

        normalized = subject.lower().strip()
        if not normalized:
            return []
        return [
            atom
            for atom in self.atoms
            if atom.kind == "quantity" and atom.subject.lower() == normalized
        ]

    def definitions_for(self, subject: str) -> list[KnowledgeAtom]:
        normalized = subject.lower().strip()
        if not normalized:
            return []
        return [
            atom
            for atom in self.atoms
            if atom.kind == "definition" and atom.subject.lower() == normalized
        ]
