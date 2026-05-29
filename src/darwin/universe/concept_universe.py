"""The internal universe: Darwin's concept graph.

This is where Darwin lives. Not in a room with curtains and a fuse — in a
graph of concepts spanning physics, mathematics, chemistry, biology, arts,
mind, language, and computing. Concepts are nodes. Relations are edges.
Domains are labelled subgraphs. Darwin's neural network IS this graph + the
self-trained causal embedding space + the causal model over conceptual
transitions.

The universe is *live*: Darwin can add concepts it encounters, hypothesize
new relations, compose new nodes from existing ones, and reflect on its own
conceptual structure as part of cognition. This is not a frozen knowledge
base. It is a working substrate.

Design commitments:
  * **No pretrained embeddings.** Vectors are learned online by the v6
    CausalEmbeddingSpace from the actual concept neighborhoods Darwin
    traverses. The seed graph supplies *names and structure*, not weights.
  * **Cross-domain by default.** Music ↔ math, chemistry ↔ physics, mind ↔
    language — the relations span domains so reasoning naturally crosses
    them.
  * **Self-extending.** ``add_concept`` and ``add_relation`` are the same
    API used by the bootstrap, by the live language grounder, and by the
    conceptual composer. There is no privileged "seed-time" entry point.
  * **First-class introspection.** ``ConceptUniverse.summary`` /
    ``neighborhood`` / ``shortest_path`` / ``walk`` are the lenses Darwin's
    own reasoner and the brain terminal use to surface what Darwin is
    actually thinking about.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass
class Concept:
    """A single node in Darwin's universe."""

    name: str
    domain: str = "general"
    definition: str = ""
    depth: int = 0  # 0 = foundational, higher = derived
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    derived_from: tuple[str, ...] = ()
    salience: float = 1.0
    visits: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "definition": self.definition,
            "depth": self.depth,
            "aliases": list(self.aliases),
            "examples": list(self.examples),
            "derived_from": list(self.derived_from),
            "salience": round(self.salience, 4),
            "visits": self.visits,
            "created_at": self.created_at,
        }

    def short_label(self) -> str:
        return f"{self.name} [{self.domain}]"


# Canonical relation kinds. The reasoner reads these to choose how to expand
# a concept's neighborhood. New kinds may be added at runtime.
RELATION_KINDS: tuple[str, ...] = (
    "is_a",            # taxonomic specialization (electron is_a particle)
    "part_of",         # composition (nucleus part_of atom)
    "composes",        # inverse of part_of for surface ergonomics
    "requires",        # depends on (combustion requires oxygen)
    "causes",          # forward causation (heat causes expansion)
    "opposes",         # antonymy / counter-force (entropy opposes order)
    "analogous_to",    # cross-domain mapping (wave analogous_to oscillation)
    "instantiates",    # concrete example of (water instantiates liquid)
    "measured_by",     # operationalization (energy measured_by joules)
    "related_to",      # generic association
    "derives_from",    # historical / mathematical derivation
    "expresses",       # one form gives rise to another (music expresses emotion)
    "describes",       # a tool that describes (equation describes physics)
)


@dataclass
class Relation:
    """A typed, weighted edge in the concept graph."""

    source: str
    target: str
    kind: str = "related_to"
    weight: float = 1.0
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "weight": round(self.weight, 4),
            "notes": self.notes,
        }


@dataclass
class Domain:
    """A labelled subgraph: 'physics', 'math', 'arts', etc."""

    name: str
    description: str = ""
    concept_names: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "count": len(self.concept_names),
        }


# --------------------------------------------------------------------------- #
# ConceptUniverse — the actual graph
# --------------------------------------------------------------------------- #


class ConceptUniverse:
    """Darwin's internal universe of concepts.

    Thread-safe under a single reentrant lock. The graph is held as
    name→Concept plus per-source adjacency lists keyed by relation kind.

    The universe is *append-mostly*. Concepts and relations can be added at
    any time. Removal is supported but discouraged at runtime; rollback of a
    learned mistake should go through the meta-gate, not by surgical
    deletion.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._concepts: dict[str, Concept] = {}
        self._domains: dict[str, Domain] = {}
        # adjacency[source] -> list[Relation] (all kinds in one list keeps
        # walks cheap; the reasoner filters by kind itself).
        self._adjacency: dict[str, list[Relation]] = defaultdict(list)
        # reverse adjacency for fast incoming-edge queries
        self._reverse: dict[str, list[Relation]] = defaultdict(list)
        self._created_at = time.time()
        self._growth_events: list[tuple[float, str]] = []

    # -- additions ----------------------------------------------------------

    def add_domain(self, name: str, description: str = "") -> Domain:
        with self._lock:
            domain = self._domains.get(name)
            if domain is None:
                domain = Domain(name=name, description=description)
                self._domains[name] = domain
            elif description and not domain.description:
                domain.description = description
            return domain

    def add_concept(
        self,
        name: str,
        *,
        domain: str = "general",
        definition: str = "",
        depth: int = 0,
        aliases: Iterable[str] = (),
        examples: Iterable[str] = (),
        derived_from: Iterable[str] = (),
        salience: float = 1.0,
    ) -> Concept:
        name = self._normalize(name)
        with self._lock:
            existing = self._concepts.get(name)
            if existing is not None:
                # Concept already in universe; enrich it.
                if definition and not existing.definition:
                    existing.definition = definition
                if examples:
                    existing.examples = tuple(
                        dict.fromkeys((*existing.examples, *examples))
                    )
                if aliases:
                    existing.aliases = tuple(
                        dict.fromkeys((*existing.aliases, *aliases))
                    )
                existing.salience = max(existing.salience, salience)
                return existing
            self.add_domain(domain)
            concept = Concept(
                name=name,
                domain=domain,
                definition=definition,
                depth=depth,
                aliases=tuple(aliases),
                examples=tuple(examples),
                derived_from=tuple(derived_from),
                salience=salience,
            )
            self._concepts[name] = concept
            self._domains[domain].concept_names.append(name)
            self._growth_events.append((time.time(), f"+concept:{name}"))
            return concept

    def add_relation(
        self,
        source: str,
        target: str,
        kind: str = "related_to",
        *,
        weight: float = 1.0,
        notes: str = "",
        ensure_concepts: bool = False,
    ) -> Relation:
        source = self._normalize(source)
        target = self._normalize(target)
        with self._lock:
            if ensure_concepts:
                self.add_concept(source)
                self.add_concept(target)
            if source not in self._concepts:
                raise KeyError(f"unknown source concept: {source!r}")
            if target not in self._concepts:
                raise KeyError(f"unknown target concept: {target!r}")
            relation = Relation(
                source=source, target=target, kind=kind, weight=weight, notes=notes
            )
            self._adjacency[source].append(relation)
            self._reverse[target].append(relation)
            self._growth_events.append((time.time(), f"+rel:{source}-{kind}-{target}"))
            return relation

    def add_relations(self, edges: Iterable[tuple[str, str, str]]) -> int:
        """Bulk-add (source, kind, target) triples. Returns count added."""

        count = 0
        for source, kind, target in edges:
            try:
                self.add_relation(source, target, kind)
                count += 1
            except KeyError:
                continue
        return count

    # -- lookups ------------------------------------------------------------

    def has(self, name: str) -> bool:
        return self._normalize(name) in self._concepts

    def get(self, name: str) -> Concept | None:
        with self._lock:
            return self._concepts.get(self._normalize(name))

    def expect(self, name: str) -> Concept:
        concept = self.get(name)
        if concept is None:
            raise KeyError(f"no such concept: {name!r}")
        return concept

    def all_concepts(self) -> list[Concept]:
        with self._lock:
            return list(self._concepts.values())

    def by_domain(self, domain: str) -> list[Concept]:
        with self._lock:
            domain_obj = self._domains.get(domain)
            if domain_obj is None:
                return []
            return [self._concepts[name] for name in domain_obj.concept_names]

    def domains(self) -> list[Domain]:
        with self._lock:
            return list(self._domains.values())

    def neighbors(
        self,
        name: str,
        *,
        kinds: Iterable[str] | None = None,
        include_incoming: bool = False,
    ) -> list[Relation]:
        name = self._normalize(name)
        with self._lock:
            outgoing = list(self._adjacency.get(name, ()))
            if include_incoming:
                outgoing.extend(self._reverse.get(name, ()))
            if kinds:
                kind_set = set(kinds)
                outgoing = [rel for rel in outgoing if rel.kind in kind_set]
            return outgoing

    def relations(self) -> list[Relation]:
        with self._lock:
            return [rel for rels in self._adjacency.values() for rel in rels]

    # -- traversal ----------------------------------------------------------

    def neighborhood(
        self, name: str, *, hops: int = 2, max_nodes: int = 64
    ) -> dict[str, Any]:
        """The k-hop neighborhood around a concept, BFS, capped by ``max_nodes``."""

        start = self._normalize(name)
        if start not in self._concepts:
            return {"center": start, "nodes": [], "edges": []}
        visited: set[str] = {start}
        nodes: list[Concept] = [self._concepts[start]]
        edges: list[Relation] = []
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        with self._lock:
            while queue and len(nodes) < max_nodes:
                node, depth = queue.popleft()
                if depth >= hops:
                    continue
                for rel in self._adjacency.get(node, ()):
                    edges.append(rel)
                    if rel.target not in visited:
                        visited.add(rel.target)
                        target_concept = self._concepts.get(rel.target)
                        if target_concept is not None:
                            nodes.append(target_concept)
                            queue.append((rel.target, depth + 1))
                            if len(nodes) >= max_nodes:
                                break
        return {
            "center": start,
            "nodes": [c.to_record() for c in nodes],
            "edges": [e.to_record() for e in edges],
        }

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 8,
    ) -> list[Relation]:
        """Edge-weighted BFS shortest path; returns the relation chain or []."""

        source = self._normalize(source)
        target = self._normalize(target)
        if source == target or source not in self._concepts or target not in self._concepts:
            return []
        prev: dict[str, tuple[str, Relation]] = {}
        seen: set[str] = {source}
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        with self._lock:
            while queue:
                node, depth = queue.popleft()
                if depth >= max_hops:
                    continue
                for rel in self._adjacency.get(node, ()):
                    if rel.target in seen:
                        continue
                    seen.add(rel.target)
                    prev[rel.target] = (node, rel)
                    if rel.target == target:
                        chain: list[Relation] = []
                        cur = target
                        while cur in prev:
                            parent, edge = prev[cur]
                            chain.append(edge)
                            cur = parent
                        chain.reverse()
                        return chain
                    queue.append((rel.target, depth + 1))
        return []

    def walk(
        self,
        start: str,
        *,
        steps: int = 6,
        rng=None,
    ) -> list[Concept]:
        """Random walk from a concept, visiting at most ``steps`` neighbors.

        Used by the reasoner for analogical search and by the brain
        terminal's reflective output. Updates ``visits`` per node so
        salience can drift toward the regions Darwin actually traverses.
        """

        import random

        rng = rng or random.Random()
        start = self._normalize(start)
        path: list[Concept] = []
        node = start
        with self._lock:
            for _ in range(steps):
                concept = self._concepts.get(node)
                if concept is None:
                    break
                concept.visits += 1
                path.append(concept)
                outgoing = self._adjacency.get(node, [])
                if not outgoing:
                    break
                chosen = rng.choices(
                    outgoing, weights=[max(0.01, rel.weight) for rel in outgoing], k=1
                )[0]
                node = chosen.target
        return path

    # -- introspection ------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            relation_counts: dict[str, int] = defaultdict(int)
            for rels in self._adjacency.values():
                for rel in rels:
                    relation_counts[rel.kind] += 1
            return {
                "concepts": len(self._concepts),
                "domains": len(self._domains),
                "relations": sum(len(r) for r in self._adjacency.values()),
                "domain_sizes": {
                    name: len(d.concept_names) for name, d in self._domains.items()
                },
                "relation_kinds": dict(relation_counts),
                "growth_events": len(self._growth_events),
                "age_seconds": time.time() - self._created_at,
            }

    def recent_growth(self, limit: int = 20) -> list[str]:
        with self._lock:
            return [event for _, event in self._growth_events[-limit:]]

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace(" ", "_")
