"""Symbolic inference over Darwin's concept universe.

The reasoner in ``reasoning.py`` walks neighborhoods. The inference engine
in this module *derives* new facts: it composes is_a relations transitively
to answer kind-questions, follows causes chains to build causal stories,
inherits properties down is_a edges, and detects contradictions when two
beliefs cannot both hold.

Every inference is *justified*. An ``Inference`` carries the chain of
graph edges that produced it; Darwin can show its work. That's the
difference between "I think gravity is a force" (lookup) and "Yes:
gravity is_a fundamental_force, and fundamental_force is_a force; both
edges are in my graph" (derivation).

No hardcoded answers. Every operator here works over whatever is in the
universe at the moment. If the universe is empty, the engine returns
empty inferences. If the universe is rich, the engine derives rich
chains.

Operators implemented:
  * ``is_a_chain(a, b)`` — does ``a`` reach ``b`` through is_a edges?
  * ``inherited_properties(c)`` — every property predicated of any
    super-kind of ``c``.
  * ``causal_chain(a, b)`` — does ``a`` cause ``b`` through a chain of
    ``causes`` / ``effect`` / ``derives_from`` edges?
  * ``contradicts(a, b)`` — does the universe assert ``opposes`` or
    incompatible kinds between ``a`` and ``b``?
  * ``explain(a, b, kind)`` — produce a structured proof chain.
  * ``derive_new_relations()`` — propose transitive / inherited edges
    not yet in the graph (these become candidate self-modifications).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.universe.concept_universe import ConceptUniverse, Relation


# Relation kinds the inference engine treats as transitive.
_TRANSITIVE_KINDS: frozenset[str] = frozenset({"is_a", "part_of", "instantiates"})
# Relation kinds that compose into causal chains.
_CAUSAL_KINDS: frozenset[str] = frozenset({"causes", "derives_from", "expresses"})
# Relation kinds asserting negation / opposition.
_OPPOSITION_KINDS: frozenset[str] = frozenset({"opposes", "contradict"})


@dataclass
class Inference:
    """A single derived fact and the proof that supports it."""

    operator: str         # e.g. "is_a_chain", "causal_chain", "inheritance"
    claim: str            # human-readable statement of the inference
    source: str
    target: str
    chain: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "claim": self.claim,
            "source": self.source,
            "target": self.target,
            "chain": list(self.chain),
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


@dataclass
class Contradiction:
    """An inconsistency the engine detected."""

    a: str
    b: str
    reason: str
    chain: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "reason": self.reason,
            "chain": list(self.chain),
        }


class InferenceEngine:
    """Bounded symbolic-reasoning operators over a ConceptUniverse."""

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        max_chain_length: int = 8,
        max_derivations_per_pass: int = 64,
    ) -> None:
        self.universe = universe
        self.max_chain_length = max_chain_length
        self.max_derivations_per_pass = max_derivations_per_pass

    # -- is_a / kind reasoning -------------------------------------------

    def is_a_chain(self, source: str, target: str) -> Inference | None:
        """Does ``source`` reach ``target`` through transitive kind edges?

        Returns an Inference with the full chain, or None if no such chain
        exists within ``max_chain_length`` hops.
        """

        path = self._transitive_path(source, target, _TRANSITIVE_KINDS)
        if not path:
            return None
        return Inference(
            operator="is_a_chain",
            claim=f"{source} is a {target}",
            source=source,
            target=target,
            chain=[rel.to_record() for rel in path],
            confidence=max(0.4, 1.0 - 0.05 * len(path)),
            notes=f"derived via {len(path)} transitive step(s)",
        )

    def super_kinds(self, concept: str, *, limit: int = 32) -> list[str]:
        """Every concept reachable from ``concept`` via transitive is_a edges."""

        return self._transitive_reachable(concept, _TRANSITIVE_KINDS, limit=limit)

    def sub_kinds(self, concept: str, *, limit: int = 32) -> list[str]:
        """Every concept that transitively is_a ``concept``."""

        return self._transitive_reachable_incoming(
            concept, _TRANSITIVE_KINDS, limit=limit
        )

    def inherited_properties(self, concept: str) -> list[Inference]:
        """Every property a super-kind has, inherited down to ``concept``.

        A "property" here is any outgoing edge of kinds ``part_of`` /
        ``requires`` / ``describes`` / ``measured_by`` from a super-kind.
        Inheritance is structural: if dog is_a mammal and mammal has a
        spine, dog inherits the spine relation as a candidate fact.
        """

        if not self.universe.has(concept):
            return []
        property_kinds = {"part_of", "requires", "describes", "measured_by", "expresses"}
        out: list[Inference] = []
        super_chain = self.super_kinds(concept)
        seen_targets: set[tuple[str, str]] = set()
        for sk in super_chain:
            for rel in self.universe.neighbors(sk, kinds=property_kinds):
                key = (rel.kind, rel.target)
                if key in seen_targets:
                    continue
                seen_targets.add(key)
                out.append(
                    Inference(
                        operator="inheritance",
                        claim=f"{concept} inherits the {rel.kind} of {rel.target} from {sk}",
                        source=concept,
                        target=rel.target,
                        chain=[
                            {"step": "super", "concept": sk},
                            rel.to_record(),
                        ],
                        confidence=0.7,
                        notes=f"inherited via {sk}",
                    )
                )
                if len(out) >= self.max_derivations_per_pass:
                    return out
        return out

    # -- causal reasoning ------------------------------------------------

    def causal_chain(self, source: str, target: str) -> Inference | None:
        """Does ``source`` cause ``target`` through a chain of causes-style edges?"""

        path = self._transitive_path(source, target, _CAUSAL_KINDS)
        if not path:
            return None
        return Inference(
            operator="causal_chain",
            claim=f"{source} ultimately causes {target}",
            source=source,
            target=target,
            chain=[rel.to_record() for rel in path],
            confidence=max(0.35, 1.0 - 0.07 * len(path)),
            notes=f"causal chain of length {len(path)}",
        )

    def downstream_effects(self, concept: str, *, limit: int = 16) -> list[str]:
        """Every concept the input transitively causes."""

        return self._transitive_reachable(concept, _CAUSAL_KINDS, limit=limit)

    def upstream_causes(self, concept: str, *, limit: int = 16) -> list[str]:
        """Every concept that transitively causes the input."""

        return self._transitive_reachable_incoming(concept, _CAUSAL_KINDS, limit=limit)

    # -- contradiction detection -----------------------------------------

    def contradicts(self, a: str, b: str) -> Contradiction | None:
        """True iff ``a`` and ``b`` cannot both hold.

        Two checks: (1) is there an explicit opposes / contradict edge
        between them or a super-kind of each? (2) do they belong to two
        sub-kinds of a common parent that explicitly opposes itself?
        """

        if not (self.universe.has(a) and self.universe.has(b)):
            return None
        # Direct opposition.
        for rel in self.universe.neighbors(a, kinds=_OPPOSITION_KINDS):
            if rel.target == b:
                return Contradiction(
                    a=a, b=b, reason="explicit opposition edge",
                    chain=[rel.to_record()],
                )
        for rel in self.universe.neighbors(b, kinds=_OPPOSITION_KINDS):
            if rel.target == a:
                return Contradiction(
                    a=a, b=b, reason="explicit opposition edge (reverse)",
                    chain=[rel.to_record()],
                )
        # Super-kind opposition.
        super_a = set(self.super_kinds(a))
        super_b = set(self.super_kinds(b))
        for sa in super_a:
            for rel in self.universe.neighbors(sa, kinds=_OPPOSITION_KINDS):
                if rel.target in super_b or rel.target == b:
                    return Contradiction(
                        a=a, b=b,
                        reason=f"super-kinds oppose: {sa} opposes {rel.target}",
                        chain=[rel.to_record()],
                    )
        return None

    # -- explanation building --------------------------------------------

    def explain(self, source: str, target: str, *, kind: str = "auto") -> list[Inference]:
        """Build every available proof chain between two concepts."""

        out: list[Inference] = []
        if kind in ("auto", "is_a"):
            inf = self.is_a_chain(source, target)
            if inf is not None:
                out.append(inf)
        if kind in ("auto", "causal"):
            inf = self.causal_chain(source, target)
            if inf is not None:
                out.append(inf)
        if kind in ("auto", "shortest"):
            path = self.universe.shortest_path(source, target, max_hops=self.max_chain_length)
            if path:
                out.append(
                    Inference(
                        operator="shortest_path",
                        claim=f"{source} is connected to {target}",
                        source=source,
                        target=target,
                        chain=[rel.to_record() for rel in path],
                        confidence=max(0.4, 1.0 - 0.08 * len(path)),
                        notes=f"shortest known graph path",
                    )
                )
        return out

    # -- proactive derivation -------------------------------------------

    def derive_new_relations(self) -> list[tuple[str, str, str]]:
        """Propose typed edges the graph does NOT yet contain but should.

        Returns triples (source, kind, target) suitable for ``add_relation``.
        Currently emits:
          * Transitive is_a closure (A is_a B and B is_a C → A is_a C)
          * Causal chain shortcuts (A causes B and B causes C → A causes C)

        Each call is bounded by ``max_derivations_per_pass`` to avoid
        flooding the graph.
        """

        out: list[tuple[str, str, str]] = []
        for concept in self.universe.all_concepts():
            for super_kind in self.super_kinds(concept.name, limit=8)[:4]:
                if super_kind == concept.name:
                    continue
                # Is the direct (concept, is_a, super_kind) edge present?
                if any(
                    rel.target == super_kind and rel.kind == "is_a"
                    for rel in self.universe.neighbors(concept.name)
                ):
                    continue
                out.append((concept.name, "is_a", super_kind))
                if len(out) >= self.max_derivations_per_pass:
                    return out
            for downstream in self.downstream_effects(concept.name, limit=4):
                if downstream == concept.name:
                    continue
                if any(
                    rel.target == downstream and rel.kind == "causes"
                    for rel in self.universe.neighbors(concept.name)
                ):
                    continue
                out.append((concept.name, "causes", downstream))
                if len(out) >= self.max_derivations_per_pass:
                    return out
        return out

    # -- private helpers -------------------------------------------------

    def _transitive_path(
        self, source: str, target: str, kinds: frozenset[str]
    ) -> list[Relation]:
        if not self.universe.has(source) or not self.universe.has(target):
            return []
        if source == target:
            return []
        prev: dict[str, tuple[str, Relation]] = {}
        seen = {source}
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_chain_length:
                continue
            for rel in self.universe.neighbors(node, kinds=kinds):
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

    def _transitive_reachable(
        self, source: str, kinds: frozenset[str], *, limit: int = 32
    ) -> list[str]:
        if not self.universe.has(source):
            return []
        seen: set[str] = set()
        out: list[str] = []
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        while queue and len(out) < limit:
            node, depth = queue.popleft()
            if depth >= self.max_chain_length:
                continue
            for rel in self.universe.neighbors(node, kinds=kinds):
                if rel.target in seen or rel.target == source:
                    continue
                seen.add(rel.target)
                out.append(rel.target)
                queue.append((rel.target, depth + 1))
        return out

    def _transitive_reachable_incoming(
        self, target: str, kinds: frozenset[str], *, limit: int = 32
    ) -> list[str]:
        if not self.universe.has(target):
            return []
        seen: set[str] = set()
        out: list[str] = []
        queue: deque[tuple[str, int]] = deque([(target, 0)])
        while queue and len(out) < limit:
            node, depth = queue.popleft()
            if depth >= self.max_chain_length:
                continue
            for rel in self.universe.neighbors(node, kinds=kinds, include_incoming=True):
                # Only consider edges pointing INTO `node`.
                if rel.target != node:
                    continue
                src = rel.source
                if src in seen or src == target:
                    continue
                seen.add(src)
                out.append(src)
                queue.append((src, depth + 1))
        return out
