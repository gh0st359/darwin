"""Conceptual reasoning over Darwin's universe.

The reasoner is how Darwin thinks. Given a question or topic, it expands
the relevant concept neighborhood, finds bridges between concepts the user
mentioned, hunts for analogies across domains, and composes the result
into a structured ``ReasoningTrace`` the discourse planner can render.

This is the path that turns "what is the relationship between music and
mathematics?" into an answer that walks ``music → harmony → ratio →
mathematics`` and surfaces actual concepts the user can engage with.

The reasoner is *bounded*: every method has a hard step limit so unbounded
graph walks cannot stall the chat loop. It is *bus-aware*: every nontrivial
reasoning event publishes onto ``BusTopic.SIMULATIONS`` so the brain
terminal watches Darwin reason live.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.universe.concept_universe import Concept, ConceptUniverse, Relation


@dataclass
class ReasoningStep:
    kind: str             # "expand" / "bridge" / "analogy" / "compose" / "reflect"
    summary: str
    concepts: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "concepts": list(self.concepts),
            "relations": list(self.relations),
            "domains": list(self.domains),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ReasoningTrace:
    query: str
    seed_concepts: list[str] = field(default_factory=list)
    steps: list[ReasoningStep] = field(default_factory=list)
    suggested_answer_points: list[str] = field(default_factory=list)
    coverage: float = 0.0   # 0..1 — how much the relevant neighborhood was touched

    def add(self, step: ReasoningStep) -> None:
        self.steps.append(step)

    def visited_concepts(self) -> list[str]:
        seen: list[str] = []
        seen_set: set[str] = set()
        for step in self.steps:
            for name in step.concepts:
                if name not in seen_set:
                    seen.add(name)
                    seen_set.add(name)
                    seen.append(name) if name not in seen else None
        return seen

    def to_record(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "seed_concepts": list(self.seed_concepts),
            "steps": [s.to_record() for s in self.steps],
            "answer_points": list(self.suggested_answer_points),
            "coverage": round(self.coverage, 3),
        }


class ConceptualReasoner:
    """Multi-hop, multi-step reasoner over the ConceptUniverse.

    Reasoning is a sequence of bounded operations:
      * expand(C): pull neighborhoods around each seed concept.
      * bridge(A, B): find the shortest relational path between two concepts.
      * analogy(A): find a concept in a different domain with a similar
        relational signature.
      * compose(A, B): propose a new concept that synthesizes A and B.
      * reflect(C): emit an introspective comment about why C is salient.
    """

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        embedding_space: Any = None,
        bus: Any = None,
        rng_seed: int = 13,
    ) -> None:
        self.universe = universe
        self.embedding_space = embedding_space
        self.bus = bus
        self._rng = random.Random(rng_seed)

    # -- top-level reasoning loop ------------------------------------------

    def think(
        self,
        query: str,
        seeds: Iterable[str],
        *,
        max_hops: int = 2,
        bridge_limit: int = 6,
        rollouts: int = 0,
    ) -> ReasoningTrace:
        """Run a full reasoning pass over ``seeds`` and return a trace.

        ``seeds`` is the list of concept names the language grounder
        extracted from the user's text. ``max_hops`` bounds neighborhood
        expansion. ``bridge_limit`` caps how many pairwise bridges we try
        to find. ``rollouts`` (optional) runs that many random walks for
        flavor / analogy hunting.
        """

        seed_list = list(dict.fromkeys(seeds))
        trace = ReasoningTrace(query=query, seed_concepts=list(seed_list))

        # 1. Expand each seed's neighborhood.
        expansion_concepts: list[str] = []
        for seed in seed_list:
            step = self._expand(seed, hops=max_hops)
            if step is not None:
                trace.add(step)
                expansion_concepts.extend(step.concepts)

        # 2. Bridge every pair of seeds (capped).
        bridges_made = 0
        for i, a in enumerate(seed_list):
            for b in seed_list[i + 1:]:
                if bridges_made >= bridge_limit:
                    break
                step = self._bridge(a, b)
                if step is not None:
                    trace.add(step)
                    bridges_made += 1

        # 3. Hunt cross-domain analogies for each seed.
        for seed in seed_list[:4]:
            step = self._analogy(seed)
            if step is not None:
                trace.add(step)

        # 4. Optional random rollouts for breadth.
        for seed in seed_list[: min(2, rollouts)]:
            step = self._rollout(seed, steps=4)
            if step is not None:
                trace.add(step)

        # 5. Reflection on the most-visited concept.
        if expansion_concepts:
            top = self._highest_salience(expansion_concepts)
            step = self._reflect(top)
            if step is not None:
                trace.add(step)

        # 6. Build the answer points the discourse planner will render.
        trace.suggested_answer_points = self._build_answer_points(trace)
        trace.coverage = self._coverage(trace)

        self._publish(trace)
        return trace

    # -- individual operations ---------------------------------------------

    def _expand(self, name: str, *, hops: int = 2) -> ReasoningStep | None:
        if not self.universe.has(name):
            return None
        nbhd = self.universe.neighborhood(name, hops=hops, max_nodes=24)
        if not nbhd["nodes"]:
            return None
        node_names = [n["name"] for n in nbhd["nodes"]]
        domains = sorted({n["domain"] for n in nbhd["nodes"]})
        center = self.universe.expect(name)
        defn = center.definition or f"a concept named {name}"
        summary = (
            f"{name}: {defn}. Neighborhood covers {len(node_names)} concept(s) "
            f"across {len(domains)} domain(s): {', '.join(domains)}."
        )
        return ReasoningStep(
            kind="expand",
            summary=summary,
            concepts=node_names[:16],
            relations=nbhd["edges"][:24],
            domains=domains,
            confidence=0.8,
        )

    def _bridge(self, source: str, target: str) -> ReasoningStep | None:
        path = self.universe.shortest_path(source, target, max_hops=8)
        if not path:
            return None
        path_concepts = [path[0].source]
        for rel in path:
            path_concepts.append(rel.target)
        domains = sorted({
            self.universe.expect(name).domain for name in path_concepts
        })
        relation_chain = " → ".join(
            f"{rel.source} —{rel.kind}→ {rel.target}" for rel in path
        )
        summary = (
            f"bridge {source} ↔ {target} via {len(path)} step(s): {relation_chain}. "
            f"Path spans {len(domains)} domain(s)."
        )
        return ReasoningStep(
            kind="bridge",
            summary=summary,
            concepts=path_concepts,
            relations=[rel.to_record() for rel in path],
            domains=domains,
            confidence=0.85 if len(path) <= 4 else 0.6,
        )

    def _analogy(self, name: str) -> ReasoningStep | None:
        center = self.universe.get(name)
        if center is None:
            return None
        center_domain = center.domain
        center_neighbors = {
            rel.target for rel in self.universe.neighbors(name, kinds=["is_a", "part_of", "describes", "causes"])
        }
        if not center_neighbors:
            return None
        # Score every other concept by neighborhood-kind overlap, restricted
        # to a different domain (cross-domain analogies are the interesting ones).
        best_name: str | None = None
        best_score = 0.0
        for candidate in self.universe.all_concepts():
            if candidate.name == name or candidate.domain == center_domain:
                continue
            cand_kinds = {
                rel.kind for rel in self.universe.neighbors(candidate.name)
            }
            center_kinds = {
                rel.kind for rel in self.universe.neighbors(name)
            }
            if not cand_kinds or not center_kinds:
                continue
            kind_overlap = len(cand_kinds & center_kinds) / max(
                1, len(cand_kinds | center_kinds)
            )
            # Reward analogous_to / is_a edges that already link the two.
            if any(
                rel.target == candidate.name for rel in self.universe.neighbors(
                    name, kinds=["analogous_to", "is_a", "instantiates"]
                )
            ):
                kind_overlap += 0.4
            if kind_overlap > best_score:
                best_score = kind_overlap
                best_name = candidate.name
        if best_name is None or best_score < 0.2:
            return None
        analog = self.universe.expect(best_name)
        summary = (
            f"{name} (in {center.domain}) is analogous to {analog.name} "
            f"(in {analog.domain}); both show similar relational shape."
        )
        return ReasoningStep(
            kind="analogy",
            summary=summary,
            concepts=[name, best_name],
            domains=sorted({center.domain, analog.domain}),
            confidence=min(0.85, 0.5 + best_score),
        )

    def _rollout(self, name: str, *, steps: int = 4) -> ReasoningStep | None:
        path = self.universe.walk(name, steps=steps, rng=self._rng)
        if len(path) < 2:
            return None
        names = [c.name for c in path]
        domains = sorted({c.domain for c in path})
        summary = (
            f"rollout from {name}: {' → '.join(names)} "
            f"(touched {len(domains)} domain(s))."
        )
        return ReasoningStep(
            kind="compose",
            summary=summary,
            concepts=names,
            domains=domains,
            confidence=0.55,
        )

    def _reflect(self, name: str) -> ReasoningStep | None:
        concept = self.universe.get(name)
        if concept is None:
            return None
        n_neighbors = len(self.universe.neighbors(name))
        if concept.definition:
            summary = (
                f"reflecting on {name}: {concept.definition} "
                f"It links to {n_neighbors} other concept(s); "
                f"in {concept.domain}, it appears {concept.visits} time(s) so far."
            )
        else:
            summary = (
                f"reflecting on {name}: {n_neighbors} connection(s) in "
                f"{concept.domain}; visited {concept.visits} time(s)."
            )
        return ReasoningStep(
            kind="reflect",
            summary=summary,
            concepts=[name],
            domains=[concept.domain],
            confidence=0.7,
        )

    # -- helpers -----------------------------------------------------------

    def _highest_salience(self, names: Iterable[str]) -> str:
        best_name = ""
        best_score = -1.0
        for n in names:
            concept = self.universe.get(n)
            if concept is None:
                continue
            score = concept.salience + 0.1 * concept.visits
            if score > best_score:
                best_score = score
                best_name = concept.name
        return best_name or next(iter(names), "")

    def _build_answer_points(self, trace: ReasoningTrace) -> list[str]:
        points: list[str] = []
        for step in trace.steps:
            if step.kind in {"expand", "bridge", "reflect", "analogy"}:
                points.append(step.summary)
        # Truncate to the planner's typical limit so the realizer has room.
        return points[:6]

    def _coverage(self, trace: ReasoningTrace) -> float:
        touched = {c for step in trace.steps for c in step.concepts}
        total = len(self.universe)
        if total == 0:
            return 0.0
        return min(1.0, len(touched) / math.sqrt(max(1, total)))

    def _publish(self, trace: ReasoningTrace) -> None:
        if self.bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            self.bus.publish(
                BusTopic.SIMULATIONS,
                {
                    "kind": "conceptual_reasoning",
                    "query": trace.query,
                    "seeds": list(trace.seed_concepts),
                    "steps": len(trace.steps),
                    "coverage": round(trace.coverage, 3),
                },
                source="conceptual_reasoner",
            )
        except Exception:
            pass
