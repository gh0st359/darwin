"""Curiosity: Darwin notices the gaps in its own universe.

A frontier-grade thinker is not just a question-answerer. It is a
question-*maker*. The curiosity engine scans Darwin's universe for the
shapes of ignorance — concepts with too few neighbors, domains with
suspiciously few cross-domain bridges, regularities that could
plausibly extend further than they currently do — and surfaces them as
``CuriosityProbe``\\s the chat path can offer back to the operator
("I'm uncertain about X. Can you tell me how X relates to Y?") or feed
into the meta-proposer.

Curiosity is bounded and ranked. The engine returns the top N probes per
call, sorted by a heuristic score that favors high-leverage gaps:
isolated concepts adjacent to dense neighborhoods, domains lacking
cross-domain bridges, and concept clusters that would close if a single
missing relation were added.

This is structural curiosity — no domain content is assumed. The
operators here work over whatever is in the universe at the moment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from darwin.universe.concept_universe import ConceptUniverse


@dataclass
class CuriosityProbe:
    """One question Darwin is curious about, with structural justification."""

    kind: str             # "isolated_concept" / "missing_bridge" / "weak_definition" / "cluster_gap"
    question: str         # natural-language formulation the chat path can speak
    concepts: list[str] = field(default_factory=list)
    score: float = 0.5
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "question": self.question,
            "concepts": list(self.concepts),
            "score": round(self.score, 3),
            "evidence": dict(self.evidence),
        }


class CuriosityEngine:
    """Find the next thing Darwin would benefit from learning."""

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        max_probes: int = 8,
        isolated_neighbor_threshold: int = 1,
        weak_definition_threshold: int = 16,  # chars
    ) -> None:
        self.universe = universe
        self.max_probes = max_probes
        self.isolated_neighbor_threshold = isolated_neighbor_threshold
        self.weak_definition_threshold = weak_definition_threshold

    def probe(self) -> list[CuriosityProbe]:
        candidates: list[CuriosityProbe] = []
        candidates.extend(self._isolated_concepts())
        candidates.extend(self._weak_definitions())
        candidates.extend(self._missing_cross_domain_bridges())
        candidates.extend(self._cluster_gaps())
        candidates.sort(key=lambda p: p.score, reverse=True)
        return candidates[: self.max_probes]

    # -- probes ----------------------------------------------------------

    def _isolated_concepts(self) -> list[CuriosityProbe]:
        out: list[CuriosityProbe] = []
        for concept in self.universe.all_concepts():
            n_out = len(self.universe.neighbors(concept.name))
            n_in = len(self.universe.neighbors(concept.name, include_incoming=True)) - n_out
            total = n_out + max(0, n_in)
            if total <= self.isolated_neighbor_threshold:
                # The fewer edges, the more interesting (per-concept score
                # peaks at total == 0).
                score = 0.8 if total == 0 else 0.55
                out.append(
                    CuriosityProbe(
                        kind="isolated_concept",
                        question=(
                            f"What does {concept.name!r} relate to? "
                            f"It's barely connected in my universe."
                        ),
                        concepts=[concept.name],
                        score=score,
                        evidence={"edge_count": total, "domain": concept.domain},
                    )
                )
        return out

    def _weak_definitions(self) -> list[CuriosityProbe]:
        out: list[CuriosityProbe] = []
        for concept in self.universe.all_concepts():
            if len(concept.definition) < self.weak_definition_threshold:
                out.append(
                    CuriosityProbe(
                        kind="weak_definition",
                        question=(
                            f"How would you define {concept.name!r}? "
                            f"My current definition is thin."
                        ),
                        concepts=[concept.name],
                        score=0.5,
                        evidence={
                            "definition_length": len(concept.definition),
                            "domain": concept.domain,
                        },
                    )
                )
        return out

    def _missing_cross_domain_bridges(self) -> list[CuriosityProbe]:
        out: list[CuriosityProbe] = []
        domains = self.universe.domains()
        if len(domains) < 2:
            return out
        # Count cross-domain edges per domain pair.
        pair_counts: Counter = Counter()
        for rel in self.universe.relations():
            sa = self.universe.get(rel.source)
            sb = self.universe.get(rel.target)
            if sa is None or sb is None:
                continue
            if sa.domain == sb.domain:
                continue
            key = tuple(sorted([sa.domain, sb.domain]))
            pair_counts[key] += 1
        # For domains with no cross-domain edges, surface a probe.
        domain_names = {d.name for d in domains}
        for a in domain_names:
            for b in domain_names:
                if a >= b:
                    continue
                key = (a, b)
                if pair_counts.get(key, 0) == 0:
                    out.append(
                        CuriosityProbe(
                            kind="missing_bridge",
                            question=(
                                f"How does {a!r} relate to {b!r}? "
                                f"I see no connection between these two domains."
                            ),
                            concepts=[],
                            score=0.6,
                            evidence={"domains": [a, b]},
                        )
                    )
        return out

    def _cluster_gaps(self) -> list[CuriosityProbe]:
        out: list[CuriosityProbe] = []
        # A "cluster gap" is a pair of concepts that share a parent (both
        # is_a X) but have no direct edge between them. Often these are
        # near-siblings whose relationship is worth asking about.
        children_of: dict[str, list[str]] = {}
        for rel in self.universe.relations():
            if rel.kind == "is_a":
                children_of.setdefault(rel.target, []).append(rel.source)
        for parent, kids in children_of.items():
            if len(kids) < 2:
                continue
            for i, a in enumerate(kids):
                for b in kids[i + 1: i + 5]:  # bounded
                    if any(
                        rel.target == b
                        for rel in self.universe.neighbors(a)
                    ):
                        continue
                    out.append(
                        CuriosityProbe(
                            kind="cluster_gap",
                            question=(
                                f"How does {a!r} differ from or relate to {b!r}? "
                                f"They're both {parent!r}."
                            ),
                            concepts=[a, b, parent],
                            score=0.55,
                            evidence={"shared_parent": parent},
                        )
                    )
        return out

    def summary(self) -> dict[str, Any]:
        probes = self.probe()
        return {
            "probes": len(probes),
            "kinds": dict(Counter(p.kind for p in probes)),
            "top_questions": [p.question for p in probes[:3]],
        }
