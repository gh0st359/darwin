"""HypothesisEngine — Darwin proposes novel edges from the patterns it sees.

A reasoner that only answers what's asked is reactive. A *frontier* reasoner
proposes its own hypotheses and offers them up for verification. This
module is that proactive layer.

Three pathways:

  1. **Transitive closure** — If A is_a B and B is_a C, hypothesize
     A is_a C. The inference engine can *derive* this on demand, but
     here we surface it as a *candidate edge* with a confidence score.
     If accepted, it becomes a real edge; if refuted by the operator,
     it's recorded as a negative example.

  2. **Analogical inference** — If A and B share a substantial fraction
     of their neighborhood (Jaccard ≥ threshold), and A has a relation
     ``A ─R→ X`` that B does not have, hypothesize ``B ─R→ X``. This is
     genuine *analogical generalization*: things that look alike along
     many dimensions probably look alike along others.

  3. **Cross-domain bridging** — When a concept in domain D1 shares
     neighborhood structure with a concept in D2, hypothesize a
     ``analogous_to`` edge spanning the domains. This is the engine
     for "music is like math" / "ecology is like economics" kinds of
     insight.

Every hypothesis comes with a justification (the supporting evidence
that made the engine propose it). The proactive dialogue path can pull
high-confidence hypotheses to surface to the operator without being
asked, turning Darwin from a question-answerer into a question-asker.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from darwin.universe.concept_universe import ConceptUniverse


@dataclass
class Hypothesis:
    """A proposed new edge plus the evidence that made the engine propose it."""

    source: str
    target: str
    kind: str
    pathway: str               # "transitive" / "analogical" / "cross_domain"
    rationale: str             # human-readable justification
    confidence: float = 0.4
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "pathway": self.pathway,
            "rationale": self.rationale,
            "confidence": round(self.confidence, 3),
            "evidence": dict(self.evidence),
            "created_at": self.created_at,
        }

    def as_question(self) -> str:
        if self.kind == "is_a":
            return f"Is {self.source} a kind of {self.target}?"
        if self.kind == "part_of":
            return f"Is {self.source} part of {self.target}?"
        if self.kind == "causes":
            return f"Does {self.source} cause {self.target}?"
        if self.kind == "analogous_to":
            return f"Is {self.source} analogous to {self.target}?"
        return f"Are {self.source} and {self.target} related by {self.kind}?"


class HypothesisEngine:
    """Generates proactive hypotheses from the current universe state."""

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        analogical_jaccard_threshold: float = 0.55,
        analogical_min_overlap: int = 2,
        max_hypotheses_per_pass: int = 16,
        min_confidence_to_surface: float = 0.4,
    ) -> None:
        self.universe = universe
        self.analogical_jaccard_threshold = analogical_jaccard_threshold
        self.analogical_min_overlap = analogical_min_overlap
        self.max_hypotheses_per_pass = max_hypotheses_per_pass
        self.min_confidence_to_surface = min_confidence_to_surface
        self._produced: list[Hypothesis] = []
        # Refuted hypotheses cache (the operator can mark a hypothesis as
        # wrong and the engine remembers not to propose it again).
        self._refuted: set[tuple[str, str, str]] = set()

    # -- generation ------------------------------------------------------

    def generate(self) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        out.extend(self._transitive_hypotheses())
        out.extend(self._analogical_hypotheses())
        out.extend(self._cross_domain_hypotheses())
        # Sort by confidence and clip.
        out.sort(key=lambda h: h.confidence, reverse=True)
        out = out[: self.max_hypotheses_per_pass]
        # Filter refuted.
        out = [
            h for h in out
            if (h.source, h.kind, h.target) not in self._refuted
        ]
        self._produced.extend(out)
        if len(self._produced) > 4096:
            self._produced = self._produced[-4096:]
        return out

    # -- pathway 1: transitive closure ----------------------------------

    def _transitive_hypotheses(self) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        # For every concept, walk two is_a hops; propose the closure edge
        # if it doesn't already exist.
        for concept in self.universe.all_concepts():
            for rel1 in self.universe.neighbors(concept.name, kinds=["is_a"]):
                for rel2 in self.universe.neighbors(rel1.target, kinds=["is_a"]):
                    target = rel2.target
                    if target == concept.name:
                        continue
                    # Is the direct edge already present?
                    already = any(
                        r.target == target and r.kind == "is_a"
                        for r in self.universe.neighbors(concept.name)
                    )
                    if already:
                        continue
                    rationale = (
                        f"{concept.name} is_a {rel1.target}, and "
                        f"{rel1.target} is_a {target}; transitivity of is_a "
                        f"suggests the direct edge."
                    )
                    out.append(
                        Hypothesis(
                            source=concept.name,
                            target=target,
                            kind="is_a",
                            pathway="transitive",
                            rationale=rationale,
                            confidence=0.8,
                            evidence={"intermediate": rel1.target},
                        )
                    )
                    if len(out) >= self.max_hypotheses_per_pass:
                        return out
        return out

    # -- pathway 2: analogical inference --------------------------------

    def _analogical_hypotheses(self) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        # Build neighborhood signatures.
        sigs: dict[str, frozenset[str]] = {}
        rel_map: dict[str, dict[str, str]] = {}   # name -> {target: kind}
        for concept in self.universe.all_concepts():
            outgoing = self.universe.neighbors(concept.name)
            sigs[concept.name] = frozenset(rel.target for rel in outgoing)
            rel_map[concept.name] = {rel.target: rel.kind for rel in outgoing}
        names = list(sigs)
        for i, a in enumerate(names):
            sa = sigs[a]
            if len(sa) < self.analogical_min_overlap:
                continue
            for b in names[i + 1: i + 30]:   # bounded scan window
                sb = sigs[b]
                if len(sb) < self.analogical_min_overlap:
                    continue
                overlap = sa & sb
                union = sa | sb
                if len(overlap) < self.analogical_min_overlap:
                    continue
                jacc = len(overlap) / len(union)
                if jacc < self.analogical_jaccard_threshold:
                    continue
                # b's neighbors that a does NOT have, and vice versa, are
                # the analogical hypotheses.
                a_missing = sigs[b] - sigs[a]
                b_missing = sigs[a] - sigs[b]
                for target in list(a_missing)[:3]:
                    kind = rel_map[b].get(target, "related_to")
                    out.append(
                        Hypothesis(
                            source=a,
                            target=target,
                            kind=kind,
                            pathway="analogical",
                            rationale=(
                                f"{a} and {b} share {len(overlap)} neighbor(s) "
                                f"({jacc:.0%} Jaccard); {b} {_human_kind(kind)} "
                                f"{target}, so by analogy {a} may too."
                            ),
                            confidence=0.4 + 0.4 * jacc,
                            evidence={"analog": b, "jaccard": jacc},
                        )
                    )
                    if len(out) >= self.max_hypotheses_per_pass:
                        return out
                for target in list(b_missing)[:3]:
                    kind = rel_map[a].get(target, "related_to")
                    out.append(
                        Hypothesis(
                            source=b,
                            target=target,
                            kind=kind,
                            pathway="analogical",
                            rationale=(
                                f"{b} and {a} share {len(overlap)} neighbor(s) "
                                f"({jacc:.0%} Jaccard); {a} {_human_kind(kind)} "
                                f"{target}, so by analogy {b} may too."
                            ),
                            confidence=0.4 + 0.4 * jacc,
                            evidence={"analog": a, "jaccard": jacc},
                        )
                    )
                    if len(out) >= self.max_hypotheses_per_pass:
                        return out
        return out

    # -- pathway 3: cross-domain bridging ------------------------------

    def _cross_domain_hypotheses(self) -> list[Hypothesis]:
        """Two concepts in different domains with similar structure should
        probably be tagged analogous_to. This is the bridge that lets
        physics ↔ math, ecology ↔ economics, music ↔ math etc. emerge
        without anyone hardcoding them.

        Conservative criteria: require relation-kind overlap >= 60% AND
        at least one *shared* neighbor concept. The shared-neighbor
        criterion cuts the noisy "everyone with one is_a edge is
        analogous to everyone else with one is_a edge" false positives.
        """

        out: list[Hypothesis] = []
        per_domain: dict[str, list[str]] = defaultdict(list)
        for concept in self.universe.all_concepts():
            per_domain[concept.domain].append(concept.name)
        domains = list(per_domain.keys())
        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                # Sample a handful from each domain.
                for a in per_domain[d1][:6]:
                    a_kinds = {rel.kind for rel in self.universe.neighbors(a)}
                    a_targets = {rel.target for rel in self.universe.neighbors(a)}
                    if not a_kinds:
                        continue
                    for b in per_domain[d2][:6]:
                        if a == b:
                            continue
                        # Skip if already linked.
                        already = any(
                            rel.target == b and rel.kind == "analogous_to"
                            for rel in self.universe.neighbors(a)
                        )
                        if already:
                            continue
                        b_kinds = {rel.kind for rel in self.universe.neighbors(b)}
                        b_targets = {rel.target for rel in self.universe.neighbors(b)}
                        if not b_kinds:
                            continue
                        kind_overlap = len(a_kinds & b_kinds) / max(
                            1, len(a_kinds | b_kinds)
                        )
                        shared_targets = a_targets & b_targets
                        if kind_overlap < 0.6 or not shared_targets:
                            continue
                        out.append(
                            Hypothesis(
                                source=a,
                                target=b,
                                kind="analogous_to",
                                pathway="cross_domain",
                                rationale=(
                                    f"{a} in {d1} and {b} in {d2} share "
                                    f"{int(kind_overlap * 100)}% of their "
                                    f"relation kinds and the shared neighbor(s) "
                                    f"{sorted(shared_targets)[:3]}; "
                                    f"cross-domain analogy seems plausible."
                                ),
                                confidence=0.3 + 0.4 * kind_overlap,
                                evidence={
                                    "d1": d1, "d2": d2,
                                    "kind_overlap": kind_overlap,
                                    "shared_targets": sorted(shared_targets)[:5],
                                },
                            )
                        )
                        if len(out) >= self.max_hypotheses_per_pass:
                            return out
        return out

    # -- feedback --------------------------------------------------------

    def refute(self, source: str, kind: str, target: str) -> None:
        """The operator has told us a hypothesis was wrong. Don't propose
        it again."""

        self._refuted.add((source, kind, target))

    def accept(self, hypothesis: Hypothesis) -> None:
        """The operator (or downstream pipeline) accepted the hypothesis;
        actually add it to the universe."""

        try:
            self.universe.add_relation(
                hypothesis.source, hypothesis.target, hypothesis.kind,
                weight=hypothesis.confidence,
                notes=f"accepted hypothesis via {hypothesis.pathway}",
            )
        except KeyError:
            pass

    def surface(self) -> list[Hypothesis]:
        """Return only those produced hypotheses worth surfacing now."""

        return [
            h for h in self._produced[-32:]
            if h.confidence >= self.min_confidence_to_surface
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "total_produced": len(self._produced),
            "refuted": len(self._refuted),
            "by_pathway": dict(Counter(h.pathway for h in self._produced[-256:])),
            "recent": [h.as_question() for h in self._produced[-5:]],
        }


def _human_kind(kind: str) -> str:
    return {
        "is_a": "is a",
        "part_of": "is part of",
        "causes": "causes",
        "describes": "describes",
        "analogous_to": "is analogous to",
        "instantiates": "is an instance of",
        "requires": "requires",
        "opposes": "is opposed to",
        "related_to": "relates to",
    }.get(kind, kind.replace("_", " "))
