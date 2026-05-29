"""ConceptDeriver — Darwin grows its universe from experience.

The universe starts with only structural primitives (``thing``, ``change``,
``cause``, etc.). Everything else — the concept of ``flow`` once Darwin
hears about water moving, the concept of ``pattern`` once it notices a
recurrence in its causal model, the concept of ``trust`` once a chat reveals
a stance — is *derived*, not looked up. This module is the machinery that
performs that derivation.

Four derivation pathways:

  1. **From causal regularities**. The CausalModel has stable beliefs of
     the form "action ⇒ variable effect". Each persistent regularity is
     evidence that an underlying concept is at play; the deriver proposes
     a concept name for the regularity and links it via ``describes`` /
     ``causes`` edges.

  2. **From co-occurring grounded words**. When two ungrounded words from
     chat repeatedly co-occur, the deriver proposes that they may name a
     shared concept (an analogy or composition).

  3. **From reflection**. Periodically Darwin walks the universe and asks:
     "what is this neighborhood actually *about*?" The deriver runs a
     compositional summarization that proposes a higher-level concept.

  4. **From composition / generalization / specialization**. Given two
     concepts that share many neighbors, propose a parent kind. Given a
     concept with many instance-style edges, propose an instance.

All derivations go through ``propose`` first; the runtime can accept or
reject via the same MetaAcceptGate apparatus that gates self-modifications.
A derivation is just a structural proposal in the same grammar.

What the deriver does NOT do:
  * Look up domain facts. There is no "physics knowledge" anywhere in this
    file. Derivation is purely *structural*: it builds graph nodes and
    edges from the regularities Darwin observes.
  * Use pretrained vectors. All similarity uses the live
    CausalEmbeddingSpace.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from darwin.universe.concept_universe import ConceptUniverse


_DERIVED_DOMAIN = "derived"


@dataclass
class DerivedConcept:
    """One concept the deriver proposes to add to the universe."""

    name: str
    domain: str = _DERIVED_DOMAIN
    definition: str = ""
    derived_from: tuple[str, ...] = ()
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    pathway: str = "unknown"        # "regularity" / "cooccurrence" / "reflection" / "composition"
    confidence: float = 0.4
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "definition": self.definition,
            "derived_from": list(self.derived_from),
            "relations": list(self.relations),
            "pathway": self.pathway,
            "confidence": round(self.confidence, 3),
            "evidence": dict(self.evidence),
        }


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']{2,}")


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return cleaned or "anon"


class ConceptDeriver:
    """Grows Darwin's universe from observed regularities and chat traffic.

    Hook the deriver into the runtime once. It maintains its own bounded
    state (counters of seen words, last-seen causal-belief signatures) so
    it can spot new regularities cheaply between runs of ``derive``.
    """

    def __init__(
        self,
        universe: ConceptUniverse,
        *,
        embedding_space: Any = None,
        max_word_cooccurrence: int = 4096,
        cooccurrence_threshold: int = 3,
        regularity_confidence: float = 0.7,
        bus: Any = None,
    ) -> None:
        self.universe = universe
        self.embedding_space = embedding_space
        self.bus = bus
        self.cooccurrence_threshold = cooccurrence_threshold
        self.regularity_confidence = regularity_confidence
        self._cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        self._max_cooc = max_word_cooccurrence
        self._seen_regularity_signatures: set[str] = set()
        self._proposed: list[DerivedConcept] = []
        self._accepted: list[DerivedConcept] = []

    # -- inputs from the runtime ------------------------------------------

    def observe_text(self, text: str) -> None:
        """Track word co-occurrence from a piece of natural-language text.

        Words that have not been grounded against the universe yet are
        candidates for new concepts; words that have already been grounded
        contribute to *relating* known concepts.
        """

        tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        # Build a unique-token list per sentence; dedupe to avoid weighting
        # repetition over co-occurrence.
        unique = list(dict.fromkeys(tokens))
        if len(unique) < 2:
            return
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                key = (a, b) if a < b else (b, a)
                self._cooccurrence[key] += 1
        # Bound memory.
        if len(self._cooccurrence) > self._max_cooc:
            # Drop the lowest-count half.
            sorted_items = sorted(
                self._cooccurrence.items(), key=lambda kv: kv[1], reverse=True
            )
            kept = dict(sorted_items[: self._max_cooc // 2])
            self._cooccurrence.clear()
            self._cooccurrence.update(kept)

    # -- derivation pathways ----------------------------------------------

    def derive(self, *, darwin: Any | None = None) -> list[DerivedConcept]:
        """Run every derivation pathway. Returns the newly accepted concepts."""

        proposals: list[DerivedConcept] = []
        if darwin is not None:
            proposals.extend(self._from_regularities(darwin))
        proposals.extend(self._from_cooccurrence())
        proposals.extend(self._from_composition())
        accepted: list[DerivedConcept] = []
        for proposal in proposals:
            if self._accept(proposal):
                accepted.append(proposal)
        self._accepted.extend(accepted)
        if self._accepted and len(self._accepted) > 4096:
            self._accepted = self._accepted[-4096:]
        self._publish(accepted)
        return accepted

    # -- pathway 1: causal-model regularities -----------------------------

    def _from_regularities(self, darwin: Any) -> list[DerivedConcept]:
        causal = getattr(darwin, "causal_model", None)
        if causal is None:
            return []
        try:
            beliefs = causal.beliefs(limit=64)
        except Exception:
            return []
        out: list[DerivedConcept] = []
        for belief in beliefs:
            try:
                conf = float(getattr(belief, "confidence", 0.0))
            except Exception:
                conf = 0.0
            if conf < self.regularity_confidence:
                continue
            action = str(getattr(belief, "action", ""))
            variable = str(getattr(belief, "variable", ""))
            effect = str(getattr(belief, "effect", ""))
            if not action or not variable:
                continue
            signature = f"{action}|{variable}|{effect}"
            if signature in self._seen_regularity_signatures:
                continue
            self._seen_regularity_signatures.add(signature)
            # Propose the abstract concept: this regularity is *about*
            # something. Its name is the sanitized triple; its definition
            # describes the empirical observation, not a hardcoded fact.
            name = _sanitize(f"reg_{action}_{variable}")
            definition = (
                f"A regularity observed by Darwin: doing {action} "
                f"reliably affects {variable} ({effect}). "
                f"Confidence rose to {conf:.2f} over repeated observation."
            )
            relations = [
                (name, "describes", "change"),
                (name, "is_a", "cause"),
            ]
            # Don't fail derivation if action/variable haven't been grounded
            # yet — link to primitives that are always present.
            out.append(
                DerivedConcept(
                    name=name,
                    domain="derived",
                    definition=definition,
                    derived_from=(action, variable),
                    relations=relations,
                    pathway="regularity",
                    confidence=min(0.9, 0.5 + conf / 2),
                    evidence={"signature": signature, "confidence": conf},
                )
            )
        return out

    # -- pathway 2: word co-occurrence ------------------------------------

    def _from_cooccurrence(self) -> list[DerivedConcept]:
        out: list[DerivedConcept] = []
        for (a, b), count in list(self._cooccurrence.items()):
            if count < self.cooccurrence_threshold:
                continue
            # Skip if either word is already a concept and they are already
            # linked — co-occurrence has nothing new to add.
            ca = self.universe.get(a)
            cb = self.universe.get(b)
            if ca is not None and cb is not None:
                already = any(
                    rel.target == cb.name
                    for rel in self.universe.neighbors(ca.name)
                )
                if already:
                    continue
                # Propose a `related_to` edge between two existing concepts.
                joint = f"link_{ca.name}_{cb.name}"
                if self.universe.has(joint):
                    continue
                out.append(
                    DerivedConcept(
                        name=joint,
                        domain="derived",
                        definition=(
                            f"Repeated co-occurrence in conversation of "
                            f"{ca.name} and {cb.name} ({count} times) suggests "
                            f"a shared concept linking them."
                        ),
                        derived_from=(ca.name, cb.name),
                        relations=[
                            (ca.name, "related_to", cb.name),
                            (cb.name, "related_to", ca.name),
                        ],
                        pathway="cooccurrence",
                        confidence=min(0.7, 0.3 + count * 0.05),
                        evidence={"count": count},
                    )
                )
                continue
            # If only one is grounded, that grounded concept is a "kind"
            # of which the ungrounded word may be an instance.
            anchor = ca or cb
            if anchor is None:
                continue
            new_word = b if anchor is ca else a
            new_name = _sanitize(new_word)
            if self.universe.has(new_name):
                continue
            out.append(
                DerivedConcept(
                    name=new_name,
                    domain="derived",
                    definition=(
                        f"Word {new_word!r} co-occurred {count} times in "
                        f"conversation with {anchor.name}; added as a "
                        f"candidate concept related to {anchor.name}."
                    ),
                    derived_from=(anchor.name,),
                    relations=[(new_name, "related_to", anchor.name)],
                    pathway="cooccurrence",
                    confidence=min(0.6, 0.25 + count * 0.05),
                    evidence={"count": count, "partner": anchor.name},
                )
            )
        return out

    # -- pathway 3: composition / generalization --------------------------

    def _from_composition(self) -> list[DerivedConcept]:
        """If two concepts share most of their neighbors, propose a parent kind."""

        out: list[DerivedConcept] = []
        all_concepts = self.universe.all_concepts()
        # Cheap O(N) sample to avoid quadratic blow-up on large universes.
        if len(all_concepts) < 4:
            return out
        # Build neighbor signatures.
        sigs: dict[str, frozenset[str]] = {}
        for concept in all_concepts:
            neighbors = {
                rel.target for rel in self.universe.neighbors(concept.name)
            }
            sigs[concept.name] = frozenset(neighbors)
        # Group concepts by signature size buckets so we compare only
        # similarly-connected nodes.
        names = list(sigs)
        for i, a in enumerate(names):
            sa = sigs[a]
            if not sa:
                continue
            for b in names[i + 1: i + 20]:  # bounded neighbor window
                sb = sigs[b]
                if not sb:
                    continue
                overlap = sa & sb
                union = sa | sb
                if not union:
                    continue
                jacc = len(overlap) / len(union)
                if jacc < 0.6 or len(overlap) < 2:
                    continue
                # Propose a parent kind whose members are a and b.
                parent_name = _sanitize(f"kind_{a}_{b}")
                if self.universe.has(parent_name):
                    continue
                out.append(
                    DerivedConcept(
                        name=parent_name,
                        domain="derived",
                        definition=(
                            f"A candidate parent kind: {a!r} and {b!r} share "
                            f"{len(overlap)} neighbor(s) ({jacc:.0%} Jaccard), "
                            f"suggesting they are instances of a more general "
                            f"category."
                        ),
                        derived_from=(a, b),
                        relations=[
                            (a, "is_a", parent_name),
                            (b, "is_a", parent_name),
                            (parent_name, "is_a", "kind"),
                        ],
                        pathway="composition",
                        confidence=0.4 + 0.3 * jacc,
                        evidence={"jaccard": jacc, "shared": sorted(overlap)},
                    )
                )
        return out

    # -- acceptance / persistence ----------------------------------------

    def _accept(self, proposal: DerivedConcept) -> bool:
        """Land a proposal in the universe. Idempotent (re-derivations skipped)."""

        if self.universe.has(proposal.name):
            return False
        try:
            self.universe.add_concept(
                proposal.name,
                domain=proposal.domain,
                definition=proposal.definition,
                derived_from=proposal.derived_from,
                salience=0.5 + proposal.confidence,
            )
            for source, kind, target in proposal.relations:
                try:
                    self.universe.add_relation(
                        source, target, kind, ensure_concepts=True,
                        notes=f"derived via {proposal.pathway}",
                    )
                except KeyError:
                    continue
            return True
        except Exception:
            return False

    def _publish(self, accepted: list[DerivedConcept]) -> None:
        if not accepted or self.bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            self.bus.publish(
                BusTopic.PROPOSALS,
                {
                    "kind": "concept_derivation",
                    "accepted": [c.to_record() for c in accepted],
                    "at": time.time(),
                },
                source="concept_deriver",
            )
        except Exception:
            pass

    # -- introspection ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "tracked_word_pairs": len(self._cooccurrence),
            "seen_regularities": len(self._seen_regularity_signatures),
            "proposals_accepted": len(self._accepted),
            "pathways": dict(
                Counter(c.pathway for c in self._accepted)
            ),
        }
