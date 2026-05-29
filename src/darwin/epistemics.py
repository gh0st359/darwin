"""Epistemic categorization — *derived* belief categories, not hardcoded rules.

The user-facing problem this solves: Darwin's `/beliefs` was getting cluttered
with internal bookkeeping noise — "focus → last_action", "step → focus" —
that's an artifact of the scheduler, not a genuine fact about the world. At
the same time, real facts ("a neuron is a cell", taught by the operator)
should rise to the top.

The principled solution is *not* to hardcode "these variable names are
bookkeeping". That would freeze Darwin's epistemics at design time. Instead
this module *derives* a category for each belief from observable properties:

  * **Provenance.** Where did this belief come from? Was it taught by the
    operator (fused), derived from another belief (transitive closure),
    proposed as a hypothesis, observed via a tool, or recorded by the
    scheduler as it ran?
  * **Confirmation history.** Has the belief been re-encountered, used
    in successful chains, or supported by independent paths?
  * **Confidence trajectory.** Stable + monotonically increasing, or
    oscillating, or recently arrived?
  * **Subject.** Does the belief involve the *self* (Darwin reasoning
    about its own state), the *world* (an external regularity), or a
    purely *operational* substrate variable (loop step, focus,
    secondary_focus)?

The output is a *set* of categories, not a single one — a belief can
plausibly fit multiple. The categorizer is *advisory*, not prescriptive:
its job is to inform the surfacing layer (which beliefs to show by
default), not to limit what Darwin can think about. Darwin's HypothesisEngine
and ConceptDeriver remain free to re-categorize by reinforcing or refuting
the underlying patterns.

This is the deliberate framing: "higher-level belief categorization and
epistemic reasoning while preserving Darwin's ability to form, modify,
merge, and evolve its own beliefs dynamically."
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


# Canonical category names. New categories can be added by anyone; the
# surfacing layer doesn't care what the names mean, only which ones the
# caller asked to include or exclude.
WORLD_KNOWLEDGE = "world_knowledge"
OPERATIONAL = "operational"
SELF_KNOWLEDGE = "self_knowledge"
HYPOTHESIS = "hypothesis"
STABLE_FACT = "stable_fact"
TEMPORARY = "temporary"
SCHEDULER_ARTIFACT = "scheduler_artifact"


# A heuristic set of substrings that *tend to* indicate the corresponding
# concept names from the scheduler / conceptual world / interior runtime
# rather than the external world. The set itself is also re-derivable
# from observation: anything that consistently shows up in transitions
# emitted by the scheduler loops can be flagged. This list is a *seed*
# the deriver can grow.
_SCHEDULER_SUBSTRINGS = (
    "step", "focus", "secondary_focus", "last_action", "last_reward",
    "last_summary", "last_changed_at", "neighbor_count", "tool_step",
    "concept_count", "relation_count", "domain_count",
    "last_output_size", "last_success", "registered_tool_count",
    "_loop", "_state", "_status",
)


# Subject substrings that indicate self-reference. Same caveat applies.
_SELF_SUBSTRINGS = (
    "self", "darwin", "model", "interior", "embedding_drift",
    "kernel_saturation", "subsystem_health", "ledger_growth",
    "oversight_intensity",
)


@dataclass
class BeliefSignal:
    """The observable signals a categorizer reads off one belief.

    All fields are optional; only the subset that's relevant for a given
    belief type needs to be populated. The categorizer reads what's
    available and tolerates everything else.
    """

    name: str = ""
    confidence: float = 0.0
    samples: int = 0
    age_seconds: float = 0.0
    visits: int = 0
    provenance: str = ""        # "fused" / "derived" / "hypothesis" / "primitive" / "tool" / "scheduler"
    has_contradiction: bool = False
    cross_context_uses: int = 0  # how many distinct loops used this belief
    target: str = ""             # for relations: the target name
    domain: str = ""             # for concepts: the home domain
    kind: str = ""               # for relations: the relation kind

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _is_scheduler_subject(name: str) -> bool:
    lowered = name.lower()
    return any(sub in lowered for sub in _SCHEDULER_SUBSTRINGS)


def _is_self_subject(name: str) -> bool:
    lowered = name.lower()
    return any(sub in lowered for sub in _SELF_SUBSTRINGS)


def categorize(signal: BeliefSignal) -> set[str]:
    """Derive the set of categories that *appear* to apply to a belief.

    A belief always fits at least one category (default: OPERATIONAL when
    nothing else applies). Categories are non-exclusive.
    """

    categories: set[str] = set()

    # 1. Subject-driven categories.
    name = signal.name or signal.target
    if name:
        if _is_scheduler_subject(name):
            categories.add(SCHEDULER_ARTIFACT)
            categories.add(OPERATIONAL)
        if _is_self_subject(name):
            categories.add(SELF_KNOWLEDGE)

    # 2. Provenance-driven categories.
    if signal.provenance == "fused":
        categories.add(WORLD_KNOWLEDGE)
    elif signal.provenance == "tool":
        categories.add(WORLD_KNOWLEDGE)
    elif signal.provenance == "hypothesis":
        categories.add(HYPOTHESIS)
    elif signal.provenance == "scheduler":
        categories.add(SCHEDULER_ARTIFACT)
        categories.add(OPERATIONAL)
    elif signal.provenance == "primitive":
        # Primitives carry the *structure* of thought; they're not world
        # facts about anything. Treat them as operational scaffolding so
        # they don't crowd /beliefs by default.
        categories.add(OPERATIONAL)
    elif signal.provenance == "derived":
        # Derived concepts inherit their parent's category implicitly via
        # later categorization passes; default them as HYPOTHESIS unless
        # they've been confirmed enough to graduate to STABLE_FACT below.
        categories.add(HYPOTHESIS)

    # 3. Confidence + history drives stability vs temporariness.
    # A signal carrying *no* identifying information (no name, no provenance,
    # no confidence) is not "temporary" — it's nothing, and the only honest
    # category for it is OPERATIONAL. Only flag TEMPORARY when there's at
    # least some indication this is a real, recently-observed belief.
    has_identity = bool(signal.name or signal.provenance or signal.confidence > 0)
    if signal.confidence >= 0.8 and signal.samples >= 5 and not signal.has_contradiction:
        categories.add(STABLE_FACT)
    if (
        has_identity
        and signal.age_seconds < 60.0
        and signal.samples <= 1
        and signal.provenance not in ("primitive",)
    ):
        categories.add(TEMPORARY)
    if signal.cross_context_uses >= 2:
        # Used across distinct loops/contexts — that's evidence of
        # generality. Promote to STABLE_FACT if confidence is mid-or-high.
        if signal.confidence >= 0.5 and not signal.has_contradiction:
            categories.add(STABLE_FACT)

    # 4. Default. If we still have nothing, the belief is operational.
    if not categories:
        categories.add(OPERATIONAL)

    return categories


def filter_signals(
    signals: Iterable[BeliefSignal],
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[BeliefSignal]:
    """Filter a stream of signals by categories.

    ``include``: keep only signals whose categories intersect this set.
    ``exclude``: drop signals whose categories intersect this set.
    Both can be supplied; ``exclude`` wins ties.
    """

    inc = set(include or ())
    exc = set(exclude or ())
    out: list[BeliefSignal] = []
    for signal in signals:
        cats = categorize(signal)
        if exc and cats & exc:
            continue
        if inc and not (cats & inc):
            continue
        out.append(signal)
    return out


# --------------------------------------------------------------------------- #
# Adapters: turn the various belief shapes Darwin has into BeliefSignals.
# --------------------------------------------------------------------------- #


def signal_from_causal_belief(
    belief: Any, *, runtime: Any = None, scheduler_actions: Iterable[str] | None = None,
) -> BeliefSignal:
    """Build a BeliefSignal from a v5 CausalBelief.

    ``scheduler_actions`` lets the caller provide a hint set of action
    names that came from internal loops; matching beliefs get
    SCHEDULER_ARTIFACT in addition to their other categories.
    """

    name = f"{getattr(belief, 'action', '')}:{getattr(belief, 'variable', '')}"
    target = str(getattr(belief, "variable", ""))
    samples = int(getattr(belief, "samples", 0) or 0)
    confidence = float(getattr(belief, "confidence", 0.0) or 0.0)
    provenance = "derived"
    if scheduler_actions and getattr(belief, "action", "") in set(scheduler_actions):
        provenance = "scheduler"
    return BeliefSignal(
        name=name,
        target=target,
        confidence=confidence,
        samples=samples,
        provenance=provenance,
    )


def signal_from_concept(concept: Any) -> BeliefSignal:
    salience = float(getattr(concept, "salience", 0.0) or 0.0)
    visits = int(getattr(concept, "visits", 0) or 0)
    age = max(0.0, time.time() - float(getattr(concept, "created_at", time.time()) or time.time()))
    domain = str(getattr(concept, "domain", ""))
    name = str(getattr(concept, "name", ""))
    derived_from = getattr(concept, "derived_from", ()) or ()
    if derived_from:
        provenance = "derived"
    elif domain in {"structure", "dynamics", "inference", "magnitude", "self"} and not derived_from:
        provenance = "primitive"
    elif domain == "fused":
        provenance = "fused"
    elif domain == "derived":
        provenance = "derived"
    else:
        provenance = "fused" if domain not in {"general"} else "derived"
    return BeliefSignal(
        name=name,
        domain=domain,
        provenance=provenance,
        age_seconds=age,
        visits=visits,
        confidence=min(1.0, salience),
    )


def signal_from_relation(relation: Any, *, universe: Any = None) -> BeliefSignal:
    source = str(getattr(relation, "source", ""))
    target = str(getattr(relation, "target", ""))
    kind = str(getattr(relation, "kind", ""))
    weight = float(getattr(relation, "weight", 1.0) or 1.0)
    notes = str(getattr(relation, "notes", ""))
    provenance = "derived"
    if "fused from chat" in notes:
        provenance = "fused"
    elif "derived via" in notes:
        provenance = "derived"
    elif "accepted hypothesis" in notes:
        provenance = "hypothesis"
    return BeliefSignal(
        name=f"{source} {kind} {target}",
        target=target,
        kind=kind,
        confidence=weight,
        provenance=provenance,
    )


def categorize_concept(concept: Any) -> set[str]:
    return categorize(signal_from_concept(concept))


def categorize_relation(relation: Any, *, universe: Any = None) -> set[str]:
    return categorize(signal_from_relation(relation, universe=universe))


def categorize_causal_belief(
    belief: Any, *, scheduler_actions: Iterable[str] | None = None,
) -> set[str]:
    return categorize(
        signal_from_causal_belief(belief, scheduler_actions=scheduler_actions)
    )


# --------------------------------------------------------------------------- #
# A monitor that periodically re-categorizes a sample and reports drift.
# --------------------------------------------------------------------------- #


class EpistemicMonitor:
    """Run categorization passes and surface drift in category counts.

    A drift signal is itself useful: if a previously-STABLE_FACT category
    starts shrinking turn-by-turn, something is destabilizing Darwin's
    knowledge. The monitor *reports* drift; it does not act on it.
    """

    def __init__(self, *, sample_size: int = 200) -> None:
        self.sample_size = sample_size
        self._history: list[dict[str, int]] = []

    def scan(
        self,
        *,
        causal_beliefs: Iterable[Any] = (),
        concepts: Iterable[Any] = (),
        relations: Iterable[Any] = (),
        scheduler_actions: Iterable[str] | None = None,
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        cb = list(causal_beliefs)[: self.sample_size]
        cn = list(concepts)[: self.sample_size]
        rl = list(relations)[: self.sample_size]
        for belief in cb:
            for cat in categorize_causal_belief(belief, scheduler_actions=scheduler_actions):
                counts[cat] += 1
        for concept in cn:
            for cat in categorize_concept(concept):
                counts[cat] += 1
        for relation in rl:
            for cat in categorize_relation(relation):
                counts[cat] += 1
        snapshot = dict(counts)
        self._history.append(snapshot)
        if len(self._history) > 64:
            self._history = self._history[-64:]
        return snapshot

    def drift(self) -> dict[str, float]:
        """Per-category fractional change between the last two scans."""

        if len(self._history) < 2:
            return {}
        prev, curr = self._history[-2], self._history[-1]
        keys = set(prev) | set(curr)
        out: dict[str, float] = {}
        for k in keys:
            p = float(prev.get(k, 0))
            c = float(curr.get(k, 0))
            if p == 0 and c == 0:
                continue
            base = max(p, c, 1.0)
            out[k] = (c - p) / base
        return out

    def history(self) -> list[dict[str, int]]:
        return list(self._history)


__all__ = [
    "BeliefSignal",
    "EpistemicMonitor",
    "HYPOTHESIS",
    "OPERATIONAL",
    "SCHEDULER_ARTIFACT",
    "SELF_KNOWLEDGE",
    "STABLE_FACT",
    "TEMPORARY",
    "WORLD_KNOWLEDGE",
    "categorize",
    "categorize_causal_belief",
    "categorize_concept",
    "categorize_relation",
    "filter_signals",
    "signal_from_causal_belief",
    "signal_from_concept",
    "signal_from_relation",
]
