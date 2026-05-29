"""Tests for the epistemic categorization layer.

These cover the *derivation* of categories from observable belief signals
(provenance, confidence, history, subject) — NOT a hardcoded mapping. The
categories are advisory: any signal can fit multiple categories, and the
mechanism is meant to inform surfacing (suppress noise from /beliefs by
default), not to constrain what Darwin can reason about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from darwin.epistemics import (
    BeliefSignal,
    EpistemicMonitor,
    HYPOTHESIS,
    OPERATIONAL,
    SCHEDULER_ARTIFACT,
    SELF_KNOWLEDGE,
    STABLE_FACT,
    TEMPORARY,
    WORLD_KNOWLEDGE,
    categorize,
    categorize_causal_belief,
    categorize_concept,
    categorize_relation,
    filter_signals,
    signal_from_causal_belief,
    signal_from_concept,
    signal_from_relation,
)


# --------------------------------------------------------------------------- #
# Direct categorization rules.
# --------------------------------------------------------------------------- #


def test_default_category_is_operational_when_no_signal_present() -> None:
    cats = categorize(BeliefSignal())
    assert OPERATIONAL in cats


def test_scheduler_substring_in_name_flags_scheduler_artifact() -> None:
    cats = categorize(BeliefSignal(name="explore_concept:focus"))
    assert SCHEDULER_ARTIFACT in cats
    assert OPERATIONAL in cats


def test_self_substring_in_name_flags_self_knowledge() -> None:
    cats = categorize(BeliefSignal(name="observe_self:darwin_uncertainty"))
    assert SELF_KNOWLEDGE in cats


def test_chat_fused_provenance_yields_world_knowledge() -> None:
    cats = categorize(BeliefSignal(name="dog is_a mammal", provenance="fused"))
    assert WORLD_KNOWLEDGE in cats


def test_tool_provenance_yields_world_knowledge() -> None:
    cats = categorize(BeliefSignal(name="x", provenance="tool"))
    assert WORLD_KNOWLEDGE in cats


def test_hypothesis_provenance_yields_hypothesis_category() -> None:
    cats = categorize(BeliefSignal(name="x", provenance="hypothesis"))
    assert HYPOTHESIS in cats


def test_primitive_provenance_is_operational_scaffolding() -> None:
    cats = categorize(BeliefSignal(name="thing", provenance="primitive"))
    assert OPERATIONAL in cats
    assert SCHEDULER_ARTIFACT not in cats


def test_high_confidence_and_many_samples_promotes_to_stable_fact() -> None:
    cats = categorize(BeliefSignal(
        name="x",
        provenance="fused",
        confidence=0.9,
        samples=12,
        has_contradiction=False,
    ))
    assert STABLE_FACT in cats


def test_contradiction_blocks_stable_fact_promotion() -> None:
    cats = categorize(BeliefSignal(
        name="x",
        provenance="fused",
        confidence=0.9,
        samples=12,
        has_contradiction=True,
    ))
    assert STABLE_FACT not in cats


def test_freshly_arrived_with_one_sample_is_temporary() -> None:
    cats = categorize(BeliefSignal(
        name="x",
        confidence=0.4,
        samples=1,
        age_seconds=10.0,
    ))
    assert TEMPORARY in cats


def test_cross_context_uses_promote_stable_fact() -> None:
    cats = categorize(BeliefSignal(
        name="x",
        confidence=0.6,
        samples=2,
        cross_context_uses=3,
        provenance="fused",
    ))
    assert STABLE_FACT in cats


# --------------------------------------------------------------------------- #
# Filtering.
# --------------------------------------------------------------------------- #


def test_filter_excludes_scheduler_artifacts_by_default() -> None:
    bookkeeping = BeliefSignal(name="explore_concept:focus")
    real = BeliefSignal(name="dog is_a mammal", provenance="fused")
    kept = filter_signals(
        [bookkeeping, real],
        exclude=[SCHEDULER_ARTIFACT],
    )
    assert real in kept
    assert bookkeeping not in kept


def test_filter_include_only_world_knowledge() -> None:
    bookkeeping = BeliefSignal(name="step:focus")
    chat = BeliefSignal(name="x", provenance="fused")
    hyp = BeliefSignal(name="y", provenance="hypothesis")
    kept = filter_signals(
        [bookkeeping, chat, hyp],
        include=[WORLD_KNOWLEDGE],
    )
    assert chat in kept
    assert hyp not in kept
    assert bookkeeping not in kept


def test_filter_handles_empty_iterable() -> None:
    assert filter_signals([], exclude=[OPERATIONAL]) == []


# --------------------------------------------------------------------------- #
# Adapters.
# --------------------------------------------------------------------------- #


@dataclass
class _FakeCausalBelief:
    action: str = "flip_switch"
    variable: str = "room_bright"
    effect: str = "+1"
    confidence: float = 0.9
    samples: int = 10


def test_signal_from_causal_belief_carries_confidence_and_samples() -> None:
    sig = signal_from_causal_belief(_FakeCausalBelief())
    assert sig.confidence == 0.9
    assert sig.samples == 10
    assert sig.name == "flip_switch:room_bright"


def test_categorize_causal_belief_with_scheduler_action_set() -> None:
    cats = categorize_causal_belief(
        _FakeCausalBelief(action="explore_concept"),
        scheduler_actions=["explore_concept", "wander_universe"],
    )
    # The action name itself is in the scheduler set => SCHEDULER_ARTIFACT.
    assert SCHEDULER_ARTIFACT in cats


@dataclass
class _FakeConcept:
    name: str = "dog"
    domain: str = "fused"
    salience: float = 0.7
    visits: int = 2
    created_at: float = 0.0
    derived_from: tuple = ()


def test_signal_from_concept_reads_provenance_from_domain() -> None:
    sig = signal_from_concept(_FakeConcept(domain="fused"))
    assert sig.provenance == "fused"
    sig2 = signal_from_concept(_FakeConcept(domain="structure"))
    # A primitive lives in the structural domain with no derived_from.
    assert sig2.provenance == "primitive"


def test_categorize_concept_for_chat_fused_emits_world_knowledge() -> None:
    cats = categorize_concept(_FakeConcept(name="neuron", domain="fused"))
    assert WORLD_KNOWLEDGE in cats


def test_categorize_concept_for_primitive_emits_operational() -> None:
    cats = categorize_concept(_FakeConcept(name="thing", domain="structure"))
    assert OPERATIONAL in cats


@dataclass
class _FakeRelation:
    source: str = "x"
    target: str = "y"
    kind: str = "is_a"
    weight: float = 1.0
    notes: str = ""


def test_signal_from_relation_reads_provenance_from_notes() -> None:
    sig = signal_from_relation(_FakeRelation(notes="fused from chat: 'x is a y'"))
    assert sig.provenance == "fused"
    sig2 = signal_from_relation(_FakeRelation(notes="derived via composition"))
    assert sig2.provenance == "derived"
    sig3 = signal_from_relation(_FakeRelation(notes="accepted hypothesis via transitive"))
    assert sig3.provenance == "hypothesis"


def test_categorize_relation_for_chat_fused_emits_world_knowledge() -> None:
    cats = categorize_relation(_FakeRelation(notes="fused from chat: 'a is a b'"))
    assert WORLD_KNOWLEDGE in cats


# --------------------------------------------------------------------------- #
# Monitor + drift.
# --------------------------------------------------------------------------- #


def test_monitor_scan_returns_category_counts() -> None:
    monitor = EpistemicMonitor()
    beliefs = [_FakeCausalBelief(action=f"a_{i}") for i in range(5)]
    snapshot = monitor.scan(causal_beliefs=beliefs)
    assert sum(snapshot.values()) > 0


def test_monitor_drift_reports_change_between_scans() -> None:
    monitor = EpistemicMonitor()
    monitor.scan(causal_beliefs=[_FakeCausalBelief()])
    monitor.scan(causal_beliefs=[_FakeCausalBelief(action="explore_concept")])
    drift = monitor.drift()
    assert drift  # at least one category changed between scans


def test_monitor_drift_returns_empty_after_single_scan() -> None:
    monitor = EpistemicMonitor()
    monitor.scan(causal_beliefs=[_FakeCausalBelief()])
    assert monitor.drift() == {}


def test_monitor_history_is_bounded() -> None:
    monitor = EpistemicMonitor()
    for _ in range(80):
        monitor.scan(causal_beliefs=[_FakeCausalBelief()])
    assert len(monitor.history()) <= 64


# --------------------------------------------------------------------------- #
# The whole point: bookkeeping noise is suppressed by default, real facts
# survive.
# --------------------------------------------------------------------------- #


def test_realistic_filter_suppresses_internal_correlations_but_keeps_world_facts() -> None:
    signals = [
        BeliefSignal(name="explore_concept:focus"),          # scheduler
        BeliefSignal(name="wander_universe:secondary_focus"),  # scheduler
        BeliefSignal(name="dog is_a mammal", provenance="fused", confidence=0.9, samples=10),
        BeliefSignal(name="neuron causes thought", provenance="fused", confidence=0.8, samples=4),
        BeliefSignal(name="self_model:darwin_uncertainty"),  # self
    ]
    kept = filter_signals(signals, exclude=[SCHEDULER_ARTIFACT])
    kept_names = {s.name for s in kept}
    assert "dog is_a mammal" in kept_names
    assert "neuron causes thought" in kept_names
    assert "explore_concept:focus" not in kept_names
    assert "wander_universe:secondary_focus" not in kept_names
    # Self-knowledge is allowed by default (only SCHEDULER_ARTIFACT was
    # excluded), which is the right behavior — talking about Darwin's
    # own state IS meaningful, just a different kind of meaningful.
    assert "self_model:darwin_uncertainty" in kept_names
