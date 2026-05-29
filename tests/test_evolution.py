"""Tests for the evolution safeguards (ledger / rollback / scoring / recovery)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from darwin.evolution import (
    MutationLedger,
    MutationScorer,
    RecoveryMonitor,
    RollbackChain,
)


# --------------------------------------------------------------------------- #
# MutationLedger
# --------------------------------------------------------------------------- #


def test_append_assigns_sequential_versions() -> None:
    ledger = MutationLedger()
    a = ledger.append(kind="PARAMETER", description="a", improvement=0.1, accepted=True)
    b = ledger.append(kind="PARAMETER", description="b", improvement=0.2, accepted=True)
    assert a.version == 1 and b.version == 2
    assert b.parent_version == a.version


def test_get_by_version() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.1, accepted=True)
    assert ledger.get(1) is not None
    assert ledger.get(999) is None


def test_summary_reports_counts() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.1, accepted=True)
    ledger.append(kind="A", description="y", improvement=0.0, accepted=False)
    s = ledger.summary()
    assert s["total"] == 2
    assert s["active"] == 1
    assert s["rejected"] == 1


def test_mark_rolled_back_is_idempotent() -> None:
    ledger = MutationLedger()
    a = ledger.append(kind="A", description="x", improvement=0.1, accepted=True)
    assert ledger.mark_rolled_back(a.version, 99) is True
    assert ledger.mark_rolled_back(a.version, 99) is False
    assert ledger.get(a.version).rolled_back_at is not None


def test_active_excludes_rolled_back_and_rejected() -> None:
    ledger = MutationLedger()
    a = ledger.append(kind="A", description="x", improvement=0.1, accepted=True)
    b = ledger.append(kind="A", description="y", improvement=0.2, accepted=True)
    c = ledger.append(kind="A", description="z", improvement=0.0, accepted=False)
    ledger.mark_rolled_back(a.version, 999)
    active = ledger.active()
    assert [r.version for r in active] == [b.version]


# --------------------------------------------------------------------------- #
# RollbackChain
# --------------------------------------------------------------------------- #


@dataclass
class _FakeSnapshot:
    snapshot_id: str
    exploration_rate: float = 0.2
    causal: dict = field(default_factory=lambda: {"min_samples": 3})
    planner: dict = field(default_factory=dict)


class _FakeSnapshotStore:
    def __init__(self) -> None:
        self._items: dict[str, _FakeSnapshot] = {}

    def add(self, snap: _FakeSnapshot) -> None:
        self._items[snap.snapshot_id] = snap

    def get(self, snapshot_id: str) -> _FakeSnapshot | None:
        return self._items.get(snapshot_id)


def test_rollback_to_unknown_version_fails_gracefully() -> None:
    ledger = MutationLedger()
    chain = RollbackChain(
        ledger=ledger,
        snapshot_store=_FakeSnapshotStore(),
        apply_snapshot=lambda snap: None,
    )
    result = chain.rollback_to(999)
    assert not result.success
    assert "no such version" in result.notes


def test_rollback_requires_snapshot_id_before() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.1, accepted=True)
    chain = RollbackChain(
        ledger=ledger,
        snapshot_store=_FakeSnapshotStore(),
        apply_snapshot=lambda snap: None,
    )
    result = chain.rollback_to(1)
    assert not result.success
    assert "snapshot_id_before" in result.notes


def test_rollback_applies_snapshot_and_marks_rolled_back() -> None:
    ledger = MutationLedger()
    store = _FakeSnapshotStore()
    store.add(_FakeSnapshot(snapshot_id="snap-0"))
    a = ledger.append(
        kind="A", description="x", improvement=0.1, accepted=True,
        snapshot_id_before="snap-0",
    )
    ledger.append(kind="A", description="y", improvement=0.05, accepted=True)
    captured = {"called": 0}

    def apply(snap: _FakeSnapshot) -> None:
        captured["called"] += 1
        captured["snapshot"] = snap

    chain = RollbackChain(
        ledger=ledger,
        snapshot_store=store,
        apply_snapshot=apply,
    )
    result = chain.rollback_to(a.version)
    assert result.success
    assert captured["called"] == 1
    assert captured["snapshot"].snapshot_id == "snap-0"
    assert ledger.get(a.version).rolled_back_at is not None
    # A rollback record was appended.
    assert ledger.summary()["rollback_records"] == 1


def test_step_back_picks_most_recent_active() -> None:
    ledger = MutationLedger()
    store = _FakeSnapshotStore()
    store.add(_FakeSnapshot(snapshot_id="snap-1"))
    ledger.append(kind="A", description="old", improvement=0.1, accepted=True)
    ledger.append(
        kind="A", description="newest", improvement=0.2, accepted=True,
        snapshot_id_before="snap-1",
    )
    chain = RollbackChain(
        ledger=ledger,
        snapshot_store=store,
        apply_snapshot=lambda snap: None,
    )
    result = chain.step_back(n=1)
    assert result.success
    assert result.rolled_back_to_version == 2


def test_rollback_with_failing_apply_returns_failure() -> None:
    ledger = MutationLedger()
    store = _FakeSnapshotStore()
    store.add(_FakeSnapshot(snapshot_id="snap-1"))
    ledger.append(
        kind="A", description="x", improvement=0.1, accepted=True,
        snapshot_id_before="snap-1",
    )

    def broken(snap):
        raise RuntimeError("boom")

    chain = RollbackChain(
        ledger=ledger, snapshot_store=store, apply_snapshot=broken,
    )
    result = chain.rollback_to(1)
    assert not result.success
    assert "boom" in result.notes


# --------------------------------------------------------------------------- #
# MutationScorer
# --------------------------------------------------------------------------- #


def test_score_for_active_mutation_full_retention() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.5, accepted=True)
    scorer = MutationScorer(ledger)
    s = scorer.score(1)
    assert s is not None
    assert s.retention == 1.0
    assert s.improvement == 0.5


def test_score_for_rolled_back_mutation_zero_retention() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.5, accepted=True)
    ledger.mark_rolled_back(1, 99)
    s = MutationScorer(ledger).score(1)
    assert s is not None
    assert s.retention == 0.0


def test_score_for_rejected_returns_none() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.0, accepted=False)
    assert MutationScorer(ledger).score(1) is None


def test_ranked_orders_by_composite() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="low", improvement=0.1, accepted=True)
    ledger.append(kind="A", description="high", improvement=0.9, accepted=True)
    ledger.append(kind="A", description="mid", improvement=0.4, accepted=True)
    ranked = MutationScorer(ledger).ranked()
    composites = [s.composite for s in ranked]
    assert composites == sorted(composites, reverse=True)
    assert ranked[0].improvement == 0.9


# --------------------------------------------------------------------------- #
# RecoveryMonitor
# --------------------------------------------------------------------------- #


def test_monitor_returns_no_recommendation_with_too_few_samples() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.5, accepted=True)
    monitor = RecoveryMonitor(ledger=ledger, baseline_window=4)
    for _ in range(3):
        rec = monitor.observe(0.8)
    assert rec is None


def test_monitor_no_recommendation_when_health_stable() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.5, accepted=True)
    monitor = RecoveryMonitor(ledger=ledger, baseline_window=3, drop_threshold=0.2)
    rec = None
    for _ in range(10):
        rec = monitor.observe(0.85)
    assert rec is None


def test_monitor_recommends_when_health_drops_below_threshold() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.5, accepted=True)
    monitor = RecoveryMonitor(ledger=ledger, baseline_window=3, drop_threshold=0.2)
    # Three healthy samples to populate the baseline.
    for _ in range(3):
        monitor.observe(0.85)
    # Then a sustained drop.
    rec = None
    for _ in range(5):
        rec = monitor.observe(0.40)
    assert rec is not None
    assert rec.target_version == 1
    assert rec.confidence > 0
    assert rec.health_drop > 0


def test_recommendations_are_bounded_in_history() -> None:
    ledger = MutationLedger()
    ledger.append(kind="A", description="x", improvement=0.5, accepted=True)
    monitor = RecoveryMonitor(ledger=ledger, baseline_window=2, drop_threshold=0.1)
    for _ in range(3):
        monitor.observe(0.9)
    for _ in range(200):
        monitor.observe(0.3)
    assert len(monitor.recommendations()) <= 64
