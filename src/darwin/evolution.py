"""Evolution safeguards — versioning, rollback chain, scoring, recovery.

This module sits *on top of* the v6 self-modification apparatus
(SelfModificationEngine + MetaAcceptGate + QuarantineQueue + SnapshotStore).
It does not replace any of them. The principle is "keep evolution powerful;
add safety nets that surface tradeoffs the operator can act on".

Four mechanisms:

  1. **MutationLedger** — every accepted modification gets a sequential
     version number, a parent version, a content hash, and a structured
     record. The ledger is append-only; rollbacks add a new version that
     references the rolled-back parent, so there is never a destructive
     edit to history.
  2. **RollbackChain** — given a target version, restore the
     ``MindSnapshot`` captured immediately *before* that mutation
     landed. The ledger records the rollback as a new version (kind:
     ``rollback``) pointing at the resurrected parent.
  3. **MutationScore** — derived score per mutation: improvement
     reported by the accept gate, plus retention (still active or
     rolled back), plus downstream impact (whether subsequent
     mutations touched the same paths).
  4. **RecoveryMonitor** — periodically checks the composite health of
     the substrate. When recent mutations correlate with degraded
     health, the monitor *proposes* a rollback target with a rationale.
     It does not act automatically unless the operator explicitly
     enables ``auto_rollback`` mode.

None of this restricts Darwin's evolution; it just makes the
consequences observable and reversible. A mutation that empirically
improves the system stays. A mutation that empirically degrades it
surfaces a rollback recommendation the operator can take or ignore.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Versioned ledger
# --------------------------------------------------------------------------- #


def _content_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class MutationRecord:
    """One entry in the mutation ledger."""

    version: int
    parent_version: int | None
    kind: str            # mutation kind (PARAMETER / RULE / MODULE / SUBSYSTEM / rollback)
    description: str
    improvement: float
    accepted: bool
    snapshot_id_before: str = ""
    snapshot_id_after: str = ""
    content_hash: str = ""
    rationale: str = ""
    created_at: float = field(default_factory=time.time)
    rolled_back_at: float | None = None
    rolled_back_by: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class MutationLedger:
    """Append-only versioned record of accepted (and rolled-back) mutations.

    Thread-safe. Version numbers start at 1 and never repeat. Every record
    carries a parent version so the operator can trace lineage.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: list[MutationRecord] = []
        self._by_version: dict[int, MutationRecord] = {}
        self._next_version = 1

    def append(
        self,
        *,
        kind: str,
        description: str,
        improvement: float,
        accepted: bool,
        snapshot_id_before: str = "",
        snapshot_id_after: str = "",
        rationale: str = "",
        metadata: dict[str, Any] | None = None,
        parent_version: int | None = None,
    ) -> MutationRecord:
        with self._lock:
            version = self._next_version
            self._next_version += 1
            if parent_version is None:
                parent_version = self._records[-1].version if self._records else None
            content_hash = _content_hash({
                "kind": kind,
                "description": description,
                "improvement": improvement,
                "accepted": accepted,
                "snapshot_id_before": snapshot_id_before,
                "snapshot_id_after": snapshot_id_after,
                "metadata": metadata or {},
            })
            record = MutationRecord(
                version=version,
                parent_version=parent_version,
                kind=kind,
                description=description,
                improvement=improvement,
                accepted=accepted,
                snapshot_id_before=snapshot_id_before,
                snapshot_id_after=snapshot_id_after,
                content_hash=content_hash,
                rationale=rationale,
                metadata=dict(metadata or {}),
            )
            self._records.append(record)
            self._by_version[version] = record
            return record

    def get(self, version: int) -> MutationRecord | None:
        with self._lock:
            return self._by_version.get(version)

    def latest(self, n: int = 8) -> list[MutationRecord]:
        with self._lock:
            return list(self._records[-n:])

    def all(self) -> list[MutationRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def mark_rolled_back(self, version: int, rolled_back_by: int) -> bool:
        with self._lock:
            record = self._by_version.get(version)
            if record is None or record.rolled_back_at is not None:
                return False
            record.rolled_back_at = time.time()
            record.rolled_back_by = rolled_back_by
            return True

    def active(self) -> list[MutationRecord]:
        with self._lock:
            return [r for r in self._records if r.accepted and r.rolled_back_at is None]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": len(self._records),
                "active": sum(
                    1 for r in self._records
                    if r.accepted and r.rolled_back_at is None
                ),
                "rolled_back": sum(
                    1 for r in self._records if r.rolled_back_at is not None
                ),
                "rejected": sum(1 for r in self._records if not r.accepted),
                "rollback_records": sum(1 for r in self._records if r.kind == "rollback"),
            }


# --------------------------------------------------------------------------- #
# Rollback chain
# --------------------------------------------------------------------------- #


@dataclass
class RollbackResult:
    """Outcome of an attempted rollback."""

    success: bool
    rolled_back_to_version: int | None = None
    restored_snapshot_id: str = ""
    new_version: int | None = None
    notes: str = ""


class RollbackChain:
    """Restore prior mind state by traversing the snapshot store backwards.

    Takes a target version and looks up the snapshot that was captured
    *before* that mutation landed. The runtime applies that snapshot
    (e.g. via the meta-gate or by reloading state from the snapshot
    store). A new ledger entry records the rollback so the lineage is
    visible.

    The actual *application* of the snapshot is delegated to a callable
    supplied at construction time (``apply_snapshot``). The chain does
    not assume anything about how the runtime materializes snapshot
    contents — it just orchestrates the bookkeeping.
    """

    def __init__(
        self,
        *,
        ledger: MutationLedger,
        snapshot_store: Any,
        apply_snapshot,
    ) -> None:
        self.ledger = ledger
        self.snapshot_store = snapshot_store
        self.apply_snapshot = apply_snapshot

    def rollback_to(self, target_version: int, *, reason: str = "") -> RollbackResult:
        target = self.ledger.get(target_version)
        if target is None:
            return RollbackResult(success=False, notes=f"no such version: {target_version}")
        if target.snapshot_id_before == "":
            return RollbackResult(
                success=False,
                notes=f"version {target_version} has no snapshot_id_before recorded",
            )
        snapshot = self.snapshot_store.get(target.snapshot_id_before) if self.snapshot_store else None
        if snapshot is None:
            return RollbackResult(
                success=False,
                notes=f"snapshot {target.snapshot_id_before!r} not found in store",
            )
        try:
            self.apply_snapshot(snapshot)
        except Exception as exc:
            return RollbackResult(
                success=False,
                notes=f"apply_snapshot failed: {type(exc).__name__}: {exc}",
            )
        new_record = self.ledger.append(
            kind="rollback",
            description=f"rolled back to version {target_version} (state before that mutation)",
            improvement=0.0,
            accepted=True,
            snapshot_id_before=target.snapshot_id_before,
            snapshot_id_after=target.snapshot_id_before,
            rationale=reason or "operator-initiated rollback",
            metadata={"rolled_back_target": target_version},
            parent_version=target.parent_version,
        )
        # Mark all versions between target..latest as rolled back by this new
        # rollback record so the active set updates.
        with self.ledger._lock:
            for record in self.ledger._records:
                if (
                    record.version >= target_version
                    and record.version != new_record.version
                    and record.rolled_back_at is None
                    and record.accepted
                ):
                    record.rolled_back_at = time.time()
                    record.rolled_back_by = new_record.version
        return RollbackResult(
            success=True,
            rolled_back_to_version=target_version,
            restored_snapshot_id=target.snapshot_id_before,
            new_version=new_record.version,
        )

    def step_back(self, n: int = 1, *, reason: str = "") -> RollbackResult:
        """Roll back the most recent ``n`` accepted mutations."""

        with self.ledger._lock:
            active = [r for r in self.ledger._records if r.accepted and r.rolled_back_at is None]
        if not active:
            return RollbackResult(success=False, notes="no active mutations to roll back")
        target = active[-min(n, len(active))]
        return self.rollback_to(
            target.version,
            reason=reason or f"step_back({n})",
        )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass
class MutationScore:
    """Derived score for a single mutation."""

    version: int
    improvement: float
    retention: float           # 1.0 = still active, 0.0 = rolled back
    downstream_impact: int     # later mutations that touched related paths
    composite: float           # combined score for ranking

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class MutationScorer:
    """Compute derived scores per mutation from ledger state."""

    def __init__(self, ledger: MutationLedger) -> None:
        self.ledger = ledger

    def score(self, version: int) -> MutationScore | None:
        record = self.ledger.get(version)
        if record is None or not record.accepted:
            return None
        retention = 1.0 if record.rolled_back_at is None else 0.0
        downstream = 0
        # Heuristic: any later record whose metadata references this
        # version (e.g., touches same target) counts as downstream impact.
        for later in self.ledger.all():
            if later.version <= version:
                continue
            if later.metadata.get("references_version") == version:
                downstream += 1
            if later.parent_version == version:
                downstream += 1
        composite = (
            0.6 * record.improvement
            + 0.3 * retention
            + 0.1 * min(1.0, downstream / 4.0)
        )
        return MutationScore(
            version=version,
            improvement=record.improvement,
            retention=retention,
            downstream_impact=downstream,
            composite=composite,
        )

    def all_scores(self) -> list[MutationScore]:
        scores: list[MutationScore] = []
        for record in self.ledger.all():
            s = self.score(record.version)
            if s is not None:
                scores.append(s)
        return scores

    def ranked(self, limit: int = 10) -> list[MutationScore]:
        scores = self.all_scores()
        scores.sort(key=lambda s: s.composite, reverse=True)
        return scores[:limit]


# --------------------------------------------------------------------------- #
# Recovery monitor
# --------------------------------------------------------------------------- #


@dataclass
class HealthSample:
    at: float
    composite: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class RecoveryRecommendation:
    """Advisory: rollback target + rationale."""

    target_version: int | None
    rationale: str
    confidence: float
    health_drop: float

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class RecoveryMonitor:
    """Watch composite health, recommend rollbacks when health degrades.

    Health is a caller-supplied scalar between 0 and 1 (or any
    monotonically-meaningful range). The monitor keeps a bounded ring
    buffer of samples; on each ``observe`` call it compares the recent
    average to a baseline and, if a drop exceeds ``drop_threshold``,
    surfaces a RecoveryRecommendation pointing at the most recent
    *non-rollback* accepted mutation.

    The monitor is *advisory only* by default. ``auto_rollback`` can be
    set to True to have the runtime actually invoke a rollback when
    confidence exceeds ``auto_rollback_confidence``; this is opt-in.
    """

    def __init__(
        self,
        *,
        ledger: MutationLedger,
        window: int = 32,
        baseline_window: int = 8,
        drop_threshold: float = 0.15,
        auto_rollback: bool = False,
        auto_rollback_confidence: float = 0.75,
    ) -> None:
        self.ledger = ledger
        self.window = window
        self.baseline_window = baseline_window
        self.drop_threshold = drop_threshold
        self.auto_rollback = auto_rollback
        self.auto_rollback_confidence = auto_rollback_confidence
        self._samples: deque[HealthSample] = deque(maxlen=window)
        self._recommendations: list[RecoveryRecommendation] = []

    def observe(
        self, composite: float, *, components: dict[str, float] | None = None,
    ) -> RecoveryRecommendation | None:
        sample = HealthSample(
            at=time.time(),
            composite=float(composite),
            components=dict(components or {}),
        )
        self._samples.append(sample)
        if len(self._samples) < self.baseline_window + 2:
            return None
        baseline_samples = list(self._samples)[: self.baseline_window]
        recent_samples = list(self._samples)[-self.baseline_window:]
        if not baseline_samples or not recent_samples:
            return None
        baseline = sum(s.composite for s in baseline_samples) / len(baseline_samples)
        recent = sum(s.composite for s in recent_samples) / len(recent_samples)
        drop = baseline - recent
        if drop < self.drop_threshold:
            return None
        # Pick the most recent active non-rollback mutation as the target.
        target = None
        for record in reversed(self.ledger.all()):
            if (
                record.accepted
                and record.rolled_back_at is None
                and record.kind != "rollback"
            ):
                target = record
                break
        if target is None:
            return None
        rationale = (
            f"composite health dropped from baseline {baseline:.3f} to recent "
            f"average {recent:.3f} (drop {drop:.3f}). Most recent active "
            f"mutation v{target.version} ({target.kind}) is a candidate."
        )
        confidence = min(1.0, drop / max(self.drop_threshold * 2.0, 1e-6))
        rec = RecoveryRecommendation(
            target_version=target.version,
            rationale=rationale,
            confidence=confidence,
            health_drop=drop,
        )
        self._recommendations.append(rec)
        if len(self._recommendations) > 64:
            self._recommendations = self._recommendations[-64:]
        return rec

    def recommendations(self) -> list[RecoveryRecommendation]:
        return list(self._recommendations)

    def samples(self) -> list[HealthSample]:
        return list(self._samples)


__all__ = [
    "HealthSample",
    "MutationLedger",
    "MutationRecord",
    "MutationScore",
    "MutationScorer",
    "RecoveryMonitor",
    "RecoveryRecommendation",
    "RollbackChain",
    "RollbackResult",
]
