"""Tests for QuarantineQueue: tag-and-inspect register."""

from __future__ import annotations

from darwin.mysterio.quarantine import QuarantineQueue, QuarantineStatus
from darwin.mysterio.safety import MutationKind


def test_parameter_kind_is_not_recorded() -> None:
    queue = QuarantineQueue()
    entry = queue.submit(
        proposal_id="p1",
        kind=MutationKind.PARAMETER,
        description="scalar tweak",
        snapshot_id="s1",
    )
    # PARAMETER kinds return a marker entry but are not stored.
    assert entry.entry_id == ""
    assert len(queue) == 0


def test_kernel_kind_is_recorded_and_returns_entry() -> None:
    queue = QuarantineQueue()
    entry = queue.submit(
        proposal_id="p1",
        kind=MutationKind.KERNEL,
        description="new kernel job",
        snapshot_id="s1",
    )
    assert entry.entry_id
    assert entry.status is QuarantineStatus.APPLIED
    assert len(queue) == 1
    assert queue.get(entry.entry_id) is entry


def test_persist_callback_is_invoked() -> None:
    sink: list[dict] = []
    queue = QuarantineQueue(persist=sink.append)
    queue.submit(
        proposal_id="p2",
        kind=MutationKind.GATE,
        description="alt gate",
        snapshot_id="s2",
    )
    assert len(sink) == 1
    assert sink[0]["kind"] == "gate"
    assert sink[0]["status"] == "applied"


def test_rollback_invokes_handler_and_updates_status() -> None:
    invocations: list[str] = []
    queue = QuarantineQueue()
    queue.register_handler(
        MutationKind.MODULE,
        lambda entry: invocations.append(entry.entry_id),
    )
    entry = queue.submit(
        proposal_id="p3",
        kind=MutationKind.MODULE,
        description="new module",
        snapshot_id="s3",
    )
    rolled = queue.rollback(entry.entry_id)
    assert rolled is entry
    assert entry.status is QuarantineStatus.ROLLED_BACK
    assert entry.rolled_back_at is not None
    assert invocations == [entry.entry_id]


def test_pending_filters_only_applied() -> None:
    queue = QuarantineQueue()
    e1 = queue.submit(
        proposal_id="p1",
        kind=MutationKind.KERNEL,
        description="job 1",
        snapshot_id="s1",
    )
    e2 = queue.submit(
        proposal_id="p2",
        kind=MutationKind.KERNEL,
        description="job 2",
        snapshot_id="s2",
    )
    queue.rollback(e1.entry_id)
    pending = queue.pending()
    assert e2 in pending
    assert e1 not in pending
