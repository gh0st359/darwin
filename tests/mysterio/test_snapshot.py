"""Tests for MindSnapshot capture + diff + SnapshotStore."""

from __future__ import annotations

import tempfile
from pathlib import Path

from darwin.agent import Darwin
from darwin.mysterio.snapshot import MindSnapshot, SnapshotStore, diff
from darwin.types import Action, Transition


def _seed_darwin() -> Darwin:
    darwin = Darwin(actions=[Action("flip_switch"), Action("open_curtains")])
    for index in range(6):
        darwin.learn(
            Transition(
                before={"switch_on": False, "room_bright": False},
                action="flip_switch",
                after={"switch_on": True, "room_bright": True},
                reward=1.0,
                t=index,
            )
        )
    return darwin


def test_mind_snapshot_captures_substrate_state() -> None:
    darwin = _seed_darwin()
    snap = MindSnapshot.capture(darwin)
    assert snap.snapshot_id
    assert snap.causal["min_samples"] == darwin.causal_model.min_samples
    assert snap.causal["total_observations"] == darwin.causal_model.total_observations()
    assert snap.exploration_rate == darwin.exploration_rate
    assert isinstance(snap.causal["beliefs"], list)


def test_diff_reports_changed_fields() -> None:
    darwin = _seed_darwin()
    snap_a = MindSnapshot.capture(darwin)
    darwin.causal_model.min_samples = 7
    darwin.exploration_rate = 0.42
    snap_b = MindSnapshot.capture(darwin)

    d = diff(snap_a, snap_b)
    keys = list(d.changed)
    assert any("causal.min_samples" in k for k in keys)
    assert any("exploration_rate" in k for k in keys)
    assert "no substantive change" not in d.summary


def test_snapshot_store_persists_and_lists() -> None:
    darwin = _seed_darwin()
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(directory=tmp)
        snap_a = MindSnapshot.capture(darwin)
        snap_b = MindSnapshot.capture(darwin)
        store.record(snap_a)
        store.record(snap_b)
        assert len(store) == 2
        recent = store.recent(limit=10)
        ids = {s.snapshot_id for s in recent}
        assert {snap_a.snapshot_id, snap_b.snapshot_id} <= ids

        # Reload from disk
        store2 = SnapshotStore(directory=tmp)
        assert len(store2) == 2
        assert store2.get(snap_a.snapshot_id) is not None


def test_content_hash_is_deterministic_per_state() -> None:
    darwin = _seed_darwin()
    snap = MindSnapshot.capture(darwin)
    assert snap.content_hash() == snap.content_hash()
