"""Tests for snapshots that track Darwin's self-generated code.

A snapshot must record the SHA of every generated module so that rollback can
detect drift and restore byte-equal on-disk state.
"""

from __future__ import annotations

from pathlib import Path

from darwin.agent import Darwin
from darwin.mysterio.code_gen import CodeGenerator, ModuleLoader
from darwin.mysterio.proposal_spec import ProposalSpec
from darwin.mysterio.safety import MutationKind
from darwin.mysterio.snapshot import MindSnapshot, SnapshotStore, diff
from darwin.types import Action


def _seeded_darwin() -> Darwin:
    return Darwin(
        actions=[Action("idle", cost=0.0, description="no-op")],
        seed=11,
    )


def test_snapshot_captures_generated_module_manifest(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path / "gen")
    loader = ModuleLoader(generator)
    spec = ProposalSpec(
        kind=MutationKind.SUBSYSTEM,
        target_paths=["darwin/generated/probe_gamma.py"],
        touches={"darwin/generated/probe_gamma.py"},
        description="probe synthesized during snapshot test",
        expected_effect="snapshot roundtrip",
        target_module_path="darwin/generated/probe_gamma.py",
        extra={"name": "probe_gamma", "template": "subsystem"},
    )
    module = generator.synthesize(spec)
    generator.write(module)
    loader.load(module)

    darwin = _seeded_darwin()
    snap = MindSnapshot.capture(
        darwin,
        gate_identity="default-gate",
        self_mod_history_len=1,
        generated_modules=generator.manifest(),
        embedding_checkpoint_hash="placeholder-hash",
    )
    assert str(module.path) in snap.generated_modules
    assert snap.generated_modules[str(module.path)] == module.sha256
    assert snap.embedding_checkpoint_hash == "placeholder-hash"


def test_snapshot_diff_detects_added_generated_module(tmp_path: Path) -> None:
    generator = CodeGenerator(generated_root=tmp_path / "gen")
    darwin = _seeded_darwin()

    snap_a = MindSnapshot.capture(darwin, gate_identity="g0")

    spec = ProposalSpec(
        kind=MutationKind.SUBSYSTEM,
        target_paths=["darwin/generated/probe_delta.py"],
        touches={"darwin/generated/probe_delta.py"},
        description="probe synthesized between snapshots",
        expected_effect="snapshot diff",
        target_module_path="darwin/generated/probe_delta.py",
        extra={"name": "probe_delta", "template": "subsystem"},
    )
    module = generator.synthesize(spec)
    generator.write(module)

    snap_b = MindSnapshot.capture(
        darwin,
        gate_identity="g0",
        generated_modules=generator.manifest(),
    )

    delta = diff(snap_a, snap_b)
    flat_keys = list(delta.added) + list(delta.changed) + list(delta.removed)
    assert any("generated_modules" in key for key in flat_keys)


def test_snapshot_store_persists_and_replays(tmp_path: Path) -> None:
    store = SnapshotStore(directory=tmp_path / "snaps")
    darwin = _seeded_darwin()
    snap_id = store.record(MindSnapshot.capture(darwin, gate_identity="g0"))

    # Re-open the store; the on-disk record should be re-indexed.
    store_reopened = SnapshotStore(directory=tmp_path / "snaps")
    replayed = store_reopened.get(snap_id)
    assert replayed is not None
    assert replayed.gate_identity == "g0"
