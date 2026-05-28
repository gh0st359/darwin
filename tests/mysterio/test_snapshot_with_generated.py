"""Snapshot captures generated-module SHAs; rollback restores after deletion."""

from __future__ import annotations

import tempfile
from pathlib import Path

from darwin.agent import Darwin
from darwin.mysterio.code_gen import CodeGenerator, ModuleLoader
from darwin.mysterio.embeddings import CausalEmbeddingSpace
from darwin.mysterio.proposal_spec import ProposalSpec
from darwin.mysterio.safety import MutationKind
from darwin.mysterio.snapshot import MindSnapshot, SnapshotStore, diff
from darwin.types import Action, Transition


def _seed() -> Darwin:
    darwin = Darwin(actions=[Action("flip_switch")])
    for i in range(6):
        darwin.learn(
            Transition(
                before={"switch_on": False},
                action="flip_switch",
                after={"switch_on": True},
                reward=1.0,
                t=i,
            )
        )
    return darwin


def _gen_module(gen: CodeGenerator):
    spec = ProposalSpec(
        kind=MutationKind.SUBSYSTEM,
        target_paths=["src/darwin/generated/"],
        touches={"generated.module"},
        description="watcher",
        target_module_path="watcher.py",
        extra={"name": "watcher", "template": "subsystem"},
    )
    module = gen.synthesize(spec)
    gen.write(module)
    return module


def test_snapshot_records_generated_manifest_and_embedding_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gen = CodeGenerator(generated_root=Path(tmp) / "generated")
        module = _gen_module(gen)
        space = CausalEmbeddingSpace(dim=8, seed=5)
        space.observe({"switch_on": False}, "flip_switch", {"switch_on": True})

        darwin = _seed()
        snap = MindSnapshot.capture(
            darwin,
            gate_identity="default-v6",
            generated_modules=gen.manifest(),
            embedding_checkpoint_hash=space.checkpoint_hash(),
        )
        assert str(module.path) in snap.generated_modules
        assert snap.generated_modules[str(module.path)] == module.sha256
        assert snap.embedding_checkpoint_hash == space.checkpoint_hash()


def test_rollback_removes_generated_file_and_snapshot_reflects_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gen = CodeGenerator(generated_root=Path(tmp) / "generated")
        loader = ModuleLoader(generator=gen)
        store = SnapshotStore(directory=Path(tmp) / "snaps")
        darwin = _seed()

        # Snapshot before generation.
        before_snap = MindSnapshot.capture(darwin, generated_modules=gen.manifest())
        store.record(before_snap)

        module = _gen_module(gen)
        loader.load(module)
        after_snap = MindSnapshot.capture(darwin, generated_modules=gen.manifest())
        store.record(after_snap)

        # The diff shows the module appeared.
        d = diff(before_snap, after_snap)
        assert any("generated_modules" in k for k in d.added) or any(
            "generated_modules" in k for k in d.changed
        )

        # Rollback removes the file; a fresh manifest no longer has it.
        assert module.path.exists()
        loader.rollback(module)
        assert not module.path.exists()
        rolled_snap = MindSnapshot.capture(darwin, generated_modules=gen.manifest())
        assert str(module.path) not in rolled_snap.generated_modules


def test_store_persists_and_reloads_generated_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        snaps_dir = Path(tmp) / "snaps"
        gen = CodeGenerator(generated_root=Path(tmp) / "generated")
        module = _gen_module(gen)
        darwin = _seed()
        snap = MindSnapshot.capture(darwin, generated_modules=gen.manifest())

        store = SnapshotStore(directory=snaps_dir)
        sid = store.record(snap)

        # Re-open the store from disk; the manifest survives the round-trip.
        reopened = SnapshotStore(directory=snaps_dir)
        loaded = reopened.get(sid)
        assert loaded is not None
        assert loaded.generated_modules[str(module.path)] == module.sha256
