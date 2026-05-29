"""Tests for v9 open-ended growth: world synthesis, live research, modalities."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest

from darwin.mysterio.code_gen import CodeGenerator, ModuleLoader
from darwin.mysterio.meta_proposer import MetaProposer, MetaProposerContext
from darwin.mysterio.modalities.code import CodeModalityAdapter
from darwin.mysterio.modalities.web import WebModalityAdapter
from darwin.mysterio.research_loop import (
    PROTECTED_TARGETS,
    LiveResearcher,
    ResearchFinding,
)
from darwin.mysterio.safety import ContainmentError, MutationKind
from darwin.mysterio.world_synthesis import WorldHypothesis, WorldSynthesizer


class _StubWorldModel:
    def __init__(self, variables: dict[str, bool]) -> None:
        self.variables = variables


class _StubDarwin:
    def __init__(self) -> None:
        self.world_model = _StubWorldModel(
            {"room_bright": False, "fuse_intact": True, "switch_on": False, "curtains_open": False}
        )
        self.causal_model = None
        self.tracks = None


def test_world_synthesizer_emits_proposal_with_parsable_source() -> None:
    synth = WorldSynthesizer()
    proposals = synth.propose(_StubDarwin())
    assert proposals
    spec = proposals[0]
    assert spec.kind is MutationKind.SUBSYSTEM
    # The generated code must parse cleanly.
    ast.parse(spec.generated_code)
    # And it must declare the canonical entrypoint for the loader.
    assert "def build(" in spec.generated_code


def test_world_synthesizer_dedupes_by_signature() -> None:
    synth = WorldSynthesizer()
    darwin = _StubDarwin()
    first = synth.propose(darwin)
    second = synth.propose(darwin)
    assert first
    assert second == []


def test_world_proposal_lands_on_disk_and_imports(tmp_path: Path) -> None:
    synth = WorldSynthesizer()
    spec = synth.propose(_StubDarwin())[0]
    generator = CodeGenerator(generated_root=tmp_path)
    module = generator.synthesize(spec)
    generator.write(module)
    loader = ModuleLoader(generator)
    py_module = loader.load(module)
    instance = loader.get_built(module.qualified_name)
    # World template exposes the World protocol: observe/possible_actions/apply.
    assert callable(getattr(instance, "observe", None))
    assert callable(getattr(instance, "possible_actions", None))
    assert callable(getattr(instance, "apply", None))


def test_live_researcher_protects_instruments() -> None:
    researcher = LiveResearcher(meta_proposer=MetaProposer())

    for target in PROTECTED_TARGETS:
        with pytest.raises(ContainmentError):
            LiveResearcher.cannot_collide([target])

    class _BenignStrategy:
        target_paths = ["darwin.mysterio.meta_proposer"]

        def __call__(self, _context: MetaProposerContext) -> list:
            return []

    researcher.register_strategy("benign", _BenignStrategy())
    assert "benign" in researcher.registered_strategies

    class _CollidingStrategy:
        target_paths = ["darwin.mysterio.probes.DivergenceProbe"]

        def __call__(self, _context: MetaProposerContext) -> list:
            return []

    with pytest.raises(ContainmentError):
        researcher.register_strategy("colliding", _CollidingStrategy())


def test_research_finding_record_serializes() -> None:
    finding = ResearchFinding(summary="something stable", confidence=0.7)
    record = finding.to_record()
    assert record["summary"] == "something stable"
    assert record["confidence"] == 0.7


def test_code_modality_adapter_emits_transitions_on_first_scan(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    adapter = CodeModalityAdapter(root=tmp_path)
    transitions = adapter.scan()
    assert len(transitions) == 2
    # A second scan with no changes is a no-op.
    again = adapter.scan()
    assert again == []
    # Modify a file: should surface a 'code:changed' transition.
    (tmp_path / "a.py").write_text("x = 99\n")
    changed = adapter.scan()
    assert any("changed" in t.action for t in changed)


def test_web_modality_inactive_emits_failed_transition_cleanly() -> None:
    adapter = WebModalityAdapter(active=False)
    transitions = adapter.observe(["http://does-not-matter.example"])
    assert transitions
    assert any("failed" in t.action for t in transitions)


def test_code_modality_default_track_is_grounded(tmp_path: Path) -> None:
    adapter = CodeModalityAdapter(root=tmp_path)
    assert adapter.track == "grounded"


def test_web_modality_default_track_is_grounded() -> None:
    adapter = WebModalityAdapter(active=False)
    assert adapter.track == "grounded"
