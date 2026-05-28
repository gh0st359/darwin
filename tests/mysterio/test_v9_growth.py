"""v9 open-ended growth: world synthesis, live research, modalities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from darwin.agent import Darwin
from darwin.mysterio.code_gen import CodeGenerator, ModuleLoader
from darwin.mysterio.meta_proposer import MetaProposer
from darwin.mysterio.modalities.code import CodeModalityAdapter
from darwin.mysterio.modalities.web import WebModalityAdapter
from darwin.mysterio.research_loop import LiveResearcher
from darwin.mysterio.safety import ContainmentError, MutationKind
from darwin.mysterio.world_synthesis import WorldHypothesis, WorldSynthesizer
from darwin.types import Action, Transition


def _seed() -> Darwin:
    d = Darwin(actions=[Action("flip")])
    for i in range(8):
        d.learn(
            Transition(before={"on": False, "bright": False, "warm": False},
                       action="flip", after={"on": True, "bright": True, "warm": True},
                       reward=1.0, t=i)
        )
    return d


# -- world synthesis --------------------------------------------------------- #

def test_synthesizer_discovers_world_from_variables() -> None:
    darwin = _seed()
    synth = WorldSynthesizer()
    hypotheses = synth.discover_hypotheses(darwin)
    assert hypotheses
    assert hypotheses[0].variables


def test_synthesizer_emits_subsystem_proposal_with_source() -> None:
    darwin = _seed()
    synth = WorldSynthesizer()
    specs = synth.propose(darwin)
    assert specs
    spec = specs[0]
    assert spec.kind is MutationKind.SUBSYSTEM
    assert spec.generated_code
    assert "class " in spec.generated_code
    # And it dedupes on a second pass.
    assert synth.propose(darwin) == []


def test_synthesized_world_module_loads_and_runs() -> None:
    darwin = _seed()
    synth = WorldSynthesizer()
    spec = synth.propose(darwin)[0]
    with tempfile.TemporaryDirectory() as tmp:
        gen = CodeGenerator(generated_root=Path(tmp))
        loader = ModuleLoader(generator=gen)
        module = gen.synthesize(spec)
        gen.write(module)
        loader.load(module)
        world = loader.get_built(module.qualified_name)
        assert world is not None
        state = world.observe()
        actions = world.possible_actions()
        assert actions
        new_state, reward = world.apply(actions[0])
        assert isinstance(new_state, dict)
        assert isinstance(reward, float)


# -- live research ----------------------------------------------------------- #

def test_researcher_finds_private_belief_regularity() -> None:
    darwin = _seed()
    # Build up the private substrate.
    for i in range(20):
        darwin.learn(
            Transition(before={"x": False}, action="probe", after={"x": True},
                       reward=1.0, t=20000 + i,
                       metadata={"track": "private_self"})
        )

    class _Rt:
        def __init__(self, d: Darwin) -> None:
            self.darwin = d
            self._loop_state: dict = {}
            self.loop_intervals: dict = {}
            self.divergence_probe = None

    res = LiveResearcher(meta_proposer=MetaProposer())
    findings = res.investigate(_Rt(darwin))
    # private-self regularity should fire when the substrate has enough beliefs
    names = [f.summary for f in findings]
    assert any("private_self" in n for n in names) or any(
        "regularity" in n for n in names
    )


def test_researcher_registers_strategy_against_meta_proposer() -> None:
    mp = MetaProposer()
    res = LiveResearcher(meta_proposer=mp)

    def my_strategy(ctx):  # pragma: no cover - inert
        return []

    my_strategy.target_paths = ["darwin.runtime.cognition_root"]
    res.register_strategy("custom", my_strategy)
    assert "custom" in mp.strategies()
    assert "custom" in res.registered_strategies


def test_researcher_blocks_protected_target_paths() -> None:
    res = LiveResearcher(meta_proposer=MetaProposer())

    def bad_strategy(ctx):  # pragma: no cover - inert
        return []

    bad_strategy.target_paths = ["darwin.mysterio.probes.DivergenceProbe"]
    with pytest.raises(ContainmentError):
        res.register_strategy("collides", bad_strategy)


def test_cannot_collide_static_check() -> None:
    LiveResearcher.cannot_collide(["darwin.runtime.cognition_root"])
    with pytest.raises(ContainmentError):
        LiveResearcher.cannot_collide(
            ["darwin.mysterio.snapshot.SnapshotStore"]
        )


# -- modalities -------------------------------------------------------------- #

def test_code_modality_emits_transitions_on_first_scan(tmp_path) -> None:
    (tmp_path / "a.py").write_text("x=1\n")
    (tmp_path / "b.py").write_text("y=2\n")
    adapter = CodeModalityAdapter(root=tmp_path)
    transitions = adapter.scan()
    assert len(transitions) == 2
    actions = {t.action for t in transitions}
    assert actions == {"code:added"}
    # Second scan: no changes → no transitions.
    assert adapter.scan() == []
    # Change one file.
    (tmp_path / "a.py").write_text("x=2\n")
    changed = adapter.scan()
    assert len(changed) == 1
    assert changed[0].action == "code:changed"


def test_code_modality_inactive_when_root_missing() -> None:
    adapter = CodeModalityAdapter(root="/nonexistent/path/xyz")
    assert not adapter.active
    assert adapter.scan() == []


def test_web_modality_observes_with_inactive_adapter() -> None:
    adapter = WebModalityAdapter()
    adapter.active = False
    transitions = adapter.observe(["http://example.invalid"])
    assert len(transitions) == 1
    assert transitions[0].action == "web:failed"
    assert transitions[0].after["error"] == "adapter inactive"
