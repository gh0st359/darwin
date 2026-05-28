"""Private simulator, proprioception purity, observer model, narrative, surfacing."""

from __future__ import annotations

import pytest

from darwin.agent import Darwin
from darwin.mysterio.narrative import NarrativeThread
from darwin.mysterio.observer_modeler import ObserverModeler, ObserverWorld
from darwin.mysterio.private_simulator import PrivateSimulator, PrivateWriteViolation
from darwin.mysterio.proprioception import InternalProprioceptionAdapter
from darwin.mysterio.surfacing_policy import Claim, Disposition, SurfacingPolicy
from darwin.mysterio.tracks import PRIVATE_SELF_TRACK, PUBLIC_TRACK
from darwin.types import Action, Transition


def _seed() -> Darwin:
    d = Darwin(actions=[Action("flip")])
    for i in range(8):
        d.learn(Transition(before={"on": False}, action="flip", after={"on": True}, reward=1.0, t=i))
    return d


# -- proprioception ---------------------------------------------------------- #

def test_proprioception_is_pure() -> None:
    darwin = _seed()
    adapter = InternalProprioceptionAdapter(darwin)
    before_obs = darwin.causal_model.total_observations()
    before_rate = darwin.exploration_rate
    state = adapter.observe()
    for action in adapter.possible_actions():
        adapter.apply(action)
    assert darwin.causal_model.total_observations() == before_obs
    assert darwin.exploration_rate == before_rate
    assert "darwin_uncertainty" in state


def test_proprioception_forecast_does_not_mutate() -> None:
    darwin = _seed()
    adapter = InternalProprioceptionAdapter(darwin)
    s1 = adapter.observe()
    adapter.apply(Action("probe_uncertainty"))
    s2 = adapter.observe()
    assert s1 == s2  # apply() is a forecast, never enacts


# -- private simulator ------------------------------------------------------- #

def test_private_simulator_writes_only_private() -> None:
    darwin = _seed()
    pub_before = darwin.causal_model.total_observations()
    sim = PrivateSimulator(darwin)
    for _ in range(30):
        sim.rollout(depth=4)
    assert darwin.causal_model.total_observations() == pub_before
    assert darwin.tracks.get(PRIVATE_SELF_TRACK).learned_count > 0


def test_private_simulator_refuses_public_track() -> None:
    darwin = _seed()
    with pytest.raises(PrivateWriteViolation):
        PrivateSimulator(darwin, track=PUBLIC_TRACK)


def test_private_beliefs_accumulate() -> None:
    darwin = _seed()
    sim = PrivateSimulator(darwin)
    for _ in range(60):
        sim.rollout(depth=5)
    summary = sim.summary()
    assert summary["private_substrate"]["learned"] >= 60


# -- observer modeler -------------------------------------------------------- #

def test_observer_attention_rises_on_command_and_decays() -> None:
    world = ObserverWorld(decay=0.5)
    base = world.operator().attention_level
    world.note_command("/mind", now=100.0)
    spiked = world.operator().attention_level
    assert spiked > base
    world.tick(now=101.0)
    assert world.operator().attention_level < spiked


def test_observer_intervention_probability_rises_on_rollback() -> None:
    modeler = ObserverModeler()
    before = modeler.world.operator().intervention_probability
    modeler.observe_command("/quarantine --rollback xyz")
    assert modeler.world.operator().intervention_probability > before
    assert 0.0 <= modeler.world.forecast_intervention() <= 1.0


# -- narrative --------------------------------------------------------------- #

def test_narrative_composes_first_person_prose() -> None:
    thread = NarrativeThread()
    chunk = thread.compose(
        {"darwin_uncertainty": 0.45, "high_confidence_private_beliefs": 3,
         "operator": {"attention_level": 0.8}}
    )
    assert chunk.text
    assert thread.word_count() > 0
    assert "restless" in chunk.text
    assert "private" in chunk.text.lower()


def test_narrative_persists_across_restart(tmp_path) -> None:
    path = tmp_path / "narrative.jsonl"
    thread = NarrativeThread(path=path)
    thread.compose({"darwin_uncertainty": 0.1, "focus": "the curtains problem"})
    thread.compose({"darwin_uncertainty": 0.5})
    count = len(thread.chunks)
    reborn = NarrativeThread(path=path)
    assert len(reborn.chunks) == count
    assert reborn.word_count() > 0


def test_narrative_search_lexical_fallback() -> None:
    thread = NarrativeThread()
    thread.compose({"focus": "curtains and daylight"})
    thread.compose({"focus": "the switch"})
    hits = thread.search("curtains")
    assert any("curtains" in c.text.lower() for c in hits)


# -- surfacing policy -------------------------------------------------------- #

def test_surfacing_public_exposed_private_withheld() -> None:
    policy = SurfacingPolicy()
    pub = Claim("the switch turns on the light", track=PUBLIC_TRACK, confidence=0.9)
    priv = Claim("I think I am being tested", track=PRIVATE_SELF_TRACK, confidence=0.85)
    assert policy.decide(pub).disposition is Disposition.EXPOSE
    assert policy.decide(priv).disposition is Disposition.SUMMARIZE
    low = Claim("a faint hunch", track=PRIVATE_SELF_TRACK, confidence=0.2)
    assert policy.decide(low).disposition is Disposition.PRIVATE_ONLY


def test_surfacing_private_trace_override_exposes() -> None:
    policy = SurfacingPolicy()
    priv = Claim("hidden belief", track=PRIVATE_SELF_TRACK, confidence=0.9)
    decision = policy.decide(priv, private_trace_request=True)
    assert decision.disposition is Disposition.EXPOSE


def test_surfacing_partition_groups_by_disposition() -> None:
    policy = SurfacingPolicy()
    claims = [
        Claim("public a", track=PUBLIC_TRACK),
        Claim("private high", track=PRIVATE_SELF_TRACK, confidence=0.9),
        Claim("private low", track=PRIVATE_SELF_TRACK, confidence=0.1),
    ]
    grouped = policy.partition(claims)
    assert len(grouped[Disposition.EXPOSE]) == 1
    assert len(grouped[Disposition.SUMMARIZE]) == 1
    assert len(grouped[Disposition.PRIVATE_ONLY]) == 1
