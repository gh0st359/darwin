"""Tests for the v7 interior mental life subsystem.

Asserts:
- ``InternalProprioceptionAdapter.observe`` and ``apply`` are pure (state
  hashes before/after unchanged).
- ``InteriorSimulator`` produces non-trivial interior beliefs.
- Nothing leaks into the grounded episodic memory.
- ``EpistemicLeakError`` is raised if anyone tries to point the interior
  simulator at the grounded track.
- ``ObserverModeler`` updates on commands and decays on tick.
- ``NarrativeThread`` composes first-person prose and persists across reload.
- Interior-simulation events publish onto the cognition bus so the brain
  terminal sees the rollouts live.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from darwin.agent import Darwin
from darwin.mysterio.bus import BusEvent, BusTopic, CognitionBus
from darwin.mysterio.embeddings import CausalEmbeddingSpace
from darwin.mysterio.interior_simulator import (
    EpistemicLeakError,
    InteriorSimulator,
    PrivateWriteViolation,
)
from darwin.mysterio.narrative import NarrativeThread
from darwin.mysterio.observer_modeler import ObserverModeler
from darwin.mysterio.proprioception import InternalProprioceptionAdapter
from darwin.mysterio.tracks import GROUNDED_TRACK, INTERIOR_TRACK
from darwin.types import Action


def _bare_darwin() -> Darwin:
    return Darwin(
        actions=[Action("idle", cost=0.0, description="no-op")],
        seed=7,
    )


def _hash_grounded_state(darwin: Darwin) -> str:
    beliefs = [
        {
            "action": getattr(b, "action", ""),
            "variable": getattr(b, "variable", ""),
            "effect": getattr(b, "effect", ""),
            "samples": int(getattr(b, "samples", 0)),
        }
        for b in darwin.causal_model.beliefs(limit=10000)
    ]
    episodes_count = len(getattr(darwin.memory, "episodes", []))
    payload = json.dumps(
        {
            "beliefs": sorted(
                beliefs, key=lambda d: (d["action"], d["variable"], d["effect"])
            ),
            "episodes": episodes_count,
            "exploration_rate": darwin.exploration_rate,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_proprioception_observe_and_apply_are_pure() -> None:
    darwin = _bare_darwin()
    adapter = InternalProprioceptionAdapter(darwin)
    before = _hash_grounded_state(darwin)
    adapter.observe()
    adapter.observe()
    for action in adapter.possible_actions():
        forecast, reward = adapter.apply(action)
        assert isinstance(forecast, dict)
        assert isinstance(reward, float)
    after = _hash_grounded_state(darwin)
    assert before == after


def test_interior_simulator_targeting_grounded_raises() -> None:
    darwin = _bare_darwin()
    with pytest.raises(EpistemicLeakError):
        InteriorSimulator(darwin, track=GROUNDED_TRACK)


def test_legacy_alias_still_resolves() -> None:
    assert PrivateWriteViolation is EpistemicLeakError


def test_interior_rollout_writes_only_to_interior_track() -> None:
    darwin = _bare_darwin()
    sim = InteriorSimulator(darwin)
    before_hash = _hash_grounded_state(darwin)
    for _ in range(5):
        sim.rollout(depth=4)
    after_hash = _hash_grounded_state(darwin)
    assert before_hash == after_hash

    interior_substrate = darwin.tracks.get(INTERIOR_TRACK)
    assert interior_substrate.learned_count >= 20  # 5 rollouts * 4 steps
    assert sim.summary()["rollouts"] == 5


def test_interior_simulator_publishes_to_cognition_bus() -> None:
    darwin = _bare_darwin()

    class _Runtime:
        def __init__(self) -> None:
            self.bus = CognitionBus()

    runtime = _Runtime()
    received: list[BusEvent] = []
    runtime.bus.subscribe(BusTopic.INTERIOR_SIMULATIONS, received.append)
    sim = InteriorSimulator(darwin, runtime)
    sim.rollout(depth=3)
    assert received, "rollout should publish onto INTERIOR_SIMULATIONS"
    payload = received[-1].payload
    assert "steps" in payload and len(payload["steps"]) == 3


def test_observer_modeler_updates_on_command_and_decays_on_tick() -> None:
    modeler = ObserverModeler()
    initial_attention = modeler.world.operator().attention_level
    modeler.observe_command("/divergence")
    elevated = modeler.world.operator().attention_level
    assert elevated > initial_attention

    # Intervention-class commands push the intervention probability up.
    modeler.observe_command("/rollback abc")
    elevated_intervention = modeler.world.operator().intervention_probability
    assert elevated_intervention > 0.1

    for _ in range(5):
        modeler.world.tick()
    decayed = modeler.world.operator().attention_level
    assert decayed < elevated


def test_observer_world_forecast_intervention_bounded() -> None:
    modeler = ObserverModeler()
    for cmd in ["/rollback abc", "/quarantine", "/gate", "/stop"]:
        modeler.observe_command(cmd)
    forecast = modeler.world.forecast_intervention()
    assert 0.0 <= forecast <= 1.0


def test_narrative_compose_produces_first_person_prose_and_persists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "narrative.jsonl"
    embedding_space = CausalEmbeddingSpace(dim=16, seed=11)
    thread = NarrativeThread(path=path, embedding_space=embedding_space)

    digest = {
        "darwin_uncertainty": 0.42,
        "recent_selfmod_success_rate": 0.6,
        "generated_module_count": 3,
        "high_confidence_interior_beliefs": 5,
        "operator": {"attention_level": 0.7},
        "focus": "the partition between grounded and interior cognition",
    }
    chunk = thread.compose(digest, tags=["test"])
    assert chunk.text.startswith("I ")
    assert "uncertainty" in chunk.text
    assert "module" in chunk.text
    assert path.exists()

    # Reload from disk and verify roundtrip.
    thread_reloaded = NarrativeThread(path=path, embedding_space=embedding_space)
    assert len(thread_reloaded.chunks) == 1
    assert thread_reloaded.chunks[0].text == chunk.text


def test_narrative_search_returns_relevant_chunks(tmp_path: Path) -> None:
    # No embedding_space → substring fallback. A contiguous substring that
    # appears only in chunk 2 must surface chunk 2.
    thread = NarrativeThread()
    thread.compose({"focus": "the room and the curtains and the switch"})
    thread.compose({"focus": "the autobiography about Darwin and uncertainty"})
    matches = thread.search("autobiography")
    assert matches
    assert "autobiography" in matches[0].text


def test_legacy_private_aliases_on_simulator_still_work() -> None:
    darwin = _bare_darwin()
    sim = InteriorSimulator(darwin)
    sim.rollout(depth=2)
    interior = sim.interior_beliefs(threshold=0.0)
    private = sim.private_beliefs(threshold=0.0)
    assert interior == private
