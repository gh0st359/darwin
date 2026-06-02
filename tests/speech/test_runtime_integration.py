"""Tests that V-Speech wires into DarwinRuntime correctly."""

from __future__ import annotations

import os
from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


def _runtime(tmp_path: Path) -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=11)
    adapter = RoomSimulationAdapter(world)
    store = PersistentStore(tmp_path / "memory.sqlite3")
    actions = ensure_chat_action(adapter.possible_actions())
    darwin = Darwin(actions=actions, store=store, seed=11, exploration_rate=0.1)
    goal = Goal(desired={"room_bright": True})
    return DarwinRuntime(
        darwin=darwin, adapter=adapter, goal=goal, store=store, interval=0.1,
    )


def test_runtime_constructs_speech_substrate(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert runtime.speech_pipeline is not None
    assert runtime.speech_dlm is not None
    assert runtime.speech_lexicon is not None


def test_default_dlm_is_speech_unless_opted_out(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    # Default DARWIN_USE_SPEECH is unset (or "1"), so the speech DLM
    # should be active.
    assert runtime.dlm.name == "speech"


def test_chat_output_passes_leak_gate(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    reply = runtime.chat("hello")
    # No structured-internals leak in chat output.
    assert "{" not in reply
    assert "}" not in reply
    assert '"thesis":' not in reply
    assert "BusTopic." not in reply
    assert "[event " not in reply


def test_chat_multi_turn_remains_leak_free(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for utterance in [
        "hello",
        "what do you know about open_curtains?",
        "tell me more",
        "A neuron is a cell.",
        "Is a neuron a cell?",
    ]:
        reply = runtime.chat(utterance)
        assert "{" not in reply and "}" not in reply, f"leak in reply to {utterance!r}: {reply!r}"
        assert '"thesis"' not in reply
        assert "BusTopic." not in reply


def test_speech_lexicon_persists_across_runtime_construction(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.speech_lexicon.register(
        surface="quark", category="N", concept="quark", pos="NN",
    )
    runtime.speech_lexicon.save(runtime.speech_lexicon_path)
    runtime2 = _runtime(tmp_path)
    entries = runtime2.speech_lexicon.lookup("quark")
    assert entries


def test_dlm_opt_out_via_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DARWIN_USE_SPEECH", "0")
    runtime = _runtime(tmp_path)
    # When opted out, default DLM is the StubDLM (composer baseline).
    assert runtime.dlm.name != "speech"
