"""runtime.mind is wired and overrides _respond on confident intents."""

from __future__ import annotations

import re

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.faculties import Mind
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


def _runtime() -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=42)
    adapter = RoomSimulationAdapter(world)
    darwin = Darwin(
        actions=ensure_chat_action(adapter.possible_actions()),
        seed=42, exploration_rate=0.0,
    )
    return DarwinRuntime(
        darwin=darwin, adapter=adapter,
        goal=Goal(desired={"room_bright": True}),
        interval=100.0,
    )


def test_runtime_has_mind_and_agent_registry_alias():
    runtime = _runtime()
    assert runtime.mind is not None
    assert isinstance(runtime.mind, Mind)
    # Back-compat alias used by autonomy / executor.
    assert runtime.agent_registry is runtime.mind


def test_agent_registry_legacy_math_access_still_works():
    runtime = _runtime()
    sol = runtime.agent_registry.math.solve("If a=3 and b=4, what is a+b?")
    assert sol.succeeded is True
    assert sol.answer == "7"


def test_chat_path_contains_no_faculty_category_labels():
    runtime = _runtime()
    prompts = [
        "Tell me about animals.",
        "What is 2 + 2?",
        "How would you plan a trip to the store?",
        "Can you write a function that adds two numbers?",
        "What relates a dog to a mammal?",
    ]
    leak_patterns = [
        r"\bcalculator\b", r"\bcoder\b", r"\bplanner\b", r"\bresearcher\b",
        r"\bscientist\b", r"\bconversationalist\b",
        r"\bcode agent\b", r"\bmath agent\b", r"\bscience agent\b",
        r"\bplanning agent\b", r"\bresearch agent\b", r"\bdialogue agent\b",
        r"\bintent kind\b", r"\bfaculty\b",
    ]
    for prompt in prompts:
        reply = runtime.chat(prompt)
        assert reply, f"empty reply to {prompt!r}"
        for pattern in leak_patterns:
            assert not re.search(pattern, reply, re.IGNORECASE), (
                f"leak {pattern!r} in reply to {prompt!r}: {reply[:200]!r}"
            )


def test_mind_override_does_not_leak_structured_tokens():
    runtime = _runtime()
    reply = runtime.chat("What is 5 + 7?")
    assert reply
    # No JSON / structured leakage.
    assert "{" not in reply
    assert "}" not in reply
