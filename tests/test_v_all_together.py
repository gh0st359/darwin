"""V-All-Together: full-stack integration test.

Exercises every V-* substrate in a single session:

1. Ingest a synthetic corpus through the V-Ingest pipeline.
2. Mesh fires for ingested concepts.
3. Forward chainer derives transitive closures.
4. Calculator faculty solves an arithmetic word problem (via Mind back-compat).
5. Capability probe runs and produces a scorecard.
6. Chat reply is leak-free.

This is the gate proving the V-Neural / V-Mind landing is coherent end-to-end.
"""

from __future__ import annotations

import re

from darwin.agent import Darwin
from darwin.bench import build_capability_suite
from darwin.bench.framework import BenchmarkRunner
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


_CORPUS = [
    "A dog is a mammal. A mammal is an animal. A dog has fur.",
    "A cat is a mammal. A cat has whiskers.",
    "Photosynthesis is a process. Plants use photosynthesis.",
    "Rain causes flooding. Flooding causes damage.",
    "Water is a liquid. Liquid is a state of matter.",
    "Electrons orbit nuclei. An atom contains a nucleus.",
    "Carbon is an element. An element is a substance.",
    "A bird is an animal. A bird can fly.",
    "Iron is a metal. Metal conducts electricity.",
    "A cell is part of a tissue. A tissue is part of an organ.",
]


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


def test_v_all_together() -> None:
    runtime = _runtime()
    # All substrates must be wired.
    assert runtime.cortical_mesh is not None, "V-Mesh missing"
    assert runtime.speech_pipeline is not None, "V-Speech missing"
    assert runtime.ingest_pipeline is not None, "V-Ingest missing"
    assert runtime.forward_chainer is not None, "V-Reason missing"
    assert runtime.agent_registry is not None, "V-Agents missing"
    assert runtime.feature_flags is not None, "V-Scale missing"

    # 1. Ingest.
    total_facts = 0
    for doc in _CORPUS:
        stats = runtime.ingest_pipeline.ingest_text(doc)
        total_facts = stats.facts_added
    assert total_facts >= 5, (
        f"expected at least 5 facts ingested across the corpus, got {total_facts}"
    )

    # 2. Mesh fires for an ingested concept.
    runtime.cortical_mesh.propagate(seed_cells=["dog"], steps=2)
    assert runtime.cortical_mesh.has("dog")

    # 3. Forward chainer derives transitive closures.
    report = runtime.forward_chainer.fixpoint_step(budget=64)
    assert report.cycles_taken >= 1

    # 4. MathAgent solves an arithmetic word problem.
    sol = runtime.agent_registry.math.solve("If a=3 and b=4, what is a+b?")
    assert sol.succeeded, f"math agent failed: {sol.notes}"
    assert sol.answer == "7"
    assert "{" not in sol.answer
    assert "}" not in sol.answer

    # 5. Capability probe runs (non-fixture, non-memorisable).
    suite = build_capability_suite(seed=7)
    card = BenchmarkRunner(suite).run(runtime, label="v-all-together")
    assert len(card.results) >= 1
    assert "capability" in card.per_category

    # 6. Chat reply is leak-free.
    reply = runtime.chat("Tell me about animals.")
    assert reply, "chat returned empty"
    # No JSON delimiters.
    assert "{" not in reply
    assert "}" not in reply
    # No structured key-value pairs.
    assert not re.search(r'"\w+":\s', reply)
    # No bracketed event lines.
    assert not re.search(r"^\s*\[event ", reply, re.MULTILINE)


def test_v_all_together_handles_repeated_chat() -> None:
    runtime = _runtime()
    for msg in ["hi", "tell me something", "what do you know about water?"]:
        reply = runtime.chat(msg)
        assert reply
        assert "{" not in reply
        assert "}" not in reply
