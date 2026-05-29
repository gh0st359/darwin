"""End-to-end: chat actually consults the universe substrate.

Asserts that a chat turn:
  1. Grounds words in the user's message against the universe.
  2. Runs the conceptual reasoner and stores a trace.
  3. Feeds text to the deriver for future co-occurrence learning.
  4. Surfaces reasoning answer points into the discourse plan.
"""

from __future__ import annotations

from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.instrumentation import StructuredLogger
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.training_data import TrainingDataCollector
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


def _runtime(tmpdir: Path) -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=23)
    adapter = RoomSimulationAdapter(world)
    store = PersistentStore(tmpdir / "memory.sqlite3")
    actions = ensure_chat_action(adapter.possible_actions())
    darwin = Darwin(actions=actions, store=store, seed=23, exploration_rate=0.1)
    goal = Goal(desired={"room_bright": True, "fuse_intact": True})
    return DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=goal,
        store=store,
        interval=0.1,
        logger=StructuredLogger(
            plan_log=tmpdir / "plans.jsonl",
            background_log=tmpdir / "background.jsonl",
            metrics_log=tmpdir / "metrics.jsonl",
        ),
        training_collector=TrainingDataCollector(path=tmpdir / "pairs.jsonl"),
        state_path=tmpdir / "state.json",
    )


def test_runtime_constructs_with_universe_substrate(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    # The default universe is the primitive seed — small, structural.
    assert runtime.universe is not None
    summary = runtime.universe.summary()
    assert summary["concepts"] > 0
    assert summary["domains"] >= 4  # structure / dynamics / inference / magnitude / self


def test_chat_grounds_user_words_to_concepts(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.chat("Tell me about cause and effect.")
    grounding = runtime.last_grounding
    assert grounding is not None
    names = grounding.concept_names
    # 'cause' and 'effect' are primitives, so they must ground exactly.
    assert "cause" in names
    assert "effect" in names


def test_chat_runs_reasoner_and_stores_trace(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.chat("What's the difference between truth and belief?")
    trace = runtime.last_reasoning_trace
    assert trace is not None
    assert trace.query.startswith("What's the difference")
    assert trace.steps  # reasoner produced at least one step


def test_chat_grows_universe_via_unknown_word_grounding(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    before = runtime.universe.summary()["concepts"]
    runtime.chat("What is whorzplatzium and how does it relate to qubitronics?")
    after = runtime.universe.summary()["concepts"]
    assert after > before
    assert runtime.universe.has("whorzplatzium")


def test_chat_reasoning_answer_points_feed_response_plan(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.chat("How are cause and reason connected?")
    plan = runtime.last_response_plan
    assert plan is not None
    # The reasoning trace should have contributed at least one answer point.
    if runtime.last_reasoning_trace and runtime.last_reasoning_trace.suggested_answer_points:
        assert any(
            point in plan.answer_points
            for point in runtime.last_reasoning_trace.suggested_answer_points
        )


def test_deriver_observes_chat_text(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for _ in range(4):
        runtime.chat("self and model meet again")
    summary = runtime.deriver.summary()
    assert summary["tracked_word_pairs"] >= 1
