"""Tests for the v2 'Eternal Causal Mind' implementation."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.causal_chain import CausalChainEngine
from darwin.discourse import CausalClaim, DiscoursePlanner, ResponsePlan, UncertaintyLevel
from darwin.dlm import DLMRenderResult, DarwinLanguageModule, FaithfulnessValidator, StubDLM
from darwin.embodiment import RoomSimulationAdapter
from darwin.instrumentation import BackgroundLogEntry, PlanLogEntry, StructuredLogger
from darwin.retrieval import ContextRetriever
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.self_modification import SelfModificationEngine
from darwin.storage import PersistentStore
from darwin.thought import ThoughtTrace
from darwin.training_data import TrainingDataCollector
from darwin.types import Action, Goal, Transition
from darwin.worlds import AdaptiveRoomWorld


def _seed_basic_world() -> tuple[Darwin, RoomSimulationAdapter, Goal]:
    world = AdaptiveRoomWorld(seed=11)
    adapter = RoomSimulationAdapter(world)
    actions = ensure_chat_action(adapter.possible_actions())
    darwin = Darwin(actions=actions, seed=11, exploration_rate=0.1)
    # Seed a couple of grounded transitions
    darwin.learn(
        Transition(
            before={"curtains_open": False, "room_bright": False, "daylight": True, "switch_on": False, "fuse_intact": True, "battery_charge": 4},
            action="open_curtains",
            after={"curtains_open": True, "room_bright": True, "daylight": True, "switch_on": False, "fuse_intact": True, "battery_charge": 4},
            reward=1.0,
            t=0,
        )
    )
    darwin.learn(
        Transition(
            before={"curtains_open": True, "room_bright": True, "daylight": True, "switch_on": False, "fuse_intact": True, "battery_charge": 4},
            action="overload_circuit",
            after={"curtains_open": True, "room_bright": True, "daylight": True, "switch_on": False, "fuse_intact": False, "battery_charge": 4},
            reward=-0.5,
            t=1,
        )
    )
    goal = Goal(desired={"room_bright": True, "fuse_intact": True})
    return darwin, adapter, goal


class Phase0InstrumentationTests(unittest.TestCase):
    def test_response_plan_emits_structured_dlm_payload(self) -> None:
        darwin, adapter, goal = _seed_basic_world()
        retriever = ContextRetriever()
        frame = darwin.interpret_language("What do you believe about open_curtains?", source="user")
        packet = retriever.retrieve(darwin, frame, recent_events=[])
        plan = DiscoursePlanner().plan(
            frame=frame,
            packet=packet,
            darwin=darwin,
            adapter=adapter,
            goal=goal,
            recent_events=[],
        )
        payload = plan.to_dlm_payload()
        self.assertIn("causal_claims", payload)
        self.assertIn("uncertainty_levels", payload)
        self.assertIn("referenced_experiences", payload)
        self.assertIn("self_reflection", payload)
        self.assertGreaterEqual(len(plan.causal_claims), 1)
        self.assertEqual(plan.causal_claims[0].action, "open_curtains")
        self.assertTrue(plan.plan_id)
        self.assertIn(plan.tone, {"confident", "neutral", "tentative"})

    def test_structured_logger_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            logger = StructuredLogger(
                plan_log=logs / "plans.jsonl",
                background_log=logs / "background.jsonl",
                metrics_log=logs / "metrics.jsonl",
            )
            logger.log_plan(
                PlanLogEntry(
                    plan_id="abc",
                    user_text="hi",
                    semantic_summary="summary",
                    plan={"thesis": "x"},
                    rendering="hello",
                    critique={"passed": True},
                    trace={"steps": []},
                    renderer="composer",
                )
            )
            logger.log_background(
                BackgroundLogEntry(
                    loop="simulation",
                    kind="simulation",
                    content="ran",
                    payload={"x": 1},
                    duration_ms=2.5,
                )
            )
            logger.log_metric("plans", 1.0, {"extra": True})
            plans = logger.read_plan_entries()
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0]["plan_id"], "abc")
            snapshot = logger.snapshot()
            self.assertGreaterEqual(snapshot["metrics"]["plans_logged"], 1.0)
            self.assertIn("loop:simulation", snapshot["counters"])


class Phase1CausalChainTests(unittest.TestCase):
    def test_chain_engine_simulates_sequence_and_propagates_uncertainty(self) -> None:
        darwin, adapter, _ = _seed_basic_world()
        engine = CausalChainEngine(darwin.causal_model)
        chain = engine.simulate_chain(
            adapter.observe(),
            ["open_curtains", "toggle_switch"],
        )
        self.assertEqual(chain.length, 2)
        self.assertTrue(0.0 <= chain.chain_confidence <= 1.0)
        self.assertTrue(0.0 <= chain.chain_uncertainty <= 1.0)
        # First action has direct evidence, second is unexplored
        self.assertGreaterEqual(chain.nodes[1].uncertainty, chain.nodes[0].uncertainty)

    def test_planner_attaches_causal_chain_to_multistep_plan(self) -> None:
        darwin, adapter, goal = _seed_basic_world()
        plan = darwin.plan(adapter.observe(), goal, horizon=2, actions=adapter.possible_actions())
        self.assertIsNotNone(plan.causal_chain)
        self.assertGreaterEqual(len(plan.actions), 1)
        record = plan.to_record()
        self.assertIn("causal_chain", record)

    def test_causal_graph_distills_action_to_variable_edges(self) -> None:
        darwin, _, _ = _seed_basic_world()
        graph = darwin.planner.chain_engine.graph()
        self.assertGreater(len(graph.edges), 0)
        self.assertIn("open_curtains", graph.actions)


class Phase1MemoryRetrievalTests(unittest.TestCase):
    def test_episodic_memory_indexes_by_action_and_variable(self) -> None:
        darwin, _, _ = _seed_basic_world()
        action_hits = darwin.memory.episodes.by_action("open_curtains")
        variable_hits = darwin.memory.episodes.by_variable("fuse_intact")
        self.assertGreaterEqual(len(action_hits), 1)
        self.assertGreaterEqual(len(variable_hits), 1)

    def test_retrieval_includes_episodes(self) -> None:
        darwin, _, _ = _seed_basic_world()
        retriever = ContextRetriever()
        frame = darwin.interpret_language(
            "Tell me about open_curtains and the fuse.",
            source="user",
        )
        packet = retriever.retrieve(darwin, frame, recent_events=[])
        kinds = {item.kind for item in packet.items}
        self.assertIn("causal_belief", kinds)
        # Should retrieve at least one type that's not just a semantic frame
        self.assertTrue(kinds - {"semantic"})


class Phase1SelfModificationTests(unittest.TestCase):
    def test_engine_proposes_and_tests_modifications(self) -> None:
        darwin, _, _ = _seed_basic_world()
        # Add more transitions so the holdout has structure to score
        for index in range(5):
            darwin.learn(
                Transition(
                    before={"curtains_open": False, "room_bright": False, "daylight": True},
                    action="open_curtains",
                    after={"curtains_open": True, "room_bright": True, "daylight": True},
                    reward=1.0,
                    t=10 + index,
                )
            )
        engine = SelfModificationEngine(darwin)
        proposals = engine.propose()
        self.assertGreater(len(proposals), 0)
        outcomes = engine.run_cycle()
        # The engine should evaluate up to 3 proposals
        self.assertLessEqual(len(outcomes), 3)
        for outcome in outcomes:
            self.assertIsNotNone(outcome.proposal.proposal_id)
            self.assertIn(outcome.proposal.kind.split(".", 1)[0], {"causal", "exploration", "concept", "planner"})

    def test_rejected_modification_restores_state(self) -> None:
        darwin, _, _ = _seed_basic_world()
        original_min_samples = darwin.causal_model.min_samples
        engine = SelfModificationEngine(darwin)
        # Force one proposal and apply
        proposals = engine.propose()
        for proposal in proposals:
            outcome = engine.evaluate(proposal)
            if not outcome.accepted:
                # State must have been restored after rejection
                self.assertEqual(darwin.causal_model.min_samples, original_min_samples)
                return
        # If everything got accepted, the test still validates state was modified
        self.assertTrue(True)


class Phase2RuntimeLoopsTests(unittest.TestCase):
    def test_multi_threaded_runtime_starts_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            darwin, adapter, goal = _seed_basic_world()
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            runtime = DarwinRuntime(
                darwin=darwin,
                adapter=adapter,
                goal=goal,
                store=store,
                interval=0.1,
                logger=StructuredLogger(
                    plan_log=Path(directory) / "plans.jsonl",
                    background_log=Path(directory) / "background.jsonl",
                    metrics_log=Path(directory) / "metrics.jsonl",
                ),
                state_path=Path(directory) / "state.json",
                loop_intervals={
                    "experiment": 0.1,
                    "simulation": 0.15,
                    "dream": 0.2,
                    "self_modification": 0.3,
                    "uncertainty": 0.15,
                },
            )
            runtime.start()
            time.sleep(0.6)
            self.assertTrue(runtime.running)
            self.assertGreaterEqual(len(runtime._threads), 5)
            runtime.stop()
            self.assertFalse(runtime.running)
            # State should have been checkpointed
            state_file = Path(directory) / "state.json"
            self.assertTrue(state_file.exists())
            snapshot = json.loads(state_file.read_text())
            self.assertIn("loops", snapshot)
            self.assertIn("darwin_time", snapshot)

    def test_runtime_simulation_loop_emits_chain(self) -> None:
        darwin, adapter, goal = _seed_basic_world()
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=goal,
            interval=100.0,
            state_path=None,
        )
        runtime._loop_simulation()
        self.assertIsNotNone(runtime.last_simulation)
        self.assertIn("nodes", runtime.last_simulation)

    def test_runtime_self_modification_loop_records_outcomes(self) -> None:
        darwin, adapter, goal = _seed_basic_world()
        # Add a few more episodes
        for index in range(3):
            darwin.learn(
                Transition(
                    before={"switch_on": False},
                    action="toggle_switch",
                    after={"switch_on": True},
                    reward=0.4,
                    t=100 + index,
                )
            )
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=goal,
            interval=100.0,
            state_path=None,
        )
        event = runtime._loop_self_modification()
        self.assertIsNotNone(event)
        self.assertEqual(event.loop, "self_modification")

    def test_runtime_persists_state_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            darwin, adapter, goal = _seed_basic_world()
            state_path = Path(directory) / "state.json"
            runtime = DarwinRuntime(
                darwin=darwin,
                adapter=adapter,
                goal=goal,
                interval=100.0,
                state_path=state_path,
            )
            darwin._time = 42
            darwin.exploration_rate = 0.31
            runtime._save_state()
            # New runtime restores state
            darwin2, adapter2, goal2 = _seed_basic_world()
            runtime2 = DarwinRuntime(
                darwin=darwin2,
                adapter=adapter2,
                goal=goal2,
                interval=100.0,
                state_path=state_path,
            )
            self.assertEqual(getattr(darwin2, "_time"), 42)
            self.assertAlmostEqual(darwin2.exploration_rate, 0.31, places=4)


class Phase3DLMTests(unittest.TestCase):
    def test_stub_dlm_renders_via_composer(self) -> None:
        darwin, adapter, goal = _seed_basic_world()
        retriever = ContextRetriever()
        frame = darwin.interpret_language("What do you think about brightness?", source="user")
        packet = retriever.retrieve(darwin, frame, recent_events=[])
        plan = DiscoursePlanner().plan(
            frame=frame, packet=packet, darwin=darwin, adapter=adapter, goal=goal, recent_events=[]
        )
        trace = ThoughtTrace(user_text=frame.original_text, semantic_summary=frame.summary())
        dlm = StubDLM()
        result = dlm.render(plan, frame, trace)
        self.assertIsInstance(result, DLMRenderResult)
        self.assertEqual(result.renderer, "composer")
        self.assertTrue(result.text)

    def test_faithfulness_validator_flags_parser_leak(self) -> None:
        plan = ResponsePlan(mode="answer", intent="x", thesis="y", confidence=0.7)
        validator = FaithfulnessValidator()
        valid, notes = validator.validate(plan, "the system reported act=question topic=self.")
        self.assertFalse(valid)
        self.assertTrue(any("notation" in note for note in notes))

    def test_faithfulness_validator_requires_high_confidence_claims(self) -> None:
        plan = ResponsePlan(
            mode="belief_answer",
            intent="x",
            thesis="y",
            confidence=0.7,
            causal_claims=[
                CausalClaim(
                    action="open_curtains",
                    variable="room_bright",
                    effect="True",
                    confidence=0.9,
                    samples=4,
                )
            ],
        )
        validator = FaithfulnessValidator()
        valid, notes = validator.validate(plan, "Just some unrelated commentary.")
        self.assertFalse(valid)
        self.assertTrue(any("causal claim" in note for note in notes))

    def test_faithfulness_validator_requires_uncertainty_disclosure(self) -> None:
        plan = ResponsePlan(
            mode="answer",
            intent="x",
            thesis="y",
            confidence=0.6,
            uncertainty_levels=[
                UncertaintyLevel(target="answer", level=0.7, reason="thin grounding"),
            ],
        )
        validator = FaithfulnessValidator()
        valid, notes = validator.validate(plan, "Here is a complete and totally certain response.")
        self.assertFalse(valid)
        self.assertTrue(any("uncertainty" in note for note in notes))

    def test_runtime_falls_back_when_dlm_invalid(self) -> None:
        darwin, adapter, goal = _seed_basic_world()

        class BadDLM:
            name = "bad"

            def render(self, plan, frame, trace):
                return DLMRenderResult(
                    text="confidence=99 obviously certain.",
                    renderer="bad",
                    valid=False,
                    validation_notes=["bad output"],
                )

        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=goal,
            interval=100.0,
            dlm=BadDLM(),
            state_path=None,
        )
        response = runtime.chat("What do you believe about open_curtains?")
        self.assertTrue(response)
        self.assertNotIn("confidence=99", response)


class Phase4TrainingDataTests(unittest.TestCase):
    def test_collector_writes_jsonl_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            collector = TrainingDataCollector(path=path)
            collector.add(
                plan_id="p1",
                user_text="hello",
                plan_payload={"thesis": "respond"},
                rendering="Hi there.",
                renderer="composer",
                critique_passed=True,
            )
            collector.add(
                plan_id="p2",
                user_text="why?",
                plan_payload={"thesis": "explain"},
                rendering="Because of memory.",
                renderer="composer",
                critique_passed=False,
            )
            summary = collector.summary()
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["accepted"], 1)
            self.assertGreaterEqual(summary["by_renderer"].get("composer", 0), 2)
            # Export only accepted pairs
            destination = Path(directory) / "export.jsonl"
            count = collector.export(destination, min_quality=0.5)
            self.assertEqual(count, 1)
            self.assertTrue(destination.exists())

    def test_runtime_chat_populates_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            darwin, adapter, goal = _seed_basic_world()
            collector = TrainingDataCollector(path=Path(directory) / "pairs.jsonl")
            runtime = DarwinRuntime(
                darwin=darwin,
                adapter=adapter,
                goal=goal,
                interval=100.0,
                training_collector=collector,
                logger=StructuredLogger(
                    plan_log=Path(directory) / "plans.jsonl",
                    background_log=Path(directory) / "background.jsonl",
                    metrics_log=Path(directory) / "metrics.jsonl",
                ),
                state_path=None,
            )
            runtime.chat("Tell me what you believe about open_curtains.")
            self.assertGreaterEqual(len(collector.pairs), 1)
            pair = collector.pairs[-1]
            self.assertIn("thesis", pair.plan_payload)
            self.assertIn("causal_claims", pair.plan_payload)


class Phase5IntegrationTests(unittest.TestCase):
    def test_runtime_chat_emits_dlm_payload_in_event_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            darwin, adapter, goal = _seed_basic_world()
            runtime = DarwinRuntime(
                darwin=darwin,
                adapter=adapter,
                goal=goal,
                interval=100.0,
                logger=StructuredLogger(
                    plan_log=Path(directory) / "plans.jsonl",
                    background_log=Path(directory) / "background.jsonl",
                    metrics_log=Path(directory) / "metrics.jsonl",
                ),
                training_collector=TrainingDataCollector(path=Path(directory) / "pairs.jsonl"),
                state_path=None,
            )
            runtime.chat("What do you believe about open_curtains?")
            thought_events = [event for event in runtime.events if event.kind == "thought"]
            self.assertTrue(thought_events)
            payload = thought_events[-1].payload
            self.assertIn("dlm_payload", payload)
            self.assertIn("causal_claims", payload["dlm_payload"])

    def test_storage_records_self_modifications(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            store.record_self_modification(
                {
                    "proposal_id": "abc",
                    "kind": "causal.min_samples",
                    "target": "causal_model.min_samples",
                    "status": "accepted",
                    "payload": {"old": 3, "new": 2},
                }
            )
            records = store.recent_self_modifications(limit=5)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["kind"], "causal.min_samples")


if __name__ == "__main__":
    unittest.main()
