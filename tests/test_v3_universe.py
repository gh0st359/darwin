import tempfile
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.causal import CausalModel
from darwin.embodiment import UniverseSimulationAdapter
from darwin.experiments import ExperimentEngine
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.self_modification import SelfModificationEngine
from darwin.semantics import SemanticParser
from darwin.storage import PersistentStore
from darwin.types import Action, Goal, Transition
from darwin.worlds import UniverseSimulation


class V3UniverseTests(unittest.TestCase):
    def test_conversation_scope_is_not_reported_as_world_belief_by_default(self) -> None:
        model = CausalModel(min_samples=1)
        model.learn(
            Transition(
                before={"conversation_active": False},
                action="chat_with_user",
                after={"conversation_active": True},
                reward=0.2,
                metadata={"scope": "conversation", "world": "interaction"},
            )
        )

        self.assertEqual(model.beliefs(), [])
        conversation_beliefs = model.beliefs(scope="conversation")
        self.assertEqual(len(conversation_beliefs), 1)
        self.assertEqual(conversation_beliefs[0].action, "chat_with_user")

    def test_legacy_chat_transitions_are_treated_as_conversation_scope(self) -> None:
        model = CausalModel(min_samples=1)
        model.learn(
            Transition(
                before={"conversation_active": False},
                action="chat_with_user",
                after={"conversation_active": True},
                reward=0.2,
            )
        )

        self.assertEqual(model.beliefs(), [])
        self.assertEqual(model.beliefs(scope="conversation")[0].variable, "conversation_active")

    def test_universe_adapter_is_one_environment_with_multiple_causal_facets(self) -> None:
        universe = UniverseSimulation(seed=4)
        adapter = UniverseSimulationAdapter(universe)

        action_names = {action.name for action in adapter.possible_actions()}

        self.assertEqual(adapter.name, "universe")
        self.assertIn("room/open_curtains", action_names)
        self.assertIn("math/add_1", action_names)
        self.assertIn("space/push_a_right", action_names)

        before = adapter.observe()
        after, _reward = adapter.apply(next(action for action in adapter.possible_actions() if action.name == "math/add_1"))

        self.assertEqual(before["math.x"], 0)
        self.assertEqual(after["math.x"], 1)
        self.assertIn("room.room_bright", after)
        self.assertIn("space.a.x", after)

    def test_runtime_tags_universe_experiments_with_scope_world_and_domain(self) -> None:
        universe = UniverseSimulation(seed=5)
        adapter = UniverseSimulationAdapter(universe)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=5)
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"room.room_bright": True, "space.a.y": 0, "math.x": 4}),
            interval=100.0,
            state_path=None,
        )

        event = runtime.cognition_cycle()

        self.assertIn(event.kind, {"experiment", "reflection"})
        transition = darwin.memory.episodes.recent(1)[0]
        self.assertEqual(transition.metadata["scope"], "world")
        self.assertEqual(transition.metadata["world"], "universe")
        self.assertIn(transition.metadata["domain"], {"room", "math", "space", "time"})

    def test_teaching_arithmetic_focuses_the_next_experiment_on_math(self) -> None:
        universe = UniverseSimulation(seed=6)
        adapter = UniverseSimulationAdapter(universe)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=6)
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"math.x": 4}),
            interval=100.0,
            state_path=None,
        )

        runtime.chat("Teach yourself arithmetic: addition changes a number by adding to it.")
        event = runtime.cognition_cycle()

        self.assertEqual(event.kind, "experiment")
        transition = darwin.memory.episodes.recent(1)[0]
        self.assertTrue(transition.action.startswith("math/"), transition.action)
        self.assertEqual(transition.metadata["attention_source"], "semantic_hypothesis")

    def test_storage_records_world_column_and_rehydrates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            store.record_transition(
                Transition(
                    before={"math.x": 0},
                    action="math/add_1",
                    after={"math.x": 1},
                    reward=0.4,
                    t=1,
                    metadata={"scope": "world", "world": "universe", "domain": "math"},
                )
            )

            loaded = store.load_transitions()

            self.assertEqual(loaded[0].metadata["world"], "universe")
            self.assertEqual(loaded[0].metadata["domain"], "math")

    def test_uncertainty_question_is_not_misclassified_as_identity(self) -> None:
        parser = SemanticParser()

        frame = parser.parse("What are you uncertain about?")

        self.assertEqual(frame.speech_act, "question")
        self.assertEqual(frame.topic, "experiments")

    def test_domain_question_about_moving_blocks_has_enough_grounding(self) -> None:
        parser = SemanticParser()

        frame = parser.parse("What have you learned about moving blocks?")

        self.assertEqual(frame.speech_act, "question")
        self.assertEqual(frame.topic, "space")
        self.assertGreaterEqual(frame.confidence, 0.45)

    def test_domain_filtered_experiment_question_does_not_mix_state_facets(self) -> None:
        model = CausalModel(min_samples=1)
        state = {
            "math.x": 3,
            "math.last_operand": 1,
            "space.a.y": 1,
            "space.held": "a",
        }
        model.learn(
            Transition(
                before=state,
                action="space/drop_a",
                after={**state, "space.a.y": 0, "space.held": "none"},
                reward=0.1,
                t=1,
                metadata={"scope": "world", "world": "universe", "domain": "space"},
            )
        )
        model.learn(
            Transition(
                before={**state, "math.x": 3},
                action="math/add_1",
                after={**state, "math.x": 4, "math.last_operand": 1},
                reward=0.1,
                t=2,
                metadata={"scope": "world", "world": "universe", "domain": "math"},
            )
        )

        proposals = ExperimentEngine(model).propose(
            state,
            [Action("space/drop_a", metadata={"domain": "space"})],
            variable_filter=lambda variable: variable.startswith("space."),
        )

        self.assertEqual(len(proposals), 1)
        self.assertIn("space.", proposals[0].question)
        self.assertNotIn("math.", proposals[0].question)
        self.assertTrue(all(key.startswith("space.") for key in proposals[0].predicted_state))

    def test_runtime_keeps_experimenting_after_current_actions_are_low_uncertainty(self) -> None:
        universe = UniverseSimulation(seed=14)
        adapter = UniverseSimulationAdapter(universe)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=14)
        darwin.causal_model.min_samples = 1
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"math.x": 4, "space.a.y": 0, "room.room_bright": True}),
            interval=100.0,
            state_path=None,
        )
        for action in adapter.possible_actions():
            before = adapter.observe()
            after, reward = adapter.apply(action)
            darwin.learn(
                Transition(
                    before=before,
                    action=action.name,
                    after=after,
                    reward=reward,
                    t=runtime._next_time(),
                    metadata=runtime._metadata_for_action(action),
                )
            )
        before_count = len(darwin.memory.episodes)

        event = runtime.cognition_cycle()

        self.assertEqual(event.kind, "experiment")
        self.assertEqual(len(darwin.memory.episodes), before_count + 1)
        self.assertIn("maintenance_experiment", darwin.memory.episodes.recent(1)[0].metadata["mode"])

    def test_natural_block_learning_question_uses_space_causal_beliefs(self) -> None:
        universe = UniverseSimulation(seed=15)
        adapter = UniverseSimulationAdapter(universe)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=15)
        darwin.causal_model.min_samples = 1
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"space.a.y": 0}),
            interval=100.0,
            state_path=None,
        )
        for action_name in ["space/push_a_right", "space/lift_a", "space/drop_a"]:
            action = next(action for action in adapter.possible_actions() if action.name == action_name)
            before = adapter.observe()
            after, reward = adapter.apply(action)
            darwin.learn(
                Transition(
                    before=before,
                    action=action.name,
                    after=after,
                    reward=reward,
                    t=runtime._next_time(),
                    metadata=runtime._metadata_for_action(action),
                )
            )

        runtime.chat("What have you learned about moving blocks?")

        plan = runtime.last_response_plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "belief_answer")
        self.assertTrue(plan.causal_claims)
        self.assertTrue(all(claim.action.startswith("space/") for claim in plan.causal_claims), plan.causal_claims)

    def test_uncertainty_question_uses_domain_local_experiment_predictions(self) -> None:
        universe = UniverseSimulation(seed=17)
        adapter = UniverseSimulationAdapter(universe)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=17)
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"math.x": 4, "space.a.y": 0, "room.room_bright": True}),
            interval=100.0,
            state_path=None,
        )
        for _ in range(35):
            runtime.cognition_cycle()

        runtime.chat("What are you uncertain about?")

        plan = runtime.last_response_plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "experiment")
        prediction_points = [point for point in plan.answer_points if point.startswith("prediction:")]
        self.assertTrue(prediction_points)
        action = plan.answer_points[0].split(":", 1)[0].removeprefix("test ").strip()
        domain = action.split("/", 1)[0]
        self.assertIn(f"{domain}.", prediction_points[0])
        for other in {"room", "math", "space", "time"} - {domain}:
            self.assertNotIn(f"{other}.", prediction_points[0])

    def test_self_modification_does_not_lower_min_samples_below_curiosity_floor(self) -> None:
        universe = UniverseSimulation(seed=16)
        adapter = UniverseSimulationAdapter(universe)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=16)
        darwin.causal_model.min_samples = 3

        proposals = SelfModificationEngine(darwin).propose()

        self.assertFalse(
            any(
                proposal.kind == "causal.min_samples" and proposal.payload.get("new", 0) < 3
                for proposal in proposals
            )
        )


if __name__ == "__main__":
    unittest.main()
