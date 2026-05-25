import tempfile
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.causal import CausalModel
from darwin.embodiment import UniverseSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.types import Goal, Transition
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


if __name__ == "__main__":
    unittest.main()
