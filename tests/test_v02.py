import tempfile
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.experiments import ExperimentEngine
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.types import Goal, Transition
from darwin.worlds import AdaptiveRoomWorld


class DarwinV02Tests(unittest.TestCase):
    def test_persistent_store_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            store = PersistentStore(path)
            transition = Transition(
                before={"room_bright": False},
                action="open_curtains",
                after={"room_bright": True},
                reward=1.0,
                t=0,
            )
            store.record_transition(transition)

            restarted = PersistentStore(path)
            loaded = restarted.load_transitions()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].action, "open_curtains")
            self.assertEqual(loaded[0].after["room_bright"], True)

    def test_hierarchical_concepts_form_affordances(self) -> None:
        darwin = Darwin(actions=AdaptiveRoomWorld(seed=1).possible_actions(), seed=1)
        darwin.learn(
            Transition(
                before={"curtains_open": False, "room_bright": False},
                action="open_curtains",
                after={"curtains_open": True, "room_bright": True},
                reward=1.0,
                t=0,
            )
        )

        names = {concept.name for concept in darwin.memory.concepts.hierarchy(limit=50)}

        self.assertIn("affordance:open_curtains:can_set:room_bright=True", names)
        self.assertIn("strategy:seek:room_bright=True:via:open_curtains", names)
        self.assertIn("meta:reliable_action:open_curtains", names)

    def test_experiment_engine_prefers_underexplored_actions(self) -> None:
        world = AdaptiveRoomWorld(seed=2)
        darwin = Darwin(actions=world.possible_actions(), seed=2)
        engine = ExperimentEngine(darwin.causal_model)

        proposals = engine.propose(world.observe(), world.possible_actions(), limit=3)

        self.assertEqual(len(proposals), 3)
        self.assertGreaterEqual(proposals[0].uncertainty, 0.9)

    def test_runtime_chat_records_conversation_as_experience(self) -> None:
        world = AdaptiveRoomWorld(seed=3)
        adapter = RoomSimulationAdapter(world)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=3)
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"room_bright": True}),
            interval=100.0,
        )

        response = runtime.chat("Darwin, remember that planning matters.")

        self.assertIn("learning priority", response)
        self.assertEqual(len(darwin.memory.episodes), 1)
        self.assertEqual(darwin.memory.episodes.recent(1)[0].action, "chat_with_user")
        self.assertGreaterEqual(len(darwin.semantic_memory.recent()), 2)

    def test_runtime_event_sink_receives_thought_events(self) -> None:
        world = AdaptiveRoomWorld(seed=5)
        adapter = RoomSimulationAdapter(world)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=5)
        events = []
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"room_bright": True}),
            interval=100.0,
            event_sink=events.append,
        )

        event = runtime.cognition_cycle()

        self.assertEqual(events[-1], event)
        self.assertIn(event.kind, {"experiment", "reflection"})

    def test_prediction_failure_priority_can_resolve(self) -> None:
        world = AdaptiveRoomWorld(seed=6)
        darwin = Darwin(actions=world.possible_actions(), seed=6)
        failure_transition = Transition(
            before={"curtains_open": False, "room_bright": False},
            action="open_curtains",
            after={"curtains_open": True, "room_bright": True},
            reward=1.0,
            t=0,
            metadata={"predicted_state": {"curtains_open": False, "room_bright": False}},
        )
        darwin.learn(failure_transition)

        first_priority = darwin.self_report().learning_priority
        darwin.learn(
            Transition(
                before={"curtains_open": False, "room_bright": False},
                action="open_curtains",
                after={"curtains_open": True, "room_bright": True},
                reward=1.0,
                t=1,
                metadata={"predicted_state": {"curtains_open": False, "room_bright": False}},
            )
        )
        second_priority = darwin.self_report().learning_priority

        self.assertIn("open_curtains", first_priority)
        self.assertNotIn("open_curtains:curtains_open", second_priority)

    def test_long_horizon_plan_returns_sequence(self) -> None:
        world = AdaptiveRoomWorld(seed=4)
        darwin = Darwin(actions=world.possible_actions(), seed=4, exploration_rate=0.0)
        goal = Goal(desired={"room_bright": True}, progress_weight=4.0)
        darwin.learn(
            Transition(
                before=world.observe(),
                action="open_curtains",
                after={
                    "switch_on": False,
                    "fuse_intact": True,
                    "curtains_open": True,
                    "daylight": True,
                    "room_bright": True,
                    "battery_charge": 4,
                },
                reward=1.0,
                t=0,
            )
        )

        plan = darwin.plan(world.observe(), goal, horizon=2)

        self.assertGreaterEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].name, "open_curtains")


if __name__ == "__main__":
    unittest.main()
