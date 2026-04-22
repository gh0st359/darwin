import unittest

from darwin.agent import Darwin
from darwin.types import Goal, Transition
from darwin.worlds import AdaptiveRoomWorld


class DarwinAgentTests(unittest.TestCase):
    def test_planner_chooses_learned_brightening_action(self) -> None:
        world = AdaptiveRoomWorld(seed=1)
        darwin = Darwin(actions=world.possible_actions(), seed=1, exploration_rate=0.0)
        goal = Goal(desired={"room_bright": True}, progress_weight=4.0)

        darwin.learn(
            Transition(
                before={
                    "switch_on": False,
                    "fuse_intact": True,
                    "curtains_open": False,
                    "daylight": True,
                    "room_bright": False,
                    "battery_charge": 4,
                },
                action="open_curtains",
                after={
                    "switch_on": False,
                    "fuse_intact": True,
                    "curtains_open": True,
                    "daylight": True,
                    "room_bright": True,
                    "battery_charge": 4,
                },
                reward=0.97,
            )
        )

        candidate = darwin.decide(world.observe(), goal)

        self.assertEqual(candidate.action.name, "open_curtains")
        self.assertTrue(candidate.predicted.state["room_bright"])

    def test_agent_records_experience(self) -> None:
        world = AdaptiveRoomWorld(seed=2)
        darwin = Darwin(actions=world.possible_actions(), seed=2, exploration_rate=1.0)
        goal = Goal(desired={"room_bright": True})

        transition = darwin.step(world, goal)

        self.assertEqual(len(darwin.memory.episodes), 1)
        self.assertEqual(transition.after, world.observe())
        self.assertGreaterEqual(len(darwin.memory.concepts.salient()), 1)


if __name__ == "__main__":
    unittest.main()

