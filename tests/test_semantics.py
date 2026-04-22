import tempfile
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.semantics import SemanticParser
from darwin.storage import PersistentStore
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


class SemanticLanguageTests(unittest.TestCase):
    def test_parser_extracts_goal_values_and_groundings(self) -> None:
        parser = SemanticParser()

        frame = parser.parse(
            "I need Darwin to understand natural language and I don't want an LLM.",
            source="user",
        )

        self.assertEqual(frame.speech_act, "goal")
        self.assertEqual(frame.goals["language_understanding"], "increase")
        self.assertEqual(frame.goals["llm_dependency"], False)
        self.assertIn("rejection", frame.values)
        self.assertTrue(any(item.name == "language_understanding" for item in frame.groundings))

    def test_parser_extracts_causal_hypothesis(self) -> None:
        parser = SemanticParser()

        frame = parser.parse("If the fuse is broken then the room will not become bright.")

        self.assertEqual(frame.speech_act, "hypothesis")
        self.assertGreaterEqual(len(frame.hypotheses), 1)
        self.assertEqual(frame.hypotheses[0].relation, "implies")

    def test_runtime_chat_stores_user_and_self_semantics(self) -> None:
        world = AdaptiveRoomWorld(seed=8)
        adapter = RoomSimulationAdapter(world)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=8)
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"room_bright": True}),
            interval=100.0,
        )

        response = runtime.chat("Darwin, learn this: natural language should be grounded in meaning.")

        self.assertIn("recorded", response.lower())
        recent = darwin.semantic_memory.recent(limit=2)
        self.assertEqual(recent[0].source, "user")
        self.assertEqual(recent[1].source, "darwin")
        self.assertIn("language_understanding", recent[0].goals)

    def test_semantic_memory_persists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            world = AdaptiveRoomWorld(seed=9)
            store = PersistentStore(path)
            darwin = Darwin.from_store(
                actions=ensure_chat_action(world.possible_actions()),
                store=store,
                seed=9,
            )
            darwin.interpret_language("Remember that cause and effect matters.", source="user")

            restarted = Darwin.from_store(
                actions=ensure_chat_action(world.possible_actions()),
                store=PersistentStore(path),
                seed=9,
            )

            self.assertGreaterEqual(len(restarted.semantic_memory.recent()), 1)
            self.assertEqual(restarted.semantic_memory.recent(1)[0].topic, "causality")


if __name__ == "__main__":
    unittest.main()

