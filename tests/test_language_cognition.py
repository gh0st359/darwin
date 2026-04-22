import io
import unittest

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.streaming import StreamingSpeaker
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


class LanguageCognitionTests(unittest.TestCase):
    def make_runtime(self) -> DarwinRuntime:
        world = AdaptiveRoomWorld(seed=11)
        adapter = RoomSimulationAdapter(world)
        darwin = Darwin(actions=ensure_chat_action(adapter.possible_actions()), seed=11)
        return DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={"room_bright": True}),
            interval=100.0,
        )

    def test_response_uses_retrieved_semantic_memory_without_parser_speak(self) -> None:
        runtime = self.make_runtime()
        runtime.chat("Darwin, learn this: regurgitation means repeating without grounded understanding.")

        response = runtime.chat("Why is repeating without grounding a problem?")

        self.assertIn("ground", response.lower())
        self.assertIn("meaning", response.lower())
        self.assertNotIn("act=", response)
        self.assertNotIn("topic=", response)
        self.assertIsNotNone(runtime.last_thought_trace)
        self.assertIsNotNone(runtime.last_retrieval)
        self.assertTrue(runtime.last_retrieval.items)

    def test_thought_trace_and_critic_are_created_before_response(self) -> None:
        runtime = self.make_runtime()

        response = runtime.chat("What are you thinking right now?")

        self.assertTrue(response)
        self.assertIsNotNone(runtime.last_thought_trace)
        self.assertIsNotNone(runtime.last_response_plan)
        self.assertIsNotNone(runtime.last_critique)
        self.assertGreaterEqual(len(runtime.last_thought_trace.steps), 3)
        self.assertNotIn("groundings=", response)
        self.assertNotIn("semantic:", response)

    def test_streaming_speaker_can_emit_incremental_text(self) -> None:
        output = io.StringIO()
        speaker = StreamingSpeaker(enabled=True, delay=0.0)

        speaker.write("Darwin speaks from a completed thought.", stream=output)

        self.assertEqual(output.getvalue(), "Darwin speaks from a completed thought.\n")


if __name__ == "__main__":
    unittest.main()
