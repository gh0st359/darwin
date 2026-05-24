"""End-to-end tests for the daemon brain + connect client."""

from __future__ import annotations

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.server import DarwinClient, DarwinDaemon, PortInUseError
from darwin.storage import PersistentStore
from darwin.training_data import TrainingDataCollector
from darwin.instrumentation import StructuredLogger
from darwin.types import Goal, Transition
from darwin.worlds import AdaptiveRoomWorld


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _seed_darwin(tmpdir: Path) -> tuple[Darwin, RoomSimulationAdapter, Goal]:
    world = AdaptiveRoomWorld(seed=21)
    adapter = RoomSimulationAdapter(world)
    actions = ensure_chat_action(adapter.possible_actions())
    store = PersistentStore(tmpdir / "memory.sqlite3")
    darwin = Darwin(actions=actions, store=store, seed=21, exploration_rate=0.1)
    darwin.learn(
        Transition(
            before={"curtains_open": False, "room_bright": False, "daylight": True, "switch_on": False, "fuse_intact": True, "battery_charge": 4},
            action="open_curtains",
            after={"curtains_open": True, "room_bright": True, "daylight": True, "switch_on": False, "fuse_intact": True, "battery_charge": 4},
            reward=1.0,
            t=0,
        )
    )
    goal = Goal(desired={"room_bright": True, "fuse_intact": True})
    return darwin, adapter, goal


def _build_daemon(tmpdir: Path, port: int) -> DarwinDaemon:
    darwin, adapter, goal = _seed_darwin(tmpdir)
    runtime = DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=goal,
        interval=0.2,
        logger=StructuredLogger(
            plan_log=tmpdir / "plans.jsonl",
            background_log=tmpdir / "background.jsonl",
            metrics_log=tmpdir / "metrics.jsonl",
        ),
        training_collector=TrainingDataCollector(path=tmpdir / "pairs.jsonl"),
        state_path=tmpdir / "state.json",
        loop_intervals={
            "experiment": 0.2,
            "simulation": 0.2,
            "dream": 0.3,
            "self_modification": 0.5,
            "uncertainty": 0.2,
        },
    )
    return DarwinDaemon(runtime, host="127.0.0.1", port=port)


class BrainDaemonTests(unittest.TestCase):
    def test_client_can_chat_with_brain_over_socket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            port = _free_port()
            daemon = _build_daemon(Path(directory), port)
            daemon.start()
            try:
                # Give the server a moment to bind
                deadline = time.time() + 2.0
                client = DarwinClient(host="127.0.0.1", port=port)
                events: list[dict] = []
                while time.time() < deadline:
                    try:
                        client.connect(events.append)
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("could not connect to daemon")

                pong = client.ping(timeout=2.0)
                self.assertEqual(pong["type"], "pong")

                result = client.chat("What do you believe about open_curtains?", timeout=5.0)
                self.assertEqual(result["type"], "response")
                self.assertTrue(result["text"])
                self.assertIn("plan", result)
                self.assertIn("causal_claims", result["plan"])

                lines = client.command("/beliefs", timeout=5.0)
                self.assertTrue(lines)
                self.assertTrue(any("open_curtains" in line for line in lines))

                client.close()
            finally:
                daemon.stop()

    def test_background_events_stream_to_subscribed_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            port = _free_port()
            daemon = _build_daemon(Path(directory), port)
            daemon.start()
            try:
                client = DarwinClient(host="127.0.0.1", port=port)
                received: list[dict] = []
                lock = threading.Lock()

                def collector(message: dict) -> None:
                    with lock:
                        received.append(message)

                deadline = time.time() + 2.0
                while time.time() < deadline:
                    try:
                        client.connect(collector)
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("could not connect to daemon")

                # Explicit opt-in: clients no longer auto-receive the firehose
                client.subscribe_events(timeout=2.0)
                # Wait long enough for background loops to fire
                time.sleep(1.5)
                client.close()

                with lock:
                    event_kinds = {m.get("kind") for m in received if m.get("type") == "event"}
                # Daemon should have streamed at least a few background events
                self.assertTrue(received)
                self.assertTrue(event_kinds & {"simulation", "experiment", "dream", "uncertainty", "self_modification", "reflection"})
            finally:
                daemon.stop()

    def test_unsubscribed_client_receives_no_events(self) -> None:
        """Default chat client must not receive the background firehose."""

        with tempfile.TemporaryDirectory() as directory:
            port = _free_port()
            daemon = _build_daemon(Path(directory), port)
            daemon.start()
            try:
                client = DarwinClient(host="127.0.0.1", port=port)
                received: list[dict] = []
                lock = threading.Lock()

                def collector(message: dict) -> None:
                    with lock:
                        received.append(message)

                deadline = time.time() + 2.0
                while time.time() < deadline:
                    try:
                        client.connect(collector)
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("could not connect to daemon")

                # NOTE: do NOT call subscribe_events. The chat REPL relies on
                # this exact silence. Background loops fire at <0.5s intervals
                # so 1.5s gives them many chances to leak.
                time.sleep(1.5)
                client.close()

                with lock:
                    event_messages = [m for m in received if m.get("type") == "event"]
                self.assertEqual(event_messages, [])
            finally:
                daemon.stop()

    def test_port_collision_raises_friendly_error(self) -> None:
        """A second brain on the same port must fail with PortInUseError
        and must not have started its background cognition loops."""

        with tempfile.TemporaryDirectory() as directory:
            port = _free_port()
            first = _build_daemon(Path(directory), port)
            first.start()
            try:
                second = _build_daemon(Path(directory) / "second", port)
                with self.assertRaises(PortInUseError):
                    second.start()
                # The second daemon must NOT have started its runtime
                # (because we bind the socket first; if bind fails, the
                # cognitive loops never spin up).
                self.assertFalse(second.runtime.running)
            finally:
                first.stop()

    def test_two_clients_share_one_brain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            port = _free_port()
            daemon = _build_daemon(Path(directory), port)
            daemon.start()
            try:
                deadline = time.time() + 2.0
                client_a = DarwinClient(host="127.0.0.1", port=port)
                events_a: list[dict] = []
                while time.time() < deadline:
                    try:
                        client_a.connect(events_a.append)
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("could not connect first client")

                client_b = DarwinClient(host="127.0.0.1", port=port)
                events_b: list[dict] = []
                client_b.connect(events_b.append)

                # Client A teaches Darwin something
                client_a.chat("The room should always be bright.", timeout=5.0)
                # Client B asks about goals
                result_b = client_b.command("/status", timeout=5.0)
                self.assertTrue(result_b)

                client_a.close()
                client_b.close()
            finally:
                daemon.stop()


if __name__ == "__main__":
    unittest.main()
