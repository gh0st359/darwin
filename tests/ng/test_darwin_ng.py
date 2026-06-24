from __future__ import annotations

import socket
import tempfile
import time
from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.server import DarwinClient, DarwinDaemon
from darwin.storage import PersistentStore
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _runtime(tmpdir: Path | None = None) -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=77)
    adapter = RoomSimulationAdapter(world)
    store = PersistentStore(tmpdir / "memory.sqlite3") if tmpdir is not None else None
    darwin = Darwin(
        actions=ensure_chat_action(adapter.possible_actions()),
        store=store,
        seed=77,
        exploration_rate=0.0,
    )
    return DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=Goal(desired={"room_bright": True}),
        interval=100.0,
        state_path=False,
    )


def test_ng_cycle_integrates_workspace_goals_and_meta_learning() -> None:
    runtime = _runtime()
    runtime.chat("A synthetic mind is an autonomous learning system.")

    state = runtime.run_ng_cycle(
        "Build toward a self-directed synthetic intelligence with memory and agency."
    )
    record = state.to_record()

    assert record["workspace"]["dynamic_core"]
    assert record["workspace"]["phi_proxy"] >= 0.0
    assert "curiosity" in record["drives"]
    assert record["goals"]
    assert record["plans"]
    assert record["capabilities"]["mode"] == "full_capability_visibility"
    assert record["capabilities"]["tools"]["count"] >= 6
    assert record["capabilities"]["autonomy"]["goal_orchestrator"] is True
    assert record["capabilities"]["self_improvement"]["self_mod_engine"] is True
    assert record["safety"]["governance_level"] == "self-directed-with-audit"
    assert record["meta_learning"]["deployment"] == (
        "self_directed_goals_with_auditable_state_changes"
    )
    assert runtime.last_ng_state is state


def test_ng_cycle_builds_full_frontier_cognitive_stack() -> None:
    runtime = _runtime()
    runtime.chat("A frontier mind should reason, learn, embody, collaborate, and improve.")

    record = runtime.run_ng_cycle(
        "Make Darwin NG a frontier autonomous synthetic mind."
    ).to_record()

    stack = record["cognitive_stack"]
    assert stack["vision"] == "frontier_breakthrough_synthetic_mind"
    assert stack["layer_0_quantum_foundation"]["mode"] in {
        "classical_emulation",
        "accelerator_ready",
    }
    assert stack["layer_1_neuro_symbolic_core"]["fusion_sources"] >= 5
    assert stack["layer_2_consciousness_engine"]["global_workspace"]["phi_proxy"] >= 0.0
    assert stack["layer_3_autonomous_agency"]["goal_graph"]["nodes"]
    assert stack["layer_4_self_improvement"]["recursive_agenda"]
    assert stack["layer_5_embodiment_social"]["embodiment"]["affordances"]
    assert stack["layer_5_embodiment_social"]["social"]["theory_of_mind_depth"] >= 1

    metrics = record["power_metrics"]
    assert metrics["architecture_orders_of_magnitude"] >= 5
    assert metrics["parallel_cognitive_streams"] >= 10
    assert metrics["recursive_improvement_index"] > 0.0
    assert metrics["autonomy_index"] > 0.0
    assert metrics["total_frontier_score"] > 0.0


def test_ng_can_activate_self_generated_goals_in_durable_autonomy_ledger() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = _runtime(Path(directory))
        state = runtime.run_ng_cycle(
            "Create long-horizon goals for autonomous intelligence growth."
        )

        report = runtime.activate_ng_autonomy(limit=2)

        assert report["activated"] >= 1
        assert report["ledger_goal_ids"]
        assert runtime.goal_ledger is not None
        goals = [runtime.goal_ledger.goal(goal_id) for goal_id in report["ledger_goal_ids"]]
        assert all(goal is not None for goal in goals)
        assert all(goal.metadata.get("source") == "darwin_ng" for goal in goals if goal)
        assert state.to_record()["frontier_protocols"]["autonomous_goal_graph"]["nodes"]


def test_ng_state_publishes_to_cognition_bus() -> None:
    runtime = _runtime()
    runtime.run_ng_cycle("Let the next-generation workspace organize itself.")
    events = runtime.bus.recent("ng_state", limit=3)
    assert events
    assert events[-1].source == "darwin_ng"
    assert events[-1].payload["workspace"]["dynamic_core"]


def test_daemon_ng_command_reports_self_directed_goals() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = _runtime(Path(directory))
        daemon = DarwinDaemon(runtime, host="127.0.0.1", port=_free_port())
        daemon.start()
        try:
            client = DarwinClient(host=daemon.host, port=daemon.port)
            deadline = time.time() + 2.0
            while time.time() < deadline:
                try:
                    client.connect(lambda _msg: None)
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                raise AssertionError("could not connect to daemon")

            lines = client.command(
                "/ng awaken a self-directed synthetic intelligence",
                timeout=5.0,
            )
            client.close()
        finally:
            daemon.stop()

    joined = "\n".join(lines)
    assert "Darwin NG cycle=" in joined
    assert "self-directed goals:" in joined
    assert "governance=self-directed-with-audit" in joined


def test_daemon_ng_capabilities_reports_full_surface() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = _runtime(Path(directory))
        daemon = DarwinDaemon(runtime, host="127.0.0.1", port=_free_port())
        daemon.start()
        try:
            client = DarwinClient(host=daemon.host, port=daemon.port)
            deadline = time.time() + 2.0
            while time.time() < deadline:
                try:
                    client.connect(lambda _msg: None)
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                raise AssertionError("could not connect to daemon")

            lines = client.command("/ng capabilities", timeout=5.0)
            client.close()
        finally:
            daemon.stop()

    joined = "\n".join(lines)
    assert "Darwin NG capability manifest:" in joined
    assert "tools (6):" in joined
    assert "autonomy:" in joined
    assert "self-improvement:" in joined
    assert "reasoning:" in joined
    assert "modalities:" in joined


def test_daemon_ng_frontier_reports_power_metrics_and_protocols() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = _runtime(Path(directory))
        daemon = DarwinDaemon(runtime, host="127.0.0.1", port=_free_port())
        daemon.start()
        try:
            client = DarwinClient(host=daemon.host, port=daemon.port)
            deadline = time.time() + 2.0
            while time.time() < deadline:
                try:
                    client.connect(lambda _msg: None)
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                raise AssertionError("could not connect to daemon")

            lines = client.command("/ng frontier", timeout=5.0)
            client.close()
        finally:
            daemon.stop()

    joined = "\n".join(lines)
    assert "Darwin NG frontier stack:" in joined
    assert "power metrics:" in joined
    assert "frontier protocols:" in joined
    assert "recursive_self_improvement_queue" in joined
    assert "autonomous_goal_graph" in joined
