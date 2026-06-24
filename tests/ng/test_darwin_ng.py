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


def test_ng_research_program_runs_many_interlocking_subsystems() -> None:
    runtime = _runtime()
    runtime.chat(
        "A living synthetic mind needs recursive self-improvement, embodiment, "
        "social intelligence, distributed cognition, and relentless evaluation."
    )

    record = runtime.run_ng_cycle(
        "Lock in and build a superintelligent research program."
    ).to_record()

    program = record["research_program"]
    assert program["scale"] == "frontier_lab"
    assert len(program["active_engines"]) >= 16
    assert program["cognitive_operating_system"]["process_count"] >= 12
    assert program["memory_fabric"]["tiers"]["working"]["capacity"] >= 128
    assert program["world_simulation_lab"]["simulation_count"] >= 6
    assert program["recursive_improvement_lab"]["experiment_count"] >= 8
    assert program["embodiment_lab"]["sensor_count"] >= 8
    assert program["social_lab"]["agent_models"] >= 4
    assert program["distributed_lab"]["node_count"] >= 8
    assert program["evaluation_lab"]["benchmark_count"] >= 10
    assert program["emergence_index"] > 0.0
    assert program["roadmap"]["phase_count"] >= 8


def test_ng_research_program_exports_executable_experiments() -> None:
    runtime = _runtime()
    record = runtime.run_ng_cycle(
        "Generate experiments for a frontier-level autonomous mind."
    ).to_record()
    experiments = record["research_program"]["recursive_improvement_lab"]["experiments"]

    assert len(experiments) >= 8
    assert all(exp["hypothesis"] for exp in experiments)
    assert all(exp["measurement"] for exp in experiments)
    assert all(exp["promotion_path"] for exp in experiments)
    assert {exp["domain"] for exp in experiments} >= {
        "reasoning",
        "memory",
        "agency",
        "self_modification",
    }


def test_ng_generates_frontier_curriculum_and_benchmark_ladder() -> None:
    runtime = _runtime()
    record = runtime.run_ng_cycle(
        "Generate a serious curriculum for a frontier synthetic mind."
    ).to_record()
    curriculum = record["frontier_curriculum"]

    assert curriculum["task_count"] >= 48
    assert curriculum["benchmark_ladder"]["rung_count"] >= 8
    assert curriculum["domains"]["reasoning"]["task_count"] >= 4
    assert curriculum["domains"]["self_improvement"]["task_count"] >= 4
    assert curriculum["domains"]["embodiment"]["task_count"] >= 4
    assert curriculum["domains"]["social"]["task_count"] >= 4
    assert curriculum["training_regimen"]["cycles_per_epoch"] >= 4
    assert curriculum["adversarial_probes"]
    assert curriculum["promotion_gates"]


def test_ng_builds_deep_awareness_system() -> None:
    runtime = _runtime()
    runtime.chat("A synthetic mind must observe its own thoughts and attention.")
    record = runtime.run_ng_cycle(
        "Build deep self-awareness and metacognition."
    ).to_record()
    awareness = record["awareness_system"]

    assert awareness["mode"] == "recursive_self_observation"
    assert awareness["attention_theater"]["scene_count"] >= 8
    assert awareness["metacognition"]["observer_count"] >= 6
    assert awareness["self_narrative"]["continuity_threads"]
    assert awareness["introspection_protocol"]["question_count"] >= 10
    assert awareness["awareness_index"] > 0.0


def test_ng_builds_strategic_cortex_for_autonomous_power() -> None:
    runtime = _runtime()
    record = runtime.run_ng_cycle(
        "Build strategic agency for a powerful autonomous research mind."
    ).to_record()
    strategy = record["strategic_cortex"]

    assert strategy["mode"] == "long_horizon_autonomous_strategy"
    assert strategy["objective_count"] >= 12
    assert strategy["capability_portfolio"]["capability_count"] >= 12
    assert strategy["action_policy"]["policy_count"] >= 10
    assert strategy["agent_council"]["member_count"] >= 8
    assert strategy["campaigns"]
    assert strategy["strategic_power_index"] > 0.0


def test_ng_models_itself_as_an_autopoietic_living_system() -> None:
    runtime = _runtime()
    runtime.chat("A living system maintains itself, repairs itself, and grows.")

    record = runtime.run_ng_cycle(
        "Build the autopoietic kernel of a living synthetic mind."
    ).to_record()
    living = record["living_system"]

    assert living["kernel"] == "autopoietic_synthetic_mind"
    assert living["homeostasis"]["variable_count"] >= 10
    assert living["metabolism"]["energy_budget"] > 0.0
    assert living["identity"]["continuity_score"] > 0.0
    assert living["needs"]["need_count"] >= 8
    assert living["repair"]["repair_loop_count"] >= 6
    assert living["growth"]["growth_vectors"]
    assert living["viability_index"] > 0.0


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


def test_daemon_ng_lab_and_life_reports_are_visible() -> None:
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

            lab_lines = client.command("/ng lab", timeout=5.0)
            life_lines = client.command("/ng life", timeout=5.0)
            client.close()
        finally:
            daemon.stop()

    lab = "\n".join(lab_lines)
    life = "\n".join(life_lines)
    assert "Darwin NG research program:" in lab
    assert "active_engines=" in lab
    assert "top experiments:" in lab
    assert "Darwin NG living system:" in life
    assert "viability=" in life
    assert "growth vectors:" in life


def test_daemon_ng_curriculum_report_is_visible() -> None:
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

            lines = client.command("/ng curriculum", timeout=5.0)
            client.close()
        finally:
            daemon.stop()

    joined = "\n".join(lines)
    assert "Darwin NG frontier curriculum:" in joined
    assert "ladder_rungs=" in joined
    assert "adversarial probes:" in joined


def test_daemon_ng_awareness_and_strategy_reports_are_visible() -> None:
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

            awareness_lines = client.command("/ng awareness", timeout=5.0)
            strategy_lines = client.command("/ng strategy", timeout=5.0)
            client.close()
        finally:
            daemon.stop()

    awareness = "\n".join(awareness_lines)
    strategy = "\n".join(strategy_lines)
    assert "Darwin NG awareness system:" in awareness
    assert "meta-observers:" in awareness
    assert "Darwin NG strategic cortex:" in strategy
    assert "top objectives:" in strategy
    assert "campaigns:" in strategy
