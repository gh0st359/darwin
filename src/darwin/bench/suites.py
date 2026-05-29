"""Concrete benchmark tasks across the seven categories.

Each task is deterministic and produces a 0..1 score. The tasks here are
*minimum-viable* probes — they're meant to give a baseline shape Darwin
can be measured against, not to be a final word on intelligence. New
tasks can be added without touching the framework.
"""

from __future__ import annotations

import time
from typing import Any

from darwin.bench.framework import BenchmarkSuite, BenchmarkTask


# --------------------------------------------------------------------------- #
# Coding
# --------------------------------------------------------------------------- #


def _coding_addition(runtime: Any) -> tuple[float, dict]:
    """Darwin's code-execution tool should produce '3' for 1+2."""

    tool = runtime.tool_registry.tool("code")
    if tool is None:
        return 0.0, {"error": "no code tool registered"}
    result = tool.execute({"source": "print(1 + 2)"})
    success = result.success and "3" in result.output
    return (1.0 if success else 0.0), {
        "output": result.output[:200],
        "error": result.error[:200],
    }


def _coding_loop_sum(runtime: Any) -> tuple[float, dict]:
    """A small numerical task — sum of 1..10 == 55."""

    tool = runtime.tool_registry.tool("code")
    if tool is None:
        return 0.0, {"error": "no code tool registered"}
    result = tool.execute({"source": "print(sum(range(1, 11)))"})
    success = result.success and "55" in result.output
    return (1.0 if success else 0.0), {"output": result.output[:200]}


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


def _memory_recall_chat_fact(runtime: Any) -> tuple[float, dict]:
    """Teach a fact via chat, then test that asking about it derives it."""

    if not hasattr(runtime, "chat"):
        return 0.0, {"error": "runtime has no chat"}
    runtime.chat("A widget is a gadget.")
    runtime.chat("A gadget is a device.")
    reply = runtime.chat("Is a widget a device?")
    success = "widget" in reply.lower() and "device" in reply.lower() and (
        "yes" in reply.lower() or "is a" in reply.lower()
    )
    return (1.0 if success else 0.5 if "widget" in reply.lower() else 0.0), {
        "reply": reply[:200],
    }


def _memory_concept_persistence(runtime: Any) -> tuple[float, dict]:
    """A concept registered via chat survives a re-grounding pass."""

    if not hasattr(runtime, "universe"):
        return 0.0, {"error": "runtime has no universe"}
    runtime.chat("A quantum_widget is a widget.")
    has_concept = runtime.universe.has("quantum_widget")
    return (1.0 if has_concept else 0.0), {"concept_present": has_concept}


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #


def _learning_new_ontology(runtime: Any) -> tuple[float, dict]:
    """Build a 3-node taxonomy and check transitive derivation."""

    runtime.chat("A cargo_drone is a drone.")
    runtime.chat("A drone is a robot.")
    reply = runtime.chat("Is a cargo_drone a robot?")
    derived = "cargo_drone" in reply.lower() and "robot" in reply.lower() and (
        "yes" in reply.lower() or "is a" in reply.lower() or "the chain" in reply.lower()
    )
    return (1.0 if derived else 0.0), {"reply": reply[:200]}


def _learning_active_probe(runtime: Any) -> tuple[float, dict]:
    """When Darwin can't derive, it should ask a structured sub-question."""

    runtime.chat("A foo_thing is a bar_thing.")
    reply = runtime.chat("Is a foo_thing a baz_thing?")
    asked = "don't have a confident derivation" in reply or "need to know" in reply
    return (1.0 if asked else 0.0), {"reply": reply[:200]}


# --------------------------------------------------------------------------- #
# Adaptation
# --------------------------------------------------------------------------- #


def _adaptation_correction_path(runtime: Any) -> tuple[float, dict]:
    """A correction should refute the previous inference's key."""

    runtime.chat("A flerb is a glorp.")
    runtime.chat("Is a flerb a glorp?")
    runtime.chat("No, that's wrong.")
    hyp = runtime.hypothesis_engine if hasattr(runtime, "hypothesis_engine") else None
    # The hypothesis engine's refuted set should now hold the refuted edge.
    refuted_count = (
        len(getattr(hyp, "_refuted", set())) if hyp is not None else 0
    )
    return (1.0 if refuted_count > 0 else 0.0), {"refuted_count": refuted_count}


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def _planning_autonomous_marker(runtime: Any) -> tuple[float, dict]:
    """Autonomous task: write a marker file. Predicate = file exists."""

    if not hasattr(runtime, "tool_world") or not hasattr(runtime, "autonomous_runner"):
        return 0.0, {"error": "no autonomous runner"}
    from darwin.tools import AutonomousTask

    sandbox = getattr(runtime, "tool_sandbox_root", None)
    marker_name = f"plan_marker_{int(time.time() * 1000) % 1_000_000}.txt"
    runtime.tool_world.default_input["fs_write"] = {
        "path": marker_name, "content": "ok",
    }
    task = AutonomousTask(
        goal=f"write {marker_name}",
        max_steps=12,
        max_seconds=4.0,
        success_predicate=lambda state: bool(state.get("last_success")) and state.get("last_action") == "fs_write",
    )
    runtime.autonomous_runner.run(task)
    success = (sandbox is not None and (sandbox / marker_name).exists())
    return (1.0 if success else 0.0), {
        "task_steps": len(task.steps),
        "reason_stopped": task.reason_stopped,
    }


# --------------------------------------------------------------------------- #
# Reasoning
# --------------------------------------------------------------------------- #


def _reasoning_transitive_inference(runtime: Any) -> tuple[float, dict]:
    """A four-hop chain Darwin should be able to derive."""

    runtime.chat("A maple is a tree.")
    runtime.chat("A tree is a plant.")
    runtime.chat("A plant is a living_thing.")
    reply = runtime.chat("Is a maple a living_thing?")
    derived = "maple" in reply.lower() and "living_thing" in reply.lower() and (
        "yes" in reply.lower() or "is a" in reply.lower() or "chain" in reply.lower()
    )
    return (1.0 if derived else 0.0), {"reply": reply[:200]}


def _reasoning_self_introspection(runtime: Any) -> tuple[float, dict]:
    """Asking about Darwin's own reasoning should produce a substrate-grounded reply."""

    reply = runtime.chat("What are you thinking about?")
    grounded = (
        "concept" in reply.lower() or "relation" in reply.lower()
        or "domain" in reply.lower()
    )
    return (1.0 if grounded else 0.0), {"reply": reply[:200]}


# --------------------------------------------------------------------------- #
# Task completion
# --------------------------------------------------------------------------- #


def _task_completion_fs_round_trip(runtime: Any) -> tuple[float, dict]:
    """Write a file and read it back via the chat→tool intent path."""

    runtime.chat("run echo task-test > task_test.txt")
    reply = runtime.chat("read task_test.txt")
    success = "task-test" in reply
    return (1.0 if success else 0.0), {"reply": reply[:200]}


def _task_completion_shell(runtime: Any) -> tuple[float, dict]:
    """A shell command via intent routing must execute and surface output."""

    reply = runtime.chat("run echo round-trip-ok")
    success = "round-trip-ok" in reply
    return (1.0 if success else 0.0), {"reply": reply[:200]}


# --------------------------------------------------------------------------- #
# Suite assembly
# --------------------------------------------------------------------------- #


def build_default_suite() -> BenchmarkSuite:
    suite = BenchmarkSuite(name="darwin-default")
    suite.add(BenchmarkTask("coding/addition", "coding",
                            "1 + 2 via code tool", _coding_addition))
    suite.add(BenchmarkTask("coding/loop_sum", "coding",
                            "sum(1..10) via code tool", _coding_loop_sum))
    suite.add(BenchmarkTask("memory/recall_chat_fact", "memory",
                            "transitive recall of a chat-taught fact",
                            _memory_recall_chat_fact))
    suite.add(BenchmarkTask("memory/concept_persistence", "memory",
                            "concept registered via chat is recoverable",
                            _memory_concept_persistence))
    suite.add(BenchmarkTask("learning/new_ontology", "learning",
                            "build a 3-node taxonomy + derive transitively",
                            _learning_new_ontology))
    suite.add(BenchmarkTask("learning/active_probe", "learning",
                            "darwin asks back when graph is thin",
                            _learning_active_probe))
    suite.add(BenchmarkTask("adaptation/correction_path", "adaptation",
                            "correction refutes prior inference",
                            _adaptation_correction_path))
    suite.add(BenchmarkTask("planning/autonomous_marker", "planning",
                            "autonomous runner writes a marker file",
                            _planning_autonomous_marker, weight=1.5))
    suite.add(BenchmarkTask("reasoning/transitive_inference", "reasoning",
                            "four-hop is_a derivation", _reasoning_transitive_inference))
    suite.add(BenchmarkTask("reasoning/self_introspection", "reasoning",
                            "substrate-grounded self-introspection",
                            _reasoning_self_introspection))
    suite.add(BenchmarkTask("task_completion/fs_round_trip", "task_completion",
                            "chat→tool round-trip: write then read",
                            _task_completion_fs_round_trip))
    suite.add(BenchmarkTask("task_completion/shell", "task_completion",
                            "shell command via intent routing",
                            _task_completion_shell))
    return suite
