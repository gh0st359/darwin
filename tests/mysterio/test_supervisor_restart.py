"""Supervisor spawns subsystems and restarts dead ones per policy."""

from __future__ import annotations

import multiprocessing as mp
import time

from darwin.mysterio.processes import (
    CognitionSupervisor,
    RestartPolicy,
    SubsystemSpec,
)


def _short_lived(**kwargs) -> None:  # pragma: no cover - child process
    """Exit almost immediately so the supervisor sees a dead child."""
    time.sleep(0.05)


def _crasher(**kwargs) -> None:  # pragma: no cover - child process
    raise SystemExit(1)


def test_spec_resolves_entrypoint() -> None:
    spec = SubsystemSpec(
        "short", "tests.mysterio.test_supervisor_restart:_short_lived"
    )
    assert callable(spec.resolve())


def test_register_grows_roster() -> None:
    sup = CognitionSupervisor(roster=[])
    assert len(sup) == 0
    sup.register(SubsystemSpec("new", "darwin.mysterio.processes:_heartbeat"))
    assert len(sup) == 1
    assert "new" in sup.handles


def test_spawn_and_restart_dead_subsystem() -> None:
    ctx = mp.get_context("spawn")
    spec = SubsystemSpec(
        "short",
        "tests.mysterio.test_supervisor_restart:_short_lived",
        restart_policy=RestartPolicy.ALWAYS,
        max_restarts=2,
    )
    sup = CognitionSupervisor(roster=[spec], context=ctx)
    sup.spawn("short")
    # Wait for the short-lived child to exit.
    deadline = time.time() + 5.0
    while sup.handles["short"].alive and time.time() < deadline:
        time.sleep(0.05)
    restarted = sup.reap()
    assert "short" in restarted
    assert sup.handles["short"].restarts == 1
    sup.shutdown()


def test_max_restarts_respected() -> None:
    ctx = mp.get_context("spawn")
    spec = SubsystemSpec(
        "crasher",
        "tests.mysterio.test_supervisor_restart:_crasher",
        restart_policy=RestartPolicy.ON_FAILURE,
        max_restarts=1,
    )
    sup = CognitionSupervisor(roster=[spec], context=ctx)
    sup.spawn("crasher")
    # Reap repeatedly; it should restart at most once then give up.
    for _ in range(20):
        time.sleep(0.1)
        sup.reap()
        if sup.handles["crasher"].restarts >= 1 and not sup.handles["crasher"].alive:
            break
    assert sup.handles["crasher"].restarts <= 1
    sup.shutdown()


def test_roster_status_sorted_by_priority() -> None:
    sup = CognitionSupervisor(
        roster=[
            SubsystemSpec("low", "darwin.mysterio.processes:_heartbeat", priority=1),
            SubsystemSpec("high", "darwin.mysterio.processes:_heartbeat", priority=99),
        ]
    )
    status = sup.roster_status()
    assert status[0]["name"] == "high"
    assert status[1]["name"] == "low"
