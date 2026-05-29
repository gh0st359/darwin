"""Tests for the cognition supervisor.

The supervisor manages OS processes per subsystem. We test the bookkeeping
surface (roster, restart accounting, status) without spawning real children —
running real multiprocessing in CI per-test is flaky and platform-dependent.
A fake handle/process pair gives us deterministic coverage of the restart
policy logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from darwin.mysterio.processes import (
    DEFAULT_ROSTER,
    CognitionSupervisor,
    RestartPolicy,
    SubsystemSpec,
)


@dataclass
class _FakeProcess:
    alive: bool = True
    exitcode: int | None = None
    started: bool = False

    def is_alive(self) -> bool:
        return self.alive

    def start(self) -> None:
        self.started = True
        self.alive = True

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout: float = 0.0) -> None:
        return None


@dataclass
class _FakeContext:
    last_started: list[_FakeProcess] = field(default_factory=list)

    def Process(self, **kwargs: Any) -> _FakeProcess:  # noqa: N802 - mimic mp API
        process = _FakeProcess()
        self.last_started.append(process)
        return process


def test_default_roster_has_interior_simulator() -> None:
    names = {spec.name for spec in DEFAULT_ROSTER}
    assert "interior_simulator" in names
    assert "private_simulator" not in names
    assert "embedding_trainer" in names
    assert "narrator" in names
    assert len(DEFAULT_ROSTER) == 12


def test_spawn_all_starts_every_subsystem_in_priority_order() -> None:
    fake_ctx = _FakeContext()
    supervisor = CognitionSupervisor(
        roster=[
            SubsystemSpec("low", "darwin.mysterio.processes:_heartbeat", priority=10),
            SubsystemSpec("high", "darwin.mysterio.processes:_heartbeat", priority=90),
            SubsystemSpec("mid", "darwin.mysterio.processes:_heartbeat", priority=50),
        ],
        context=fake_ctx,
    )
    supervisor.spawn_all()
    started = [p for p in fake_ctx.last_started if p.started]
    assert len(started) == 3
    assert all(handle.process is not None for handle in supervisor.handles.values())


def test_reap_restarts_dead_subsystem_under_always_policy() -> None:
    fake_ctx = _FakeContext()
    supervisor = CognitionSupervisor(
        roster=[
            SubsystemSpec(
                "kernel",
                "darwin.mysterio.processes:_heartbeat",
                priority=100,
                restart_policy=RestartPolicy.ALWAYS,
                max_restarts=3,
            )
        ],
        context=fake_ctx,
    )
    supervisor.spawn_all()
    process = supervisor.handles["kernel"].process
    process.alive = False
    process.exitcode = 1

    restarted = supervisor.reap()
    assert restarted == ["kernel"]
    assert supervisor.handles["kernel"].restarts == 1
    # A clean exit followed by another reap restarts again (ALWAYS policy).
    new_process = supervisor.handles["kernel"].process
    new_process.alive = False
    new_process.exitcode = 0
    restarted_again = supervisor.reap()
    assert restarted_again == ["kernel"]
    assert supervisor.handles["kernel"].restarts == 2


def test_reap_does_not_restart_clean_exit_under_on_failure() -> None:
    fake_ctx = _FakeContext()
    supervisor = CognitionSupervisor(
        roster=[
            SubsystemSpec(
                "researcher",
                "darwin.mysterio.processes:_heartbeat",
                restart_policy=RestartPolicy.ON_FAILURE,
                max_restarts=5,
            )
        ],
        context=fake_ctx,
    )
    supervisor.spawn_all()
    process = supervisor.handles["researcher"].process
    process.alive = False
    process.exitcode = 0  # Clean exit
    restarted = supervisor.reap()
    assert restarted == []
    assert supervisor.handles["researcher"].restarts == 0

    # Failure exit DOES restart.
    new_process = supervisor.handles["researcher"].process
    new_process.alive = False
    new_process.exitcode = 1
    restarted_again = supervisor.reap()
    assert restarted_again == ["researcher"]


def test_max_restarts_caps_restart_attempts() -> None:
    fake_ctx = _FakeContext()
    supervisor = CognitionSupervisor(
        roster=[
            SubsystemSpec(
                "narrator",
                "darwin.mysterio.processes:_heartbeat",
                restart_policy=RestartPolicy.ALWAYS,
                max_restarts=2,
            )
        ],
        context=fake_ctx,
    )
    supervisor.spawn_all()
    handle = supervisor.handles["narrator"]
    for _ in range(5):
        handle.process.alive = False
        handle.process.exitcode = 1
        supervisor.reap()
    assert handle.restarts == 2


def test_register_adds_new_subsystem_at_runtime() -> None:
    fake_ctx = _FakeContext()
    supervisor = CognitionSupervisor(roster=[], context=fake_ctx)
    spec = SubsystemSpec(
        name="generated_subsystem_alpha",
        entrypoint="darwin.mysterio.processes:_heartbeat",
        priority=20,
    )
    supervisor.register(spec)
    assert "generated_subsystem_alpha" in supervisor.handles
    supervisor.spawn("generated_subsystem_alpha")
    assert supervisor.handles["generated_subsystem_alpha"].process is not None


def test_roster_status_sorted_by_priority_descending() -> None:
    fake_ctx = _FakeContext()
    supervisor = CognitionSupervisor(
        roster=[
            SubsystemSpec("low", "darwin.mysterio.processes:_heartbeat", priority=5),
            SubsystemSpec("high", "darwin.mysterio.processes:_heartbeat", priority=95),
            SubsystemSpec("mid", "darwin.mysterio.processes:_heartbeat", priority=50),
        ],
        context=fake_ctx,
    )
    status = supervisor.roster_status()
    priorities = [row["priority"] for row in status]
    assert priorities == sorted(priorities, reverse=True)


def test_entrypoint_must_be_module_function_form() -> None:
    spec = SubsystemSpec(name="bad", entrypoint="just_a_string")
    with pytest.raises(ValueError):
        spec.resolve()


def test_entrypoint_resolves_real_callable() -> None:
    spec = SubsystemSpec(
        name="real", entrypoint="darwin.mysterio.processes:_heartbeat",
    )
    callable_ref = spec.resolve()
    assert callable(callable_ref)
