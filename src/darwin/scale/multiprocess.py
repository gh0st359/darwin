"""Multi-process supervision integration for V-Agents.

When DARWIN_MULTIPROCESS=1, the CognitionSupervisor spawns each agent in
its own subprocess. The agents communicate via the CognitionBus. This
module exposes the integration callable so runtime.py can opt in.
"""

from __future__ import annotations

from typing import Any

from darwin.mysterio.processes import RestartPolicy, SubsystemSpec


def agent_subsystem_specs(registry: Any) -> list[SubsystemSpec]:
    """Build SubsystemSpec entries for each agent in the registry.

    Each spec targets a per-agent loop entrypoint; when V-Scale spawns the
    spec, the loop wakes on AGENT_SOLVE bus events and dispatches to the
    agent's solve() method.
    """

    if registry is None:
        return []
    specs: list[SubsystemSpec] = []
    agent_names = ["code", "math", "science", "planning", "research", "dialogue"]
    for name in agent_names:
        agent = getattr(registry, name, None)
        if agent is None:
            continue
        specs.append(SubsystemSpec(
            name=f"agent_{name}",
            entrypoint="darwin.scale.multiprocess:_agent_loop",
            topics=["agent_solve"],
            priority=50,
            restart_policy=RestartPolicy.ON_FAILURE,
            max_restarts=8,
            kwargs={"agent_name": name},
        ))
    return specs


def _agent_loop(agent_name: str = "", **_: Any) -> None:
    """Subprocess entrypoint stub.

    In a fully-deployed setup this would attach to the bus and dispatch
    AGENT_SOLVE events. For V-Scale's reference build, it exits cleanly so
    the supervisor's restart loop is harmless when DARWIN_MULTIPROCESS is
    off in tests.
    """

    return None


__all__ = ["_agent_loop", "agent_subsystem_specs"]
