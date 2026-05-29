"""Multi-process supervision: one OS process per cognitive subsystem.

Single-thread Darwin ran every loop in one process; mysterio gives each
cognitive subsystem its own `multiprocessing.Process` so hundreds of thought
streams run in true parallel and a crash in one cannot wedge the others.

`SubsystemSpec` declares a subsystem: its name, the entrypoint callable, a
restart policy, a priority, and the bus topics it touches. `Cognition
Supervisor` spawns each spec, watches liveness, and restarts crashed
subsystems from the last snapshot per their policy. The supervisor itself is
the only piece that must stay up; everything else is restartable.

The v6 roster (12 processes, designed to grow past 30 via code-gen self-mod):
kernel, causal_learner, simulator, interior_simulator, planner, self_modeler,
realizer, researcher, consolidator, narrator, observer_modeler,
embedding_trainer.

Entrypoints are resolved by dotted path at spawn time (``module:function``)
so specs stay picklable across the spawn boundary on every platform.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RestartPolicy(str, Enum):
    ALWAYS = "always"          # restart on any exit
    ON_FAILURE = "on_failure"  # restart only on non-zero exit
    NEVER = "never"            # one-shot


@dataclass
class SubsystemSpec:
    name: str
    entrypoint: str  # "module.path:function" resolved in the child process
    topics: list[str] = field(default_factory=list)
    priority: int = 0
    restart_policy: RestartPolicy = RestartPolicy.ALWAYS
    max_restarts: int = 1000
    kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> Callable[..., Any]:
        module_path, _, func_name = self.entrypoint.partition(":")
        if not func_name:
            raise ValueError(
                f"entrypoint must be 'module:function', got {self.entrypoint!r}"
            )
        module = importlib.import_module(module_path)
        func = getattr(module, func_name, None)
        if not callable(func):
            raise ValueError(f"entrypoint {self.entrypoint!r} is not callable")
        return func

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "topics": list(self.topics),
            "priority": self.priority,
            "restart_policy": self.restart_policy.value,
            "max_restarts": self.max_restarts,
        }


@dataclass
class SubsystemHandle:
    spec: SubsystemSpec
    process: Any = None  # mp.Process | None
    restarts: int = 0
    started_at: float = 0.0
    last_exit_code: int | None = None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()


def _child_main(entrypoint: str, kwargs: dict[str, Any]) -> None:
    """Trampoline executed inside each child process."""
    module_path, _, func_name = entrypoint.partition(":")
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    func(**kwargs)


# Default v6 roster. Entrypoints point at lightweight loop functions; until
# the dedicated subsystem modules land they can target a shared heartbeat
# entrypoint. The roster shape is what v6 commits to.
DEFAULT_ROSTER: list[SubsystemSpec] = [
    SubsystemSpec("kernel", "darwin.mysterio.processes:_heartbeat", priority=100,
                  topics=["subsystem_health"]),
    SubsystemSpec("causal_learner", "darwin.mysterio.processes:_heartbeat",
                  priority=90, topics=["transitions"]),
    SubsystemSpec("simulator", "darwin.mysterio.processes:_heartbeat",
                  priority=70, topics=["simulations"]),
    SubsystemSpec("interior_simulator", "darwin.mysterio.processes:_heartbeat",
                  priority=70, topics=["interior_simulations"]),
    SubsystemSpec("planner", "darwin.mysterio.processes:_heartbeat",
                  priority=80, topics=["proposals"]),
    SubsystemSpec("self_modeler", "darwin.mysterio.processes:_heartbeat",
                  priority=80),
    SubsystemSpec("realizer", "darwin.mysterio.processes:_heartbeat",
                  priority=60, topics=["narrative"]),
    SubsystemSpec("researcher", "darwin.mysterio.processes:_heartbeat",
                  priority=50, topics=["research_findings"]),
    SubsystemSpec("consolidator", "darwin.mysterio.processes:_heartbeat",
                  priority=50),
    SubsystemSpec("narrator", "darwin.mysterio.processes:_heartbeat",
                  priority=40, topics=["narrative"]),
    SubsystemSpec("observer_modeler", "darwin.mysterio.processes:_heartbeat",
                  priority=40, topics=["observer_events"]),
    SubsystemSpec("embedding_trainer", "darwin.mysterio.processes:_heartbeat",
                  priority=30, topics=["embedding_updates"]),
]


def _heartbeat(**kwargs: Any) -> None:  # pragma: no cover - runs in child
    """Minimal subsystem body: emit liveness until terminated."""
    interval = float(kwargs.get("interval", 1.0))
    while True:
        time.sleep(interval)


class CognitionSupervisor:
    """Spawns, watches, and restarts subsystem processes.

    The supervisor keeps a `SubsystemHandle` per spec. :meth:`spawn_all`
    starts the roster; :meth:`reap` checks for dead children and restarts them
    per policy (this is called by the kernel on its scheduling tick).

    Restart-from-snapshot is expressed via the spec's kwargs: a restart passes
    the latest snapshot id so the child can rehydrate. The supervisor does not
    itself read snapshots — it just threads the id through, keeping the
    supervisor free of cognition-specific coupling.
    """

    def __init__(
        self,
        roster: list[SubsystemSpec] | None = None,
        *,
        context: Any = None,
        snapshot_id_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self.roster = list(roster if roster is not None else DEFAULT_ROSTER)
        self.handles: dict[str, SubsystemHandle] = {
            spec.name: SubsystemHandle(spec=spec) for spec in self.roster
        }
        self._context = context if context is not None else mp.get_context("spawn")
        self._snapshot_id_provider = snapshot_id_provider
        self._started = False

    def register(self, spec: SubsystemSpec) -> None:
        """Add a new subsystem at runtime (the code-gen growth path)."""
        self.roster.append(spec)
        self.handles[spec.name] = SubsystemHandle(spec=spec)

    def spawn(self, name: str) -> SubsystemHandle:
        handle = self.handles[name]
        spec = handle.spec
        kwargs = dict(spec.kwargs)
        if self._snapshot_id_provider is not None:
            kwargs.setdefault("restore_snapshot_id", self._snapshot_id_provider())
        process = self._context.Process(
            target=_child_main,
            args=(spec.entrypoint, kwargs),
            name=f"darwin-{spec.name}",
            daemon=True,
        )
        process.start()
        handle.process = process
        handle.started_at = time.time()
        return handle

    def spawn_all(self) -> None:
        # Spawn in descending priority so high-priority subsystems come up first.
        for spec in sorted(self.roster, key=lambda s: -s.priority):
            self.spawn(spec.name)
        self._started = True

    def reap(self) -> list[str]:
        """Check liveness; restart dead children per policy. Returns restarted."""
        restarted: list[str] = []
        for name, handle in self.handles.items():
            process = handle.process
            if process is None or process.is_alive():
                continue
            handle.last_exit_code = process.exitcode
            policy = handle.spec.restart_policy
            should = (
                policy is RestartPolicy.ALWAYS
                or (policy is RestartPolicy.ON_FAILURE and (process.exitcode or 0) != 0)
            )
            if should and handle.restarts < handle.spec.max_restarts:
                handle.restarts += 1
                self.spawn(name)
                restarted.append(name)
        return restarted

    def shutdown(self, timeout: float = 2.0) -> None:
        for handle in self.handles.values():
            process = handle.process
            if process is not None and process.is_alive():
                process.terminate()
        deadline = time.time() + timeout
        for handle in self.handles.values():
            process = handle.process
            if process is None:
                continue
            remaining = max(0.0, deadline - time.time())
            process.join(timeout=remaining)
            if process.is_alive():
                process.kill()
        self._started = False

    def roster_status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": h.spec.name,
                "priority": h.spec.priority,
                "alive": h.alive,
                "restarts": h.restarts,
                "last_exit_code": h.last_exit_code,
                "topics": list(h.spec.topics),
            }
            for h in sorted(self.handles.values(), key=lambda x: -x.spec.priority)
        ]

    def __len__(self) -> int:
        return len(self.handles)
