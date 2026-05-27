from __future__ import annotations

import heapq
import itertools
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from darwin.runtime import DarwinRuntime, RuntimeEvent


# ---------------------------------------------------------------------------
# Job / metrics types.
# ---------------------------------------------------------------------------


@dataclass
class KernelJob:
    """A single unit of cognitive work the scheduler can dispatch.

    ``priority`` is a real number; HIGHER values are processed first. We
    store the negated priority inside the heap so Python's min-heap acts
    as a max-priority queue.
    """

    kind: str
    priority: float
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    job_id: int = 0


@dataclass
class KernelMetrics:
    jobs_scheduled: int = 0
    jobs_completed: int = 0
    repeated_loop_redirects: int = 0
    experiments_started: int = 0
    useful_beliefs: int = 0
    dlm_rejections: int = 0
    saturation_skips: int = 0
    starvation_lifts: int = 0
    completions_by_kind: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        elapsed = max(0.001, time.time() - self.started_at)
        return {
            "jobs_scheduled": self.jobs_scheduled,
            "jobs_completed": self.jobs_completed,
            "repeated_loop_redirects": self.repeated_loop_redirects,
            "experiments_started": self.experiments_started,
            "useful_beliefs": self.useful_beliefs,
            "dlm_rejections": self.dlm_rejections,
            "saturation_skips": self.saturation_skips,
            "starvation_lifts": self.starvation_lifts,
            "completions_by_kind": dict(self.completions_by_kind),
            "experiments_per_minute": 60.0 * self.experiments_started / elapsed,
            "useful_beliefs_per_hour": 3600.0 * self.useful_beliefs / elapsed,
        }


# Default per-kind in-flight saturation caps. The scheduler will skip a
# job whose kind already has at least this many handlers running.
DEFAULT_SATURATION_CAPS: dict[str, int] = {
    "experiment": 4,
    "simulation": 2,
    "dream": 1,
    "self_modification": 1,
    "uncertainty": 1,
    "consolidation": 1,
}


# ---------------------------------------------------------------------------
# ActorScheduler — heapq-backed priority queue with per-kind in-flight caps.
# ---------------------------------------------------------------------------


class ActorScheduler:
    """Priority-queue scheduler for Darwin's background cognition.

    ``schedule(job)`` adds work to the heap. ``pop_next()`` returns the
    highest-priority job whose kind hasn't saturated. ``complete(job)``
    records the completion + drops the in-flight count back down.

    The driver thread (``KernelDriver``) is what actually runs jobs. The
    scheduler is intentionally driver-agnostic — Phase F can stack other
    drivers (e.g. a workstealing pool) on top.
    """

    def __init__(
        self,
        workers: str = "auto",
        accelerator: str = "auto",
        saturation_caps: dict[str, int] | None = None,
    ) -> None:
        self.workers = os.cpu_count() if workers == "auto" else max(1, int(workers))
        self.accelerator = accelerator
        self.metrics = KernelMetrics()
        self._heap: list[tuple[float, int, int, KernelJob]] = []
        self._counter = itertools.count()
        self._in_flight: dict[str, int] = {}
        self._completion_window: deque[tuple[float, str]] = deque(maxlen=2000)
        self._lock = threading.RLock()
        self.saturation_caps: dict[str, int] = dict(DEFAULT_SATURATION_CAPS)
        if saturation_caps:
            self.saturation_caps.update(saturation_caps)

    # -- enqueue / dispatch -------------------------------------------------

    def schedule(self, job: KernelJob) -> None:
        with self._lock:
            counter_value = next(self._counter)
            job.job_id = counter_value
            # heapq is a min-heap: negate priority for max-priority order.
            # The tuple key uses (-priority, age, counter) so older jobs of
            # equal priority drain first, and equal-(priority, age) jobs use
            # the counter for stable ordering.
            heapq.heappush(
                self._heap,
                (-float(job.priority), counter_value, counter_value, job),
            )
            self.metrics.jobs_scheduled += 1

    def pop_next(self) -> KernelJob | None:
        """Return the next runnable job, or None if heap is empty/saturated."""

        with self._lock:
            holding: list[tuple[float, int, int, KernelJob]] = []
            chosen: KernelJob | None = None
            while self._heap:
                entry = heapq.heappop(self._heap)
                job = entry[3]
                cap = self.saturation_caps.get(job.kind, 99)
                if self._in_flight.get(job.kind, 0) >= cap:
                    self.metrics.saturation_skips += 1
                    holding.append(entry)
                    continue
                chosen = job
                self._in_flight[job.kind] = self._in_flight.get(job.kind, 0) + 1
                break
            # Re-heapify the held-back jobs so a later tick can try them
            # again once in-flight counts drop.
            for entry in holding:
                heapq.heappush(self._heap, entry)
            return chosen

    def complete(self, job: KernelJob) -> None:
        with self._lock:
            self.metrics.jobs_completed += 1
            self.metrics.completions_by_kind[job.kind] = (
                self.metrics.completions_by_kind.get(job.kind, 0) + 1
            )
            self._completion_window.append((time.time(), job.kind))
            self._in_flight[job.kind] = max(0, self._in_flight.get(job.kind, 0) - 1)
            if job.kind == "experiment":
                self.metrics.experiments_started += 1

    # -- introspection ------------------------------------------------------

    def queue_size(self) -> int:
        with self._lock:
            return len(self._heap)

    def in_flight(self) -> dict[str, int]:
        with self._lock:
            return dict(self._in_flight)

    def completion_rate(self, kind: str, window_seconds: float = 600.0) -> float:
        """Completions/minute of a given kind over the last ``window_seconds``."""

        cutoff = time.time() - window_seconds
        with self._lock:
            hits = sum(1 for ts, k in self._completion_window if k == kind and ts >= cutoff)
        elapsed_min = max(0.001, window_seconds / 60.0)
        return hits / elapsed_min

    def lift_starvation(self, kind: str) -> None:
        """Record that the driver intentionally lifted a saturation cap."""

        with self._lock:
            self.metrics.starvation_lifts += 1


# ---------------------------------------------------------------------------
# KernelDriver — single thread, pulls jobs from the scheduler.
# ---------------------------------------------------------------------------


JOB_HANDLERS = {
    "experiment": "_loop_experiment",
    "simulation": "_loop_simulation",
    "dream": "_loop_dream",
    "self_modification": "_loop_self_modification",
    "uncertainty": "_loop_uncertainty",
    "consolidation": "_handle_consolidation",
}


class KernelDriver:
    """Single-thread driver that pulls jobs from ``ActorScheduler``.

    Each tick:
      1. Pop the highest-priority runnable job (saturation-aware).
      2. Invoke the matching handler on the runtime (the existing v3/v4
         ``_loop_*`` methods double as job handlers — we don't fork the
         implementation, just the trigger).
      3. Record completion.
      4. If the heap is empty, call ``_replenish`` which inspects Darwin's
         self-report and enqueues new work biased toward the current
         learning priority.
    """

    def __init__(
        self,
        runtime: "DarwinRuntime",
        scheduler: ActorScheduler,
        tick_interval: float = 0.5,
        replenish_floor: int = 4,
    ) -> None:
        self.runtime = runtime
        self.scheduler = scheduler
        self.tick_interval = tick_interval
        self.replenish_floor = replenish_floor
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Phase D priority formula coefficients — tunable in Phase E.
        self.priority_weights = {
            "uncertainty": 0.6,
            "learning_priority_match": 0.3,
            "age": 0.1,
        }

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="darwin-kernel-driver",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- main loop ----------------------------------------------------------

    def _run(self) -> None:
        # Seed the scheduler so the first tick has something to do.
        self._replenish()
        while not self._stop.wait(self.tick_interval):
            self.tick()

    def tick(self) -> None:
        """One scheduler tick. Public so tests can drive deterministically."""

        if self.scheduler.queue_size() < self.replenish_floor:
            self._replenish()
        job = self.scheduler.pop_next()
        if job is None:
            return
        try:
            self._dispatch(job)
        except Exception as exc:  # pragma: no cover - defensive
            self.runtime._event(
                "error",
                f"kernel job {job.kind} failed: {exc!r}",
                payload={"job_kind": job.kind, "job_priority": job.priority},
                loop=f"kernel:{job.kind}",
            )
        finally:
            self.scheduler.complete(job)

    def _dispatch(self, job: KernelJob) -> None:
        handler_name = JOB_HANDLERS.get(job.kind)
        if handler_name is None:
            return
        handler = getattr(self.runtime, handler_name, None)
        if not callable(handler):
            return
        # Handlers were originally background-loop functions that take no
        # args; treat the job's payload as a side-channel for future use.
        handler()

    # -- replenishment ------------------------------------------------------

    def _replenish(self) -> None:
        """Enqueue jobs biased toward Darwin's current learning priority."""

        try:
            report = self.runtime.darwin.self_report()
        except Exception:  # pragma: no cover - defensive
            return

        priority_target = self._normalize_priority_string(report.learning_priority)
        weakest = self._normalize_priority_string(report.weakest_area)

        # Always enqueue at least one of each kind so no loop starves.
        kinds = [
            ("experiment", 0.65),
            ("simulation", 0.45),
            ("uncertainty", 0.4),
            ("dream", 0.3),
            ("self_modification", 0.25),
            ("consolidation", 0.2),
        ]
        for kind, base in kinds:
            # Match-bonus: if the kind is what the learning priority calls for,
            # bump it. priority_target strings like "test hidden factor
            # hypothesis X" want experiments; "find hidden conditions for X"
            # also wants experiments + uncertainty scans.
            match = 0.0
            if kind == "experiment" and ("hidden" in priority_target or "test" in priority_target):
                match = 0.5
            elif kind == "uncertainty" and "uncertain" in priority_target:
                match = 0.4
            elif kind == "dream" and "collect" in priority_target:
                match = 0.2
            elif kind == "self_modification" and "competence" in priority_target:
                match = 0.3
            age_bonus = 0.0  # fresh enqueues have no age yet; the heap takes care of it.

            priority = (
                self.priority_weights["uncertainty"] * base
                + self.priority_weights["learning_priority_match"] * match
                + self.priority_weights["age"] * age_bonus
            )
            self.scheduler.schedule(
                KernelJob(
                    kind=kind,
                    priority=priority,
                    payload={
                        "source": "replenish",
                        "learning_priority": priority_target,
                        "weakest": weakest,
                    },
                )
            )

    def _normalize_priority_string(self, value: Any) -> str:
        if not value:
            return ""
        return str(value).lower()
