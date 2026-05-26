from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KernelJob:
    kind: str
    priority: float
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class KernelMetrics:
    jobs_scheduled: int = 0
    jobs_completed: int = 0
    repeated_loop_redirects: int = 0
    experiments_started: int = 0
    useful_beliefs: int = 0
    dlm_rejections: int = 0
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
            "experiments_per_minute": 60.0 * self.experiments_started / elapsed,
            "useful_beliefs_per_hour": 3600.0 * self.useful_beliefs / elapsed,
        }


class ActorScheduler:
    """Small local-first v4 actor scheduler skeleton.

    It gives the v4 kernel a concrete job surface without replacing the
    existing daemon loops in one risky jump. Jobs are prioritized and
    can later move to process/GPU workers behind this same interface.
    """

    def __init__(self, workers: str = "auto", accelerator: str = "auto") -> None:
        self.workers = os.cpu_count() if workers == "auto" else max(1, int(workers))
        self.accelerator = accelerator
        self.metrics = KernelMetrics()
        self._jobs: list[KernelJob] = []

    def schedule(self, job: KernelJob) -> None:
        self._jobs.append(job)
        self._jobs.sort(key=lambda item: (item.priority, -item.created_at), reverse=True)
        self.metrics.jobs_scheduled += 1

    def pop_next(self) -> KernelJob | None:
        if not self._jobs:
            return None
        return self._jobs.pop(0)

    def complete(self, job: KernelJob) -> None:
        self.metrics.jobs_completed += 1
        if job.kind == "experiment":
            self.metrics.experiments_started += 1
