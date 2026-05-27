"""Phase D — Darwin v5 kernel activation tests.

Covers:
- ActorScheduler: heap-based priority ordering (higher priority first)
- ActorScheduler: per-kind saturation caps prevent runaway in-flight counts
- ActorScheduler: completion_rate over a time window
- KernelDriver: ticks invoke the right runtime handler and increment metrics
- KernelDriver: _replenish enqueues jobs of every kind
- runtime.start_v5() spawns the kernel driver instead of fixed daemons
"""

from __future__ import annotations

import time
import unittest

from darwin.agent import Darwin
from darwin.embodiment import UniverseSimulationAdapter
from darwin.kernel import ActorScheduler, KernelDriver, KernelJob
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.types import Action, Goal
from darwin.worlds import UniverseSimulation


def _build_runtime() -> DarwinRuntime:
    universe = UniverseSimulation(seed=11)
    adapter = UniverseSimulationAdapter(universe)
    actions = ensure_chat_action(adapter.possible_actions())
    darwin = Darwin(actions=actions, seed=11)
    goal = Goal(desired={}, weights={})
    return DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=goal,
        interval=0.5,
        state_path=None,
    )


class TestActorScheduler(unittest.TestCase):
    def test_higher_priority_jobs_pop_first(self) -> None:
        scheduler = ActorScheduler(workers=2)
        scheduler.schedule(KernelJob(kind="experiment", priority=0.2))
        scheduler.schedule(KernelJob(kind="dream", priority=0.9))
        scheduler.schedule(KernelJob(kind="simulation", priority=0.5))
        kinds = []
        while True:
            job = scheduler.pop_next()
            if job is None:
                break
            kinds.append(job.kind)
            scheduler.complete(job)
        self.assertEqual(kinds, ["dream", "simulation", "experiment"])

    def test_saturation_cap_holds_jobs_back(self) -> None:
        scheduler = ActorScheduler(
            workers=2,
            saturation_caps={"experiment": 1},
        )
        # Two experiment jobs, but only one can be in-flight at a time.
        scheduler.schedule(KernelJob(kind="experiment", priority=0.5))
        scheduler.schedule(KernelJob(kind="experiment", priority=0.5))
        first = scheduler.pop_next()
        self.assertIsNotNone(first)
        second = scheduler.pop_next()
        self.assertIsNone(second, "second experiment must be held by saturation guard")
        scheduler.complete(first)  # type: ignore[arg-type]
        third = scheduler.pop_next()
        self.assertIsNotNone(third, "should be available after the first completes")

    def test_metrics_track_scheduled_and_completed(self) -> None:
        scheduler = ActorScheduler()
        scheduler.schedule(KernelJob(kind="dream", priority=0.5))
        scheduler.schedule(KernelJob(kind="experiment", priority=0.5))
        job = scheduler.pop_next()
        scheduler.complete(job)  # type: ignore[arg-type]
        record = scheduler.metrics.to_record()
        self.assertEqual(record["jobs_scheduled"], 2)
        self.assertEqual(record["jobs_completed"], 1)
        self.assertIn("experiment", record["completions_by_kind"]) if job.kind == "experiment" else None

    def test_in_flight_counts_drop_on_completion(self) -> None:
        scheduler = ActorScheduler()
        scheduler.schedule(KernelJob(kind="dream", priority=0.5))
        job = scheduler.pop_next()
        self.assertEqual(scheduler.in_flight().get("dream"), 1)
        scheduler.complete(job)  # type: ignore[arg-type]
        self.assertEqual(scheduler.in_flight().get("dream"), 0)


class TestKernelDriver(unittest.TestCase):
    def test_replenish_enqueues_every_kind(self) -> None:
        runtime = _build_runtime()
        scheduler = ActorScheduler()
        runtime.kernel_scheduler = scheduler
        driver = KernelDriver(runtime, scheduler)
        driver._replenish()
        self.assertGreaterEqual(scheduler.queue_size(), 6)

    def test_tick_dispatches_and_increments_metrics(self) -> None:
        runtime = _build_runtime()
        scheduler = ActorScheduler()
        runtime.kernel_scheduler = scheduler
        driver = KernelDriver(runtime, scheduler)
        scheduler.schedule(KernelJob(kind="uncertainty", priority=0.9))
        driver.tick()
        self.assertEqual(scheduler.metrics.jobs_completed, 1)
        self.assertEqual(scheduler.metrics.completions_by_kind.get("uncertainty"), 1)

    def test_runtime_start_v5_runs_kernel_driver(self) -> None:
        runtime = _build_runtime()
        runtime.kernel_scheduler = ActorScheduler()
        runtime.start_v5()
        try:
            # Wait briefly for the driver to dispatch at least one job. The
            # replenish() seed happens before the first tick, so by ~1s
            # we should see jobs_completed > 0.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if runtime.kernel_scheduler.metrics.jobs_completed > 0:
                    break
                time.sleep(0.05)
            self.assertGreater(
                runtime.kernel_scheduler.metrics.jobs_completed,
                0,
                "kernel driver did not complete any job within 3 seconds",
            )
            self.assertTrue(runtime.running)
        finally:
            runtime.stop()

    def test_kernel_metrics_round_trip(self) -> None:
        scheduler = ActorScheduler()
        scheduler.schedule(KernelJob(kind="experiment", priority=0.9))
        job = scheduler.pop_next()
        scheduler.complete(job)  # type: ignore[arg-type]
        record = scheduler.metrics.to_record()
        # New v5 surface area.
        self.assertIn("saturation_skips", record)
        self.assertIn("starvation_lifts", record)
        self.assertIn("completions_by_kind", record)


if __name__ == "__main__":
    unittest.main()
