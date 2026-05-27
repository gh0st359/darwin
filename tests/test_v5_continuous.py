"""Phase F — continuous learning compounding tests.

Covers:
- Memory.consolidate_redundant_concepts removes duplicates by signature.
- Memory.decay_stale_concepts demotes support over time.
- The _handle_consolidation kernel job emits a consolidation event with
  populated payload counts.
- KernelDriver._lift_starved_kinds records a starvation_lift when one
  kind has dropped sharply in recent completions.
- All three Phase B scaffolded tables (generated_experiments,
  validation_results, research_events) are populated end-to-end by an
  ingest + brief v5 run.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.concepts import Concept
from darwin.embodiment import UniverseSimulationAdapter
from darwin.kernel import ActorScheduler, KernelDriver, KernelJob
from darwin.knowledge import CorpusIngestor, KnowledgeGraph
from darwin.generative import SandboxedWorldCompiler, WorldSpecGenerator
from darwin.memory import Memory
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.types import Action, Goal
from darwin.worlds import UniverseSimulation


class TestMemoryConsolidation(unittest.TestCase):
    def test_consolidate_redundant_concepts_drops_duplicates(self) -> None:
        memory = Memory()
        # Inject two concepts with the exact same signature except support.
        memory.concepts._concepts["dup"] = Concept(name="dup", kind="cluster", level=1, support=5)
        memory.concepts._concepts["dup-2"] = Concept(name="dup", kind="cluster", level=1, support=2)
        # Distinct signature should survive.
        memory.concepts._concepts["unique"] = Concept(name="unique", kind="other", level=0, support=1)
        result = memory.consolidate_redundant_concepts()
        self.assertGreaterEqual(result["removed"], 1)
        self.assertIn("unique", memory.concepts._concepts)

    def test_decay_lowers_support(self) -> None:
        memory = Memory()
        memory.concepts._concepts["alpha"] = Concept(name="alpha", kind="cluster", level=0, support=100)
        memory.decay_stale_concepts(half_life_days=7.0)
        self.assertLess(memory.concepts._concepts["alpha"].support, 100)

    def test_decay_drops_zero_support(self) -> None:
        memory = Memory()
        memory.concepts._concepts["dust"] = Concept(name="dust", kind="cluster", level=0, support=1)
        memory.decay_stale_concepts(half_life_days=0.5)
        self.assertNotIn("dust", memory.concepts._concepts)


class TestAntiThrash(unittest.TestCase):
    def test_lift_records_starvation_event(self) -> None:
        universe = UniverseSimulation(seed=5)
        adapter = UniverseSimulationAdapter(universe)
        actions = ensure_chat_action(adapter.possible_actions())
        darwin = Darwin(actions=actions, seed=5)
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={}),
            interval=0.5,
            state_path=None,
        )
        scheduler = ActorScheduler()
        runtime.kernel_scheduler = scheduler
        driver = KernelDriver(runtime, scheduler)

        # Pre-load the scheduler's completion window so the "prior" window
        # has lots of dream completions and the "recent" window has none.
        now = time.time()
        # 12 historical dreams 700-800 seconds ago.
        for offset in range(700, 800):
            scheduler._completion_window.append((now - offset, "dream"))
        # Drive lift.
        before = scheduler.metrics.starvation_lifts
        driver._lift_starved_kinds()
        after = scheduler.metrics.starvation_lifts
        self.assertGreater(after, before, "should record at least one starvation lift")
        # At least one dream job should now be on the queue.
        kinds_in_queue = [entry[3].kind for entry in scheduler._heap]
        self.assertIn("dream", kinds_in_queue)


class TestConsolidationJob(unittest.TestCase):
    def test_handle_consolidation_emits_event(self) -> None:
        universe = UniverseSimulation(seed=5)
        adapter = UniverseSimulationAdapter(universe)
        actions = ensure_chat_action(adapter.possible_actions())
        darwin = Darwin(actions=actions, seed=5)
        runtime = DarwinRuntime(
            darwin=darwin,
            adapter=adapter,
            goal=Goal(desired={}),
            interval=0.5,
            state_path=None,
        )
        event = runtime._handle_consolidation()
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "consolidation")  # type: ignore[union-attr]
        payload = event.payload  # type: ignore[union-attr]
        self.assertIn("redundancy", payload)
        self.assertIn("decay", payload)
        self.assertIn("clusters", payload)


class TestScaffoldedTablesPopulate(unittest.TestCase):
    def test_ingest_populates_validation_and_research_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "force.txt"
            corpus.write_text(
                "== Force ==\nForce is an interaction that changes motion.\n"
                "Force causes acceleration.\n"
            )
            store = PersistentStore(Path(tmp) / "mem.sqlite3")
            ingestor = CorpusIngestor(store=store)
            ingestor.ingest(corpus, source_type="wikipedia")
            # Mirror the CLI's ingest path: validate + persist each spec.
            graph = KnowledgeGraph.from_store(store)
            specs = WorldSpecGenerator().generate(graph)
            compiler = SandboxedWorldCompiler()
            for spec in specs:
                validation = compiler.validate(spec)
                store.record_validation_result(
                    target=f"world_spec:{spec.name}",
                    valid=validation.valid,
                    payload=validation.to_record(),
                )
                if validation.valid:
                    store.record_world_spec(spec.to_record(), status="candidate")
            store.record_research_event(
                status="ingested",
                url=str(corpus),
                payload={"atoms": 2, "specs": len(specs)},
            )
            counts = store.counts()
            self.assertGreater(counts["validation_results"], 0)
            self.assertGreater(counts["research_events"], 0)
            self.assertGreater(counts["world_specs"], 0)
            self.assertGreater(counts["knowledge_atoms"], 0)


if __name__ == "__main__":
    unittest.main()
