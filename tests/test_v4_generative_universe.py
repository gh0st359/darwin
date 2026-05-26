import tempfile
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.dlm import FaithfulnessValidator
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.types import Goal


class DarwinV4GenerativeUniverseTests(unittest.TestCase):
    def test_ingest_corpus_cli_persists_atoms_and_world_specs(self) -> None:
        from darwin.cli import main

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mini.txt"
            memory = Path(directory) / "memory.sqlite3"
            source.write_text("Force causes acceleration.", encoding="utf-8")

            exit_code = main(["ingest-corpus", "--source", "wikidump", "--path", str(source), "--memory", str(memory)])
            store = PersistentStore(memory)

            self.assertEqual(exit_code, 0)
            self.assertGreaterEqual(store.counts()["knowledge_atoms"], 1)
            self.assertGreaterEqual(store.counts()["world_specs"], 1)

    def test_brain_can_build_v4_adapter_from_persisted_specs_without_legacy_universe(self) -> None:
        from darwin.cli import _build_v4_adapter
        from darwin.knowledge import CorpusIngestor

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mini.txt"
            source.write_text("Force causes acceleration.", encoding="utf-8")
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            CorpusIngestor(store=store).ingest(source, source_type="wikipedia")

            adapter = _build_v4_adapter(store)

            self.assertEqual(adapter.name, "generative_universe")
            self.assertTrue(adapter.possible_actions())
            self.assertNotIn("curtains", " ".join(action.name for action in adapter.possible_actions()))

    def test_server_v4_introspection_commands_expose_knowledge_worlds_and_research(self) -> None:
        from darwin.generative import GenerativeUniverse, GenerativeUniverseAdapter, WorldSpecGenerator
        from darwin.knowledge import CorpusIngestor, KnowledgeGraph
        from darwin.server import DarwinDaemon

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mini.txt"
            source.write_text("Force causes acceleration.", encoding="utf-8")
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            CorpusIngestor(store=store).ingest(source, source_type="wikipedia")
            graph = KnowledgeGraph.from_store(store)
            specs = WorldSpecGenerator().generate(graph)
            for spec in specs:
                store.record_world_spec(spec.to_record(), status="candidate")
            adapter = GenerativeUniverseAdapter(GenerativeUniverse.from_specs(specs))
            darwin = Darwin.from_store(actions=ensure_chat_action(adapter.possible_actions()), store=store, seed=31)
            runtime = DarwinRuntime(
                darwin=darwin,
                adapter=adapter,
                goal=Goal(desired={}),
                store=store,
                interval=100.0,
                state_path=None,
            )
            daemon = DarwinDaemon(runtime, port=0)

            self.assertTrue(any("Force causes acceleration" in line for line in daemon._handle_named_command("/knowledge force")))
            self.assertTrue(any("generated_world_specs" in line for line in daemon._handle_named_command("/worlds")))
            self.assertTrue(any("enabled=False" in line for line in daemon._handle_named_command("/research status")))

    def test_curated_corpus_ingest_creates_provenance_rich_unpromoted_atoms(self) -> None:
        from darwin.knowledge import CorpusIngestor, KnowledgeGraph

        text = """
        == Force ==
        Force is an interaction that changes the motion of an object.
        Force causes acceleration.
        Alias: push, pull

        == Mass ==
        Mass is a quantity measured in kilograms.
        Greater mass resists acceleration.
        """
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mini_wiki.txt"
            source.write_text(text, encoding="utf-8")
            store = PersistentStore(Path(directory) / "memory.sqlite3")

            result = CorpusIngestor(store=store).ingest(source, source_type="wikipedia")
            graph = KnowledgeGraph.from_store(store)

            self.assertGreaterEqual(result.atoms_created, 4)
            force_atoms = graph.search("force")
            self.assertTrue(force_atoms)
            self.assertTrue(all(atom.provenance.source_type == "wikipedia" for atom in force_atoms))
            self.assertTrue(any(atom.kind == "definition" for atom in force_atoms))
            causal = [atom for atom in force_atoms if atom.kind == "causal_hypothesis"]
            self.assertTrue(causal)
            self.assertFalse(any(atom.promoted for atom in causal))
            self.assertEqual(store.counts()["knowledge_atoms"], result.atoms_created)

    def test_definition_containing_change_word_is_not_double_counted_as_causal(self) -> None:
        from darwin.knowledge import CorpusIngestor

        atoms = CorpusIngestor().extract(
            "Force is an interaction that changes motion.\nForce causes acceleration.",
            source_type="wikipedia",
            source_id="fixture",
        )

        causal_texts = [atom.text for atom in atoms if atom.kind == "causal_hypothesis"]
        self.assertEqual(causal_texts, ["Force causes acceleration."])

    def test_generated_world_specs_are_data_sandboxed_and_experimentable(self) -> None:
        from darwin.generative import SandboxedWorldCompiler, WorldSpecGenerator
        from darwin.knowledge import CorpusIngestor, KnowledgeGraph

        text = "Force causes acceleration. Acceleration changes velocity."
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "physics.txt"
            source.write_text(text, encoding="utf-8")
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            CorpusIngestor(store=store).ingest(source, source_type="wikipedia")
            graph = KnowledgeGraph.from_store(store)

            specs = WorldSpecGenerator().generate(graph)
            self.assertTrue(specs)
            spec = specs[0]
            self.assertEqual(spec.trust_level, "sandboxed")
            self.assertFalse(spec.contains_code)

            validation = SandboxedWorldCompiler().validate(spec)
            self.assertTrue(validation.valid, validation.errors)
            adapter = SandboxedWorldCompiler().compile(spec)
            before = adapter.observe()
            action = adapter.possible_actions()[0]
            after, reward = adapter.apply(action)

            self.assertNotEqual(before, after)
            self.assertGreaterEqual(reward, 0.0)
            self.assertTrue(action.name.startswith("generated/"))
            self.assertEqual(adapter.action_metadata(action)["scope"], "world")
            self.assertEqual(adapter.action_metadata(action)["world"], spec.name)

    def test_invalid_generated_world_specs_are_rejected_before_activation(self) -> None:
        from darwin.generative import ActionSpec, RuleSpec, SandboxedWorldCompiler, WorldSpec

        spec = WorldSpec(
            name="unsafe",
            description="attempts unsupported mutation",
            concepts=["unsafe"],
            initial_state={"unsafe.x": 0},
            actions=[
                ActionSpec(
                    name="generated/unsafe_exec",
                    description="not allowed",
                    rules=[RuleSpec(variable="__import__('os').system", operation="eval", operand="rm -rf /")],
                )
            ],
            provenance_ids=["fixture"],
        )

        validation = SandboxedWorldCompiler().validate(spec)

        self.assertFalse(validation.valid)
        self.assertTrue(any("operation" in error or "variable" in error for error in validation.errors))

    def test_v4_runtime_promotes_corpus_hypothesis_only_after_generated_experiment(self) -> None:
        from darwin.generative import GenerativeUniverse, GenerativeUniverseAdapter, WorldSpecGenerator
        from darwin.knowledge import CorpusIngestor, KnowledgeGraph

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "physics.txt"
            source.write_text("Force causes acceleration.", encoding="utf-8")
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            CorpusIngestor(store=store).ingest(source, source_type="wikipedia")
            graph = KnowledgeGraph.from_store(store)
            spec = WorldSpecGenerator().generate(graph)[0]
            store.record_world_spec(spec.to_record(), status="candidate")

            universe = GenerativeUniverse.from_specs([spec])
            adapter = GenerativeUniverseAdapter(universe)
            darwin = Darwin.from_store(
                actions=ensure_chat_action(adapter.possible_actions()),
                store=store,
                seed=22,
            )
            runtime = DarwinRuntime(
                darwin=darwin,
                adapter=adapter,
                goal=Goal(desired={}),
                store=store,
                interval=100.0,
                state_path=None,
            )

            self.assertFalse(any(atom.promoted for atom in KnowledgeGraph.from_store(store).atoms))
            event = runtime.cognition_cycle()
            promoted = KnowledgeGraph.from_store(store).promoted_atoms()

            self.assertEqual(event.kind, "experiment")
            self.assertTrue(promoted)
            self.assertTrue(darwin.causal_model.beliefs())
            self.assertEqual(promoted[0].support_kind, "generated_experiment")

    def test_v4_chat_queries_unified_knowledge_without_curtain_fallback(self) -> None:
        from darwin.generative import GenerativeUniverse, GenerativeUniverseAdapter, WorldSpecGenerator
        from darwin.knowledge import CorpusIngestor, KnowledgeGraph

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "physics.txt"
            source.write_text("Force is an interaction. Force causes acceleration.", encoding="utf-8")
            store = PersistentStore(Path(directory) / "memory.sqlite3")
            CorpusIngestor(store=store).ingest(source, source_type="wikipedia")
            graph = KnowledgeGraph.from_store(store)
            universe = GenerativeUniverse.from_specs(WorldSpecGenerator().generate(graph))
            adapter = GenerativeUniverseAdapter(universe)
            runtime = DarwinRuntime(
                darwin=Darwin.from_store(
                    actions=ensure_chat_action(adapter.possible_actions()),
                    store=store,
                    seed=23,
                ),
                adapter=adapter,
                goal=Goal(desired={}),
                store=store,
                interval=100.0,
                state_path=None,
            )

            response = runtime.chat("What do you know about force?")

            self.assertIn("force", response.lower())
            self.assertNotIn("curtain", response.lower())
            self.assertNotIn("semantic parse is weak", response.lower())
            self.assertEqual(runtime.last_response_plan.mode, "knowledge_answer")

    def test_live_research_is_present_but_disabled_by_default(self) -> None:
        from darwin.research import LiveResearchConfig, LiveResearcher

        researcher = LiveResearcher()

        self.assertFalse(LiveResearchConfig().enabled)
        with self.assertRaises(PermissionError):
            researcher.fetch("https://example.com")
        self.assertEqual(researcher.status()["enabled"], False)

    def test_dlm_payload_for_v4_knowledge_does_not_include_raw_hidden_state(self) -> None:
        from darwin.discourse import ResponsePlan

        plan = ResponsePlan(
            mode="knowledge_answer",
            intent="answer from unified knowledge graph",
            thesis="Answer with provenance-backed knowledge atoms.",
            answer_points=["force causes acceleration"],
            evidence=["knowledge_atom::abc123"],
            confidence=0.6,
        )
        payload = plan.to_dlm_payload()

        self.assertNotIn("hidden_state", payload)
        self.assertNotIn("raw_trace", payload)
        valid, notes = FaithfulnessValidator().validate(plan, "Force causes acceleration.")
        self.assertTrue(valid, notes)


if __name__ == "__main__":
    unittest.main()
