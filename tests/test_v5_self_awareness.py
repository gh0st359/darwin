"""Phase A — Darwin v5 self-awareness substrate tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from darwin.agent import Darwin
from darwin.self_awareness import (
    DARWIN_VERSION,
    REALIZER_KIND_GEMMA,
    REALIZER_KIND_STUB,
    REALIZER_KIND_SYMBOLIC,
    SelfIntrospector,
    SystemIdentity,
)
from darwin.storage import PersistentStore
from darwin.types import Action


class TestSelfIntrospector(unittest.TestCase):
    def _make_darwin(self) -> Darwin:
        actions = [
            Action("noop", description="do nothing"),
            Action("chat_with_user", description="exchange language"),
        ]
        return Darwin(actions=actions, seed=7)

    def test_identity_is_grounded(self) -> None:
        darwin = self._make_darwin()
        introspector = SelfIntrospector(
            darwin,
            kernel_mode="v5",
            realizer_kind=REALIZER_KIND_SYMBOLIC,
            realizer_name="symbolic-v1",
            memory_path="/tmp/test-v5.sqlite3",
        )
        identity = introspector.identity()
        self.assertIsInstance(identity, SystemIdentity)
        self.assertEqual(identity.name, "Darwin")
        self.assertEqual(identity.version, DARWIN_VERSION)
        self.assertEqual(identity.kernel_mode, "v5")
        self.assertEqual(identity.realizer_kind, REALIZER_KIND_SYMBOLIC)
        self.assertEqual(identity.memory_path, "/tmp/test-v5.sqlite3")
        self.assertGreater(identity.pid, 0)
        module_names = {m.name for m in identity.modules}
        self.assertIn("causal_model", module_names)
        self.assertIn("memory", module_names)
        self.assertIn("world_model", module_names)
        self.assertIn("self_model", module_names)

    def test_identity_lines_are_readable(self) -> None:
        darwin = self._make_darwin()
        introspector = SelfIntrospector(darwin, kernel_mode="v5")
        lines = introspector.identity().lines()
        first_block = "\n".join(lines[:10])
        self.assertIn("name=Darwin", first_block)
        self.assertIn("kernel=v5", first_block)
        # Every line should be a string with no template artifacts.
        for line in lines:
            self.assertIsInstance(line, str)
            self.assertNotIn("{", line)
            self.assertNotIn("[SLOT", line)

    def test_capabilities_reports_zeros_on_empty_darwin(self) -> None:
        darwin = self._make_darwin()
        introspector = SelfIntrospector(darwin)
        caps = introspector.capabilities()
        self.assertEqual(caps["total_beliefs"], 0)
        self.assertEqual(caps["confident_beliefs"], 0)
        self.assertEqual(caps["top_competence"], [])

    def test_current_focus_is_string(self) -> None:
        darwin = self._make_darwin()
        introspector = SelfIntrospector(darwin)
        focus = introspector.current_focus()
        self.assertIsInstance(focus, str)
        self.assertGreater(len(focus), 0)

    def test_history_handles_missing_store(self) -> None:
        darwin = self._make_darwin()
        introspector = SelfIntrospector(darwin)
        self.assertEqual(introspector.history(), [])

    def test_history_reads_from_store_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersistentStore(Path(tmp) / "mem.sqlite3")
            actions = [Action("noop")]
            darwin = Darwin(actions=actions, store=store, seed=11)
            introspector = SelfIntrospector(darwin, store=store)
            # Legacy table empty -> empty history list.
            self.assertEqual(introspector.history(limit=5), [])

    def test_darwin_constructs_default_introspector(self) -> None:
        darwin = self._make_darwin()
        self.assertIsNotNone(darwin.introspector)
        identity = darwin.introspector.identity()
        # Without explicit configuration, defaults to v3/stub. CLI swaps it
        # for kernel=v5 with the symbolic realizer kind.
        self.assertEqual(identity.kernel_mode, "v3")
        self.assertEqual(identity.realizer_kind, REALIZER_KIND_STUB)


class TestRealizerKindConstants(unittest.TestCase):
    def test_constants_are_distinct(self) -> None:
        kinds = {REALIZER_KIND_STUB, REALIZER_KIND_GEMMA, REALIZER_KIND_SYMBOLIC}
        self.assertEqual(len(kinds), 3)


if __name__ == "__main__":
    unittest.main()
