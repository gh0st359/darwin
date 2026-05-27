"""Phase B — Darwin v5 rich simulation substrate tests.

Covers:
- ExpressionSpec evaluator and validator
- New rule operations: compute, clamp, if_then
- Derived rules running to fixed point
- Invariants surfacing violations as negative reward + violation list
- counterfactual() not mutating live state
- WorldSpecGenerator multi-hypothesis composition (force + mass -> acceleration)
- Backward compat with v4 add/set/toggle worlds
"""

from __future__ import annotations

import unittest

from darwin.generative import (
    ActionSpec,
    ExpressionSpec,
    GenerativeUniverse,
    RuleSpec,
    SandboxedGeneratedAdapter,
    SandboxedWorldCompiler,
    WorldSpec,
    WorldSpecGenerator,
    _eval_expression,
)
from darwin.knowledge import KnowledgeAtom, KnowledgeGraph, Provenance


def _make_atom(
    kind: str,
    subject: str,
    relation: str,
    obj: str,
    *,
    confidence: float = 0.62,
) -> KnowledgeAtom:
    return KnowledgeAtom(
        kind=kind,
        subject=subject,
        relation=relation,
        object=obj,
        text=f"{subject} {relation} {obj}",
        provenance=Provenance("wikipedia", "test", "deterministic-text-v1", confidence),
        confidence=confidence,
    )


class TestExpressionSpec(unittest.TestCase):
    def test_lit_and_var(self) -> None:
        self.assertEqual(_eval_expression(ExpressionSpec.lit(3), {}), 3)
        self.assertEqual(_eval_expression(ExpressionSpec.var("x"), {"x": 7}), 7)
        self.assertEqual(_eval_expression(ExpressionSpec.var("missing"), {}), 0)

    def test_arithmetic(self) -> None:
        expr = ExpressionSpec(
            kind="add",
            args=[ExpressionSpec.var("a"), ExpressionSpec.var("b"), ExpressionSpec.lit(1)],
        )
        self.assertEqual(_eval_expression(expr, {"a": 2, "b": 3}), 6)
        sub = ExpressionSpec(
            kind="sub",
            args=[ExpressionSpec.lit(10), ExpressionSpec.lit(3), ExpressionSpec.lit(2)],
        )
        self.assertEqual(_eval_expression(sub, {}), 5)
        mul = ExpressionSpec(
            kind="mul",
            args=[ExpressionSpec.var("a"), ExpressionSpec.var("b")],
        )
        self.assertEqual(_eval_expression(mul, {"a": 4, "b": 5}), 20)

    def test_safe_division_by_zero(self) -> None:
        div = ExpressionSpec(
            kind="div",
            args=[ExpressionSpec.lit(10), ExpressionSpec.var("denom")],
        )
        self.assertEqual(_eval_expression(div, {"denom": 0}), 0.0)
        self.assertEqual(_eval_expression(div, {"denom": 2}), 5.0)

    def test_comparison_and_logic(self) -> None:
        gt = ExpressionSpec.cmp(">", ExpressionSpec.var("x"), ExpressionSpec.lit(0))
        self.assertTrue(_eval_expression(gt, {"x": 5}))
        self.assertFalse(_eval_expression(gt, {"x": -1}))
        and_expr = ExpressionSpec(
            kind="and",
            args=[
                ExpressionSpec.cmp(">", ExpressionSpec.var("x"), ExpressionSpec.lit(0)),
                ExpressionSpec.cmp("<", ExpressionSpec.var("x"), ExpressionSpec.lit(10)),
            ],
        )
        self.assertTrue(_eval_expression(and_expr, {"x": 5}))
        self.assertFalse(_eval_expression(and_expr, {"x": 50}))

    def test_serialization_roundtrip(self) -> None:
        expr = ExpressionSpec(
            kind="div",
            args=[
                ExpressionSpec.var("force.value"),
                ExpressionSpec(
                    kind="max",
                    args=[ExpressionSpec.var("mass.value"), ExpressionSpec.lit(1.0)],
                ),
            ],
        )
        record = expr.to_record()
        restored = ExpressionSpec.from_record(record)
        self.assertEqual(restored.to_record(), record)
        self.assertEqual(
            _eval_expression(restored, {"force.value": 6, "mass.value": 3}),
            2.0,
        )


class TestRuleOperationsValidator(unittest.TestCase):
    def test_compute_rule_accepts_expression(self) -> None:
        spec = WorldSpec(
            name="generated/test_compute",
            description="",
            concepts=["x"],
            initial_state={"x.value": 0.0, "x.derived": 0.0},
            actions=[
                ActionSpec(
                    name="generated/poke",
                    description="",
                    rules=[
                        RuleSpec(variable="x.value", operation="add", operand=1.0),
                        RuleSpec(
                            variable="x.derived",
                            operation="compute",
                            expression=ExpressionSpec(
                                kind="mul",
                                args=[ExpressionSpec.var("x.value"), ExpressionSpec.lit(2.0)],
                            ),
                        ),
                    ],
                )
            ],
            provenance_ids=[],
        )
        validation = SandboxedWorldCompiler().validate(spec)
        self.assertTrue(validation.valid, validation.errors)

    def test_compute_rule_requires_expression(self) -> None:
        spec = WorldSpec(
            name="generated/bad_compute",
            description="",
            concepts=[],
            initial_state={"x.value": 0.0},
            actions=[
                ActionSpec(
                    name="generated/poke",
                    description="",
                    rules=[RuleSpec(variable="x.value", operation="compute")],
                )
            ],
            provenance_ids=[],
        )
        validation = SandboxedWorldCompiler().validate(spec)
        self.assertFalse(validation.valid)
        self.assertTrue(any("expression" in err for err in validation.errors))

    def test_clamp_rule_requires_min_and_max(self) -> None:
        spec = WorldSpec(
            name="generated/bad_clamp",
            description="",
            concepts=[],
            initial_state={"x.value": 0.0},
            actions=[
                ActionSpec(
                    name="generated/poke",
                    description="",
                    rules=[RuleSpec(variable="x.value", operation="clamp", operand={"min": 0})],
                )
            ],
            provenance_ids=[],
        )
        validation = SandboxedWorldCompiler().validate(spec)
        self.assertFalse(validation.valid)
        self.assertTrue(any("min" in err and "max" in err for err in validation.errors))


class TestDerivedAndInvariants(unittest.TestCase):
    def _build_force_mass_world(self) -> WorldSpec:
        return WorldSpec(
            name="generated/composite_acceleration",
            description="F = m * a",
            concepts=["force", "mass", "acceleration"],
            initial_state={
                "force.value": 0.0,
                "mass.value": 1.0,
                "composite.acceleration": 0.0,
            },
            actions=[
                ActionSpec(
                    name="generated/apply_force",
                    description="",
                    rules=[RuleSpec(variable="force.value", operation="add", operand=1.0)],
                ),
                ActionSpec(
                    name="generated/apply_mass",
                    description="",
                    rules=[RuleSpec(variable="mass.value", operation="add", operand=1.0)],
                ),
            ],
            provenance_ids=[],
            derived=[
                RuleSpec(
                    variable="composite.acceleration",
                    operation="compute",
                    derived=True,
                    expression=ExpressionSpec(
                        kind="div",
                        args=[
                            ExpressionSpec.var("force.value"),
                            ExpressionSpec(
                                kind="max",
                                args=[ExpressionSpec.var("mass.value"), ExpressionSpec.lit(1.0)],
                            ),
                        ],
                    ),
                )
            ],
            invariants=[
                ExpressionSpec.cmp(">=", ExpressionSpec.var("mass.value"), ExpressionSpec.lit(0.0)),
            ],
        )

    def test_compile_force_mass_world(self) -> None:
        spec = self._build_force_mass_world()
        validation = SandboxedWorldCompiler().validate(spec)
        self.assertTrue(validation.valid, validation.errors)
        adapter = SandboxedWorldCompiler().compile(spec)
        # Initial derived value is 0 / max(1, 1) = 0; it only changes after
        # the first action runs the derived rule.
        self.assertEqual(adapter.observe()["composite.acceleration"], 0.0)

    def test_force_action_drives_derived_acceleration(self) -> None:
        adapter = SandboxedWorldCompiler().compile(self._build_force_mass_world())
        force_action = next(a for a in adapter.possible_actions() if a.name == "generated/apply_force")
        state, _reward = adapter.apply(force_action)
        # mass.value=1, force.value=1, so derived acceleration = 1/1 = 1.
        self.assertAlmostEqual(state["composite.acceleration"], 1.0)
        state, _reward = adapter.apply(force_action)
        self.assertAlmostEqual(state["composite.acceleration"], 2.0)

    def test_mass_action_reduces_acceleration(self) -> None:
        adapter = SandboxedWorldCompiler().compile(self._build_force_mass_world())
        force_action = next(a for a in adapter.possible_actions() if a.name == "generated/apply_force")
        mass_action = next(a for a in adapter.possible_actions() if a.name == "generated/apply_mass")
        for _ in range(6):
            adapter.apply(force_action)
        before = adapter.observe()["composite.acceleration"]
        # Double the mass (1 -> 2) by applying once.
        state, _reward = adapter.apply(mass_action)
        self.assertLess(state["composite.acceleration"], before)
        self.assertAlmostEqual(state["composite.acceleration"], 6.0 / 2.0)

    def test_counterfactual_does_not_mutate(self) -> None:
        adapter = SandboxedWorldCompiler().compile(self._build_force_mass_world())
        force_action = next(a for a in adapter.possible_actions() if a.name == "generated/apply_force")
        before = adapter.observe()
        result = adapter.counterfactual(force_action)
        self.assertEqual(adapter.observe(), before)
        self.assertIn("force.value", result["delta"])
        self.assertEqual(result["after"]["force.value"], 1.0)

    def test_invariant_violation_emits_negative_reward(self) -> None:
        spec = WorldSpec(
            name="generated/invariant_world",
            description="",
            concepts=["x"],
            initial_state={"x.value": 0.0},
            actions=[
                ActionSpec(
                    name="generated/drop",
                    description="",
                    rules=[RuleSpec(variable="x.value", operation="add", operand=-5.0)],
                )
            ],
            provenance_ids=[],
            invariants=[
                ExpressionSpec.cmp(">=", ExpressionSpec.var("x.value"), ExpressionSpec.lit(0.0)),
            ],
        )
        adapter = SandboxedWorldCompiler().compile(spec)
        drop = adapter.possible_actions()[0]
        _state, reward = adapter.apply(drop)
        self.assertLess(reward, 0.0)
        self.assertEqual(len(adapter.last_invariant_violations), 1)


class TestWorldSpecGeneratorComposite(unittest.TestCase):
    def test_force_mass_acceleration_composite(self) -> None:
        atoms = [
            _make_atom("causal_hypothesis", "Force", "causes", "acceleration"),
            _make_atom("causal_hypothesis", "Mass", "resists", "acceleration"),
        ]
        graph = KnowledgeGraph(atoms)
        specs = WorldSpecGenerator().generate(graph)
        composite = next((s for s in specs if "composite" in s.name), None)
        self.assertIsNotNone(composite, "expected a composite world")
        action_names = {action.name for action in composite.actions}
        self.assertIn("generated/apply_force", action_names)
        self.assertIn("generated/apply_mass", action_names)
        self.assertEqual(len(composite.derived), 1)
        self.assertGreaterEqual(len(composite.invariants), 1)
        # All four provenance ids (causes + resists) should be in the spec.
        self.assertEqual(len(composite.provenance_ids), 2)

    def test_single_hypothesis_falls_back_to_simple_world(self) -> None:
        atoms = [_make_atom("causal_hypothesis", "Energy", "causes", "change")]
        specs = WorldSpecGenerator().generate(KnowledgeGraph(atoms))
        self.assertEqual(len(specs), 1)
        self.assertNotIn("composite", specs[0].name)
        # Backward compat: still add/add rules.
        rules = specs[0].actions[0].rules
        self.assertTrue(all(rule.operation == "add" for rule in rules))


class TestBackwardCompatV4Worlds(unittest.TestCase):
    def test_v4_add_world_still_compiles_and_runs(self) -> None:
        spec = WorldSpec(
            name="generated/legacy_world",
            description="",
            concepts=["x"],
            initial_state={"x.value": 0.0, "x.flag": False},
            actions=[
                ActionSpec(
                    name="generated/poke",
                    description="",
                    rules=[
                        RuleSpec(variable="x.value", operation="add", operand=1.0),
                        RuleSpec(variable="x.flag", operation="toggle", operand=None),
                    ],
                )
            ],
            provenance_ids=[],
        )
        adapter = SandboxedWorldCompiler().compile(spec)
        state, reward = adapter.apply(adapter.possible_actions()[0])
        self.assertEqual(state["x.value"], 1.0)
        self.assertTrue(state["x.flag"])
        self.assertGreater(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
