from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from darwin.knowledge import KnowledgeAtom, KnowledgeGraph
from darwin.types import Action, State


# ---------------------------------------------------------------------------
# ExpressionSpec — typed AST for rich generated worlds (v5).
#
# An ExpressionSpec is a data-only tree of arithmetic/comparison/logical
# operations over named state variables and literals. The evaluator below
# never calls `eval` or `exec`; it walks the AST. The compiler's validator
# rejects any tree that contains an unknown kind, an unknown operator, or
# (optionally) variables outside an allowed set.
# ---------------------------------------------------------------------------


EXPR_KINDS = frozenset({"lit", "var", "add", "sub", "mul", "div", "neg", "cmp", "and", "or", "not", "min", "max"})
CMP_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})


@dataclass
class ExpressionSpec:
    kind: str
    value: Any = None
    variable: str = ""
    op: str = ""
    args: list["ExpressionSpec"] = field(default_factory=list)

    # ----- factory helpers ------------------------------------------------

    @classmethod
    def lit(cls, value: Any) -> "ExpressionSpec":
        return cls(kind="lit", value=value)

    @classmethod
    def var(cls, name: str) -> "ExpressionSpec":
        return cls(kind="var", variable=name)

    @classmethod
    def binop(cls, kind: str, left: "ExpressionSpec", right: "ExpressionSpec") -> "ExpressionSpec":
        return cls(kind=kind, args=[left, right])

    @classmethod
    def cmp(cls, op: str, left: "ExpressionSpec", right: "ExpressionSpec") -> "ExpressionSpec":
        return cls(kind="cmp", op=op, args=[left, right])

    # ----- (de)serialization ---------------------------------------------

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"kind": self.kind}
        if self.kind == "lit":
            record["value"] = self.value
        elif self.kind == "var":
            record["variable"] = self.variable
        else:
            record["args"] = [arg.to_record() for arg in self.args]
            if self.op:
                record["op"] = self.op
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ExpressionSpec":
        kind = str(record.get("kind", ""))
        if kind == "lit":
            return cls(kind="lit", value=record.get("value"))
        if kind == "var":
            return cls(kind="var", variable=str(record.get("variable", "")))
        args = [cls.from_record(item) for item in record.get("args", [])]
        return cls(kind=kind, op=str(record.get("op", "")), args=args)


def _expression_variables(expr: ExpressionSpec) -> set[str]:
    """Walk an expression and collect every referenced variable name."""

    if expr.kind == "var":
        return {expr.variable}
    referenced: set[str] = set()
    for arg in expr.args:
        referenced.update(_expression_variables(arg))
    return referenced


def _validate_expression(expr: ExpressionSpec, errors: list[str]) -> None:
    if expr.kind not in EXPR_KINDS:
        errors.append(f"unknown expression kind: {expr.kind}")
        return
    if expr.kind == "lit":
        if not isinstance(expr.value, (int, float, bool)):
            errors.append(f"literal must be number or bool, got {type(expr.value).__name__}")
        return
    if expr.kind == "var":
        if not expr.variable:
            errors.append("var expression must name a variable")
        elif not SandboxedWorldCompiler.variable_pattern.match(expr.variable):
            errors.append(f"invalid variable in expression: {expr.variable}")
        return
    if expr.kind == "cmp":
        if expr.op not in CMP_OPS:
            errors.append(f"unsupported comparison operator: {expr.op}")
        if len(expr.args) != 2:
            errors.append("cmp expression requires exactly 2 args")
    elif expr.kind in {"and", "or"}:
        if len(expr.args) < 1:
            errors.append(f"{expr.kind} requires at least 1 arg")
    elif expr.kind == "not":
        if len(expr.args) != 1:
            errors.append("not expression requires exactly 1 arg")
    elif expr.kind == "neg":
        if len(expr.args) != 1:
            errors.append("neg expression requires exactly 1 arg")
    elif expr.kind in {"add", "sub", "mul", "div", "min", "max"}:
        if len(expr.args) < 2:
            errors.append(f"{expr.kind} requires at least 2 args")
    for arg in expr.args:
        _validate_expression(arg, errors)


def _eval_expression(expr: ExpressionSpec, state: Mapping[str, Any]) -> Any:
    """Evaluate an expression against the given state. Missing variables -> 0."""

    kind = expr.kind
    if kind == "lit":
        return expr.value
    if kind == "var":
        return state.get(expr.variable, 0)
    if kind == "neg":
        return -_to_number(_eval_expression(expr.args[0], state))
    if kind == "not":
        return not _to_bool(_eval_expression(expr.args[0], state))
    if kind == "and":
        return all(_to_bool(_eval_expression(arg, state)) for arg in expr.args)
    if kind == "or":
        return any(_to_bool(_eval_expression(arg, state)) for arg in expr.args)
    if kind == "cmp":
        left = _eval_expression(expr.args[0], state)
        right = _eval_expression(expr.args[1], state)
        if expr.op == "==":
            return left == right
        if expr.op == "!=":
            return left != right
        left_n = _to_number(left)
        right_n = _to_number(right)
        if expr.op == "<":
            return left_n < right_n
        if expr.op == "<=":
            return left_n <= right_n
        if expr.op == ">":
            return left_n > right_n
        if expr.op == ">=":
            return left_n >= right_n
        raise ValueError(f"unsupported cmp op: {expr.op}")
    # Arithmetic n-ary forms.
    values = [_to_number(_eval_expression(arg, state)) for arg in expr.args]
    if kind == "add":
        return sum(values)
    if kind == "sub":
        result = values[0]
        for v in values[1:]:
            result -= v
        return result
    if kind == "mul":
        result = values[0]
        for v in values[1:]:
            result *= v
        return result
    if kind == "div":
        result = values[0]
        for v in values[1:]:
            # Safe division: zero denominator returns 0.0. Worlds that need
            # a guard can encode it explicitly with clamp / max.
            if v == 0:
                return 0.0
            result /= v
        return result
    if kind == "min":
        return min(values)
    if kind == "max":
        return max(values)
    raise ValueError(f"unknown expression kind: {kind}")


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return False
    return bool(value)


# ---------------------------------------------------------------------------
# RuleSpec — supports v4 ops (add/set/toggle) and new v5 ops (compute/clamp/
# if_then). v5 ops carry their data either in `operand` (clamp dict, etc.)
# or in `expression` (an ExpressionSpec).
# ---------------------------------------------------------------------------


@dataclass
class RuleSpec:
    variable: str
    operation: str
    operand: Any = None
    expression: ExpressionSpec | None = None
    derived: bool = False
    then_rules: list["RuleSpec"] = field(default_factory=list)
    else_rules: list["RuleSpec"] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "variable": self.variable,
            "operation": self.operation,
            "operand": self.operand,
        }
        if self.expression is not None:
            record["expression"] = self.expression.to_record()
        if self.derived:
            record["derived"] = True
        if self.then_rules:
            record["then_rules"] = [rule.to_record() for rule in self.then_rules]
        if self.else_rules:
            record["else_rules"] = [rule.to_record() for rule in self.else_rules]
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RuleSpec":
        expression_record = record.get("expression")
        return cls(
            variable=str(record.get("variable", "")),
            operation=str(record.get("operation", "")),
            operand=record.get("operand"),
            expression=ExpressionSpec.from_record(expression_record) if expression_record else None,
            derived=bool(record.get("derived", False)),
            then_rules=[cls.from_record(item) for item in record.get("then_rules", [])],
            else_rules=[cls.from_record(item) for item in record.get("else_rules", [])],
        )


@dataclass
class ActionSpec:
    name: str
    description: str
    rules: list[RuleSpec]
    cost: float = 0.01
    vocabulary: list[str] = field(default_factory=list)
    provenance_ids: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_record() for rule in self.rules],
            "cost": self.cost,
            "vocabulary": list(self.vocabulary),
            "provenance_ids": list(self.provenance_ids),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ActionSpec":
        return cls(
            name=str(record.get("name", "")),
            description=str(record.get("description", "")),
            rules=[RuleSpec.from_record(item) for item in record.get("rules", [])],
            cost=float(record.get("cost", 0.01)),
            vocabulary=list(record.get("vocabulary", [])),
            provenance_ids=list(record.get("provenance_ids", [])),
        )


@dataclass
class WorldSpec:
    name: str
    description: str
    concepts: list[str]
    initial_state: State
    actions: list[ActionSpec]
    provenance_ids: list[str]
    trust_level: str = "sandboxed"
    contains_code: bool = False
    step_budget: int = 10_000
    # v5 additions: derived recompute rules + invariant booleans.
    derived: list[RuleSpec] = field(default_factory=list)
    invariants: list[ExpressionSpec] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "concepts": list(self.concepts),
            "initial_state": dict(self.initial_state),
            "actions": [action.to_record() for action in self.actions],
            "provenance_ids": list(self.provenance_ids),
            "trust_level": self.trust_level,
            "contains_code": self.contains_code,
            "step_budget": self.step_budget,
            "derived": [rule.to_record() for rule in self.derived],
            "invariants": [inv.to_record() for inv in self.invariants],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "WorldSpec":
        return cls(
            name=str(record.get("name", "")),
            description=str(record.get("description", "")),
            concepts=list(record.get("concepts", [])),
            initial_state=dict(record.get("initial_state", {})),
            actions=[ActionSpec.from_record(item) for item in record.get("actions", [])],
            provenance_ids=list(record.get("provenance_ids", [])),
            trust_level=str(record.get("trust_level", "sandboxed")),
            contains_code=bool(record.get("contains_code", False)),
            step_budget=int(record.get("step_budget", 10_000)),
            derived=[RuleSpec.from_record(item) for item in record.get("derived", [])],
            invariants=[ExpressionSpec.from_record(item) for item in record.get("invariants", [])],
        )


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors)}


class WorldSpecGenerator:
    """Build sandboxed WorldSpecs from a KnowledgeGraph.

    v4 produced one world per causal hypothesis, with a single integer-bump
    action. v5 also produces *composite* worlds when multiple hypotheses
    share a variable (so "Force causes acceleration" and "Mass resists
    acceleration" become one world with two actions and a derived rule
    relating force, mass, and acceleration). Definitions and relations
    contribute invariants and concepts.
    """

    def generate(self, graph: KnowledgeGraph, limit: int = 12) -> list[WorldSpec]:
        causal_atoms = graph.causal_hypotheses()[:limit]
        if not causal_atoms:
            return []

        # Group hypotheses by the object-slug (the variable they target).
        # When 2+ hypotheses share an object, build a composite world.
        by_object: dict[str, list[KnowledgeAtom]] = {}
        for atom in causal_atoms:
            object_slug = self._slug(atom.object.split()[0] if atom.object else "effect")
            by_object.setdefault(object_slug, []).append(atom)

        specs: list[WorldSpec] = []
        used: set[str] = set()
        for object_slug, atoms in by_object.items():
            if len(atoms) >= 2:
                composite = self._spec_from_composite(object_slug, atoms, graph)
                if composite is not None:
                    specs.append(composite)
                    for atom in atoms:
                        used.add(atom.atom_id)

        # Anything not absorbed into a composite world becomes its own spec
        # via the v4 path. This keeps backward behavior intact for corpora
        # whose hypotheses don't overlap.
        for atom in causal_atoms:
            if atom.atom_id in used:
                continue
            specs.append(self._spec_from_hypothesis(atom))

        return specs

    # -- spec building -----------------------------------------------------

    def _spec_from_hypothesis(self, atom: KnowledgeAtom) -> WorldSpec:
        subject = self._slug(atom.subject)
        obj = self._slug(atom.object.split()[0] if atom.object else "effect")
        world_name = f"generated/{subject}_{obj}"
        variable = f"{subject}.{obj}"
        action_name = f"generated/apply_{subject}"
        concepts = sorted({atom.subject.lower(), atom.object.lower(), atom.relation.lower()})
        return WorldSpec(
            name=world_name,
            description=f"Sandbox generated from corpus hypothesis: {atom.text}",
            concepts=concepts,
            initial_state={variable: 0.0, f"{subject}.interventions": 0},
            actions=[
                ActionSpec(
                    name=action_name,
                    description=f"Intervene on {atom.subject} to test {atom.object}.",
                    rules=[
                        RuleSpec(variable=variable, operation="add", operand=1.0),
                        RuleSpec(variable=f"{subject}.interventions", operation="add", operand=1),
                    ],
                    cost=0.02,
                    vocabulary=concepts,
                    provenance_ids=[atom.atom_id],
                )
            ],
            provenance_ids=[atom.atom_id],
        )

    def _spec_from_composite(
        self,
        object_slug: str,
        atoms: list[KnowledgeAtom],
        graph: KnowledgeGraph,
    ) -> WorldSpec | None:
        """Build a composite world for multiple hypotheses targeting one var."""

        # Group atoms by the relation kind: "causes" vs "resists" or other.
        # The first subject we see for "causes" becomes the numerator; the
        # first "resists" / "is_inverse" subject becomes a denominator that
        # the target variable is divided by. This is the simplest data-only
        # encoding that produces emergent F=m/a-style relations.
        positive: list[KnowledgeAtom] = []
        negative: list[KnowledgeAtom] = []
        for atom in atoms:
            relation = atom.relation.lower()
            if relation in {"resists", "inhibits", "reduces", "decreases"}:
                negative.append(atom)
            else:
                positive.append(atom)
        if not positive:
            return None

        provenance_ids: list[str] = [atom.atom_id for atom in atoms]
        primary_subject = self._slug(positive[0].subject)
        secondary_subject = self._slug(negative[0].subject) if negative else None

        target_variable = f"composite.{object_slug}"
        primary_value_var = f"{primary_subject}.value"
        primary_count_var = f"{primary_subject}.interventions"
        actions: list[ActionSpec] = []
        initial_state: State = {
            target_variable: 0.0,
            primary_value_var: 0.0,
            primary_count_var: 0,
        }
        concepts: set[str] = set()
        for atom in atoms:
            concepts.update({atom.subject.lower(), atom.object.lower(), atom.relation.lower()})
        actions.append(
            ActionSpec(
                name=f"generated/apply_{primary_subject}",
                description=(
                    f"Increment {positive[0].subject} and let the derived "
                    f"law propagate to {positive[0].object}."
                ),
                rules=[
                    RuleSpec(variable=primary_value_var, operation="add", operand=1.0),
                    RuleSpec(variable=primary_count_var, operation="add", operand=1),
                ],
                cost=0.02,
                vocabulary=sorted(concepts),
                provenance_ids=[atom.atom_id for atom in positive],
            )
        )
        if secondary_subject is not None:
            secondary_value_var = f"{secondary_subject}.value"
            secondary_count_var = f"{secondary_subject}.interventions"
            # Start the "mass-like" variable at 1.0 so the derived
            # relation is well-defined from the first step.
            initial_state[secondary_value_var] = 1.0
            initial_state[secondary_count_var] = 0
            actions.append(
                ActionSpec(
                    name=f"generated/apply_{secondary_subject}",
                    description=(
                        f"Increment {negative[0].subject}; this should reduce "
                        f"{negative[0].object} via the derived relation."
                    ),
                    rules=[
                        RuleSpec(variable=secondary_value_var, operation="add", operand=1.0),
                        RuleSpec(variable=secondary_count_var, operation="add", operand=1),
                    ],
                    cost=0.02,
                    vocabulary=sorted(concepts),
                    provenance_ids=[atom.atom_id for atom in negative],
                )
            )

        # Derived rule: target := primary_value / max(secondary_value, 1).
        if secondary_subject is not None:
            derived_expr = ExpressionSpec(
                kind="div",
                args=[
                    ExpressionSpec.var(primary_value_var),
                    ExpressionSpec(
                        kind="max",
                        args=[
                            ExpressionSpec.var(f"{secondary_subject}.value"),
                            ExpressionSpec.lit(1.0),
                        ],
                    ),
                ],
            )
            invariants = [
                ExpressionSpec.cmp(">=", ExpressionSpec.var(f"{secondary_subject}.value"), ExpressionSpec.lit(0.0)),
            ]
        else:
            derived_expr = ExpressionSpec.var(primary_value_var)
            invariants = []
        derived = [
            RuleSpec(
                variable=target_variable,
                operation="compute",
                expression=derived_expr,
                derived=True,
            )
        ]

        return WorldSpec(
            name=f"generated/composite_{object_slug}",
            description=(
                f"Composite sandbox for {object_slug}: {len(positive)} positive driver(s)"
                + (f", {len(negative)} resistive driver(s)" if negative else "")
            ),
            concepts=sorted(concepts),
            initial_state=initial_state,
            actions=actions,
            provenance_ids=provenance_ids,
            derived=derived,
            invariants=invariants,
        )

    def _slug(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
        return slug[:40] or "concept"


class SandboxedWorldCompiler:
    # v4 ops kept; v5 adds compute/clamp/if_then.
    primary_operations = {"add", "set", "toggle", "compute", "clamp", "if_then"}
    variable_pattern = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")

    @property
    def allowed_operations(self) -> set[str]:
        """Backwards-compatible alias kept for any external readers."""

        return set(self.primary_operations)

    def validate(self, spec: WorldSpec) -> ValidationResult:
        errors: list[str] = []
        if spec.contains_code:
            errors.append("world specs must be data, not code")
        if spec.trust_level != "sandboxed":
            errors.append("generated worlds must start sandboxed")
        if not spec.actions:
            errors.append("world spec must expose at least one action")
        for variable in spec.initial_state:
            if not self.variable_pattern.match(variable):
                errors.append(f"invalid variable name: {variable}")
        for action in spec.actions:
            if not action.name.startswith("generated/"):
                errors.append(f"generated action must use generated/ prefix: {action.name}")
            for rule in action.rules:
                self._validate_rule(rule, errors)
        for derived_rule in spec.derived:
            if not derived_rule.derived:
                errors.append(f"derived rule must set derived=True: {derived_rule.variable}")
            self._validate_rule(derived_rule, errors)
        for invariant in spec.invariants:
            _validate_expression(invariant, errors)
        return ValidationResult(valid=not errors, errors=errors)

    def _validate_rule(self, rule: RuleSpec, errors: list[str]) -> None:
        if rule.operation not in self.primary_operations:
            errors.append(f"unsupported operation: {rule.operation}")
            return
        if not self.variable_pattern.match(rule.variable):
            errors.append(f"invalid variable name: {rule.variable}")
        if rule.operation == "add":
            if not isinstance(rule.operand, (int, float)):
                errors.append(f"add operand must be numeric for {rule.variable}")
        elif rule.operation == "compute":
            if rule.expression is None:
                errors.append(f"compute rule must include an expression for {rule.variable}")
            else:
                _validate_expression(rule.expression, errors)
        elif rule.operation == "clamp":
            if not isinstance(rule.operand, Mapping):
                errors.append(f"clamp operand must be a dict with 'min' and 'max' for {rule.variable}")
            else:
                if "min" not in rule.operand or "max" not in rule.operand:
                    errors.append(f"clamp operand requires both 'min' and 'max' for {rule.variable}")
        elif rule.operation == "if_then":
            if rule.expression is None:
                errors.append(f"if_then rule must include a condition expression for {rule.variable}")
            else:
                _validate_expression(rule.expression, errors)
            for nested in rule.then_rules + rule.else_rules:
                self._validate_rule(nested, errors)

    def compile(self, spec: WorldSpec) -> "SandboxedGeneratedAdapter":
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return SandboxedGeneratedAdapter(spec)


class SandboxedGeneratedAdapter:
    """Runs a WorldSpec as an environment.

    A step is:
      1. Apply the action's primary rules in order.
      2. Re-apply every derived rule to fixed point (cap MAX_DERIVED_ITERS).
      3. Check every invariant. A violation does not raise; it surfaces as
         a negative reward and gets recorded in ``last_invariant_violations``.

    ``counterfactual(action, base_state)`` performs steps 1-3 on a copy of
    the state and returns the delta without committing — used by Phase D
    for cheap "what if" planning.
    """

    MAX_DERIVED_ITERS = 8

    def __init__(self, spec: WorldSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self._state = dict(spec.initial_state)
        self._steps = 0
        self.last_invariant_violations: list[str] = []

    def observe(self) -> State:
        return dict(self._state)

    def possible_actions(self) -> list[Action]:
        return [
            Action(
                action.name,
                cost=action.cost,
                description=action.description,
                metadata={
                    "domain": self.spec.name,
                    "vocabulary": set(action.vocabulary),
                    "provenance_ids": list(action.provenance_ids or self.spec.provenance_ids),
                },
            )
            for action in self.spec.actions
        ]

    def apply(self, action: Action) -> tuple[State, float]:
        if self._steps >= self.spec.step_budget:
            return self.observe(), -1.0
        action_spec = self._action_spec(action.name)
        if action_spec is None:
            raise ValueError(f"Unknown generated action: {action.name}")
        self._steps += 1
        state, violations = self._step_in_place(self._state, action_spec, action)
        self._state = state
        self.last_invariant_violations = violations
        reward = max(0.0, 0.25 - float(action.cost))
        if violations:
            reward -= 0.5
        return self.observe(), reward

    def counterfactual(self, action: Action | str, base_state: State | None = None) -> dict[str, Any]:
        """Apply ``action`` to a *copy* of state. Return delta + violations.

        Used by simulation/planning for cheap "what if" rollouts that do not
        mutate the live adapter.
        """

        action_name = action.name if isinstance(action, Action) else action
        action_spec = self._action_spec(action_name)
        if action_spec is None:
            raise ValueError(f"Unknown generated action: {action_name}")
        state = dict(base_state if base_state is not None else self._state)
        before = dict(state)
        proxy_action = action if isinstance(action, Action) else next(
            (a for a in self.possible_actions() if a.name == action_name), None
        )
        if proxy_action is None:
            proxy_action = Action(action_name)
        state, violations = self._step_in_place(state, action_spec, proxy_action)
        delta: dict[str, Any] = {}
        keys = set(before) | set(state)
        for key in keys:
            if before.get(key) != state.get(key):
                delta[key] = {"before": before.get(key), "after": state.get(key)}
        return {
            "action": action_name,
            "delta": delta,
            "after": state,
            "violations": violations,
        }

    # ----- step machinery -------------------------------------------------

    def _step_in_place(
        self,
        state: State,
        action_spec: ActionSpec,
        action: Action,
    ) -> tuple[State, list[str]]:
        for rule in action_spec.rules:
            self._apply_rule(state, rule)
        # Re-apply derived rules to fixed point.
        for _ in range(self.MAX_DERIVED_ITERS):
            snapshot = dict(state)
            for derived_rule in self.spec.derived:
                self._apply_rule(state, derived_rule)
            if state == snapshot:
                break
        violations: list[str] = []
        for invariant in self.spec.invariants:
            try:
                if not _to_bool(_eval_expression(invariant, state)):
                    violations.append(self._describe_expression(invariant))
            except Exception as exc:  # pragma: no cover - defensive
                violations.append(f"invariant raised {exc!r}")
        return state, violations

    def _apply_rule(self, state: State, rule: RuleSpec) -> None:
        if rule.operation == "add":
            current = state.get(rule.variable, 0)
            state[rule.variable] = _to_number(current) + _to_number(rule.operand)
        elif rule.operation == "set":
            state[rule.variable] = rule.operand
        elif rule.operation == "toggle":
            state[rule.variable] = not _to_bool(state.get(rule.variable, False))
        elif rule.operation == "compute":
            if rule.expression is None:
                return
            value = _eval_expression(rule.expression, state)
            state[rule.variable] = value
        elif rule.operation == "clamp":
            current = _to_number(state.get(rule.variable, 0))
            lo = _to_number(rule.operand.get("min", current))
            hi = _to_number(rule.operand.get("max", current))
            state[rule.variable] = max(lo, min(hi, current))
        elif rule.operation == "if_then":
            if rule.expression is None:
                return
            taken = rule.then_rules if _to_bool(_eval_expression(rule.expression, state)) else rule.else_rules
            for nested in taken:
                self._apply_rule(state, nested)
        # Unknown operations are blocked by the compiler validator.

    def _action_spec(self, name: str) -> ActionSpec | None:
        for candidate in self.spec.actions:
            if candidate.name == name:
                return candidate
        return None

    def _describe_expression(self, expr: ExpressionSpec) -> str:
        if expr.kind == "lit":
            return repr(expr.value)
        if expr.kind == "var":
            return expr.variable
        if expr.kind == "cmp":
            return f"({self._describe_expression(expr.args[0])} {expr.op} {self._describe_expression(expr.args[1])})"
        if expr.kind in {"and", "or"}:
            joined = f" {expr.kind} ".join(self._describe_expression(arg) for arg in expr.args)
            return f"({joined})"
        if expr.kind == "not":
            return f"(not {self._describe_expression(expr.args[0])})"
        joined = f" {expr.kind} ".join(self._describe_expression(arg) for arg in expr.args)
        return f"({joined})"

    # ----- adapter interface ---------------------------------------------

    def action_metadata(self, action: Action | str) -> dict[str, Any]:
        action_name = action.name if isinstance(action, Action) else action
        action_spec = self._action_spec(action_name)
        provenance_ids = action_spec.provenance_ids if action_spec else self.spec.provenance_ids
        return {
            "scope": "world",
            "world": self.spec.name,
            "domain": self.spec.name,
            "generated": True,
            "provenance_ids": list(provenance_ids or self.spec.provenance_ids),
        }

    def actions_for_terms(self, terms: set[str]) -> list[Action]:
        terms = {term.lower() for term in terms}
        matches = []
        for action in self.possible_actions():
            vocabulary = {str(item).lower() for item in action.metadata.get("vocabulary", set())}
            if terms & vocabulary:
                matches.append(action)
        return matches

    def variables_for_domain(self, domain: str) -> list[str]:
        if domain != self.spec.name:
            return []
        return sorted(self._state)


class GenerativeUniverse:
    def __init__(self, adapters: list[SandboxedGeneratedAdapter]) -> None:
        self.adapters = adapters
        self._cursor = 0

    @classmethod
    def from_specs(cls, specs: list[WorldSpec]) -> "GenerativeUniverse":
        compiler = SandboxedWorldCompiler()
        return cls([compiler.compile(spec) for spec in specs])

    def observe(self) -> State:
        state: State = {}
        for adapter in self.adapters:
            state.update(adapter.observe())
        return state

    def possible_actions(self) -> list[Action]:
        actions: list[Action] = []
        for adapter in self.adapters:
            actions.extend(adapter.possible_actions())
        return actions

    def apply(self, action: Action) -> tuple[State, float]:
        adapter = self._adapter_for_action(action.name)
        _after, reward = adapter.apply(action)
        return self.observe(), reward

    def adapter_for_action(self, action: Action | str) -> SandboxedGeneratedAdapter:
        action_name = action.name if isinstance(action, Action) else action
        return self._adapter_for_action(action_name)

    def _adapter_for_action(self, action_name: str) -> SandboxedGeneratedAdapter:
        for adapter in self.adapters:
            if any(action.name == action_name for action in adapter.possible_actions()):
                return adapter
        raise ValueError(f"Unknown generated action: {action_name}")


class GenerativeUniverseAdapter:
    name = "generative_universe"

    def __init__(self, universe: GenerativeUniverse) -> None:
        self.universe = universe

    def observe(self) -> State:
        return self.universe.observe()

    def possible_actions(self) -> list[Action]:
        return self.universe.possible_actions()

    def apply(self, action: Action) -> tuple[State, float]:
        return self.universe.apply(action)

    def action_metadata(self, action: Action | str) -> dict[str, Any]:
        return self.universe.adapter_for_action(action).action_metadata(action)

    def actions_for_terms(self, terms: set[str]) -> list[Action]:
        actions: list[Action] = []
        for adapter in self.universe.adapters:
            actions.extend(adapter.actions_for_terms(terms))
        return actions

    def variables_for_domain(self, domain: str) -> list[str]:
        variables: list[str] = []
        for adapter in self.universe.adapters:
            variables.extend(adapter.variables_for_domain(domain))
        return sorted(variables)

    def counterfactual(self, action: Action | str) -> dict[str, Any]:
        """Forward a counterfactual rollout to the right sub-adapter."""

        adapter = self.universe.adapter_for_action(action)
        return adapter.counterfactual(action)
