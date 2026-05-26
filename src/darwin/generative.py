from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from darwin.knowledge import KnowledgeAtom, KnowledgeGraph
from darwin.types import Action, State


@dataclass
class RuleSpec:
    variable: str
    operation: str
    operand: Any

    def to_record(self) -> dict[str, Any]:
        return {"variable": self.variable, "operation": self.operation, "operand": self.operand}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RuleSpec":
        return cls(
            variable=str(record.get("variable", "")),
            operation=str(record.get("operation", "")),
            operand=record.get("operand"),
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
        )


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class WorldSpecGenerator:
    def generate(self, graph: KnowledgeGraph, limit: int = 12) -> list[WorldSpec]:
        specs: list[WorldSpec] = []
        for atom in graph.causal_hypotheses()[:limit]:
            specs.append(self._spec_from_hypothesis(atom))
        return specs

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

    def _slug(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
        return slug[:40] or "concept"


class SandboxedWorldCompiler:
    allowed_operations = {"add", "set", "toggle"}
    variable_pattern = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")

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
                if rule.operation not in self.allowed_operations:
                    errors.append(f"unsupported operation: {rule.operation}")
                if not self.variable_pattern.match(rule.variable):
                    errors.append(f"invalid variable name: {rule.variable}")
                if rule.operation == "add" and not isinstance(rule.operand, (int, float)):
                    errors.append(f"add operand must be numeric for {rule.variable}")
        return ValidationResult(valid=not errors, errors=errors)

    def compile(self, spec: WorldSpec) -> "SandboxedGeneratedAdapter":
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return SandboxedGeneratedAdapter(spec)


class SandboxedGeneratedAdapter:
    def __init__(self, spec: WorldSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self._state = dict(spec.initial_state)
        self._steps = 0

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
        action_spec = next((candidate for candidate in self.spec.actions if candidate.name == action.name), None)
        if action_spec is None:
            raise ValueError(f"Unknown generated action: {action.name}")
        self._steps += 1
        for rule in action_spec.rules:
            current = self._state.get(rule.variable)
            if rule.operation == "add":
                self._state[rule.variable] = float(current or 0.0) + float(rule.operand)
            elif rule.operation == "set":
                self._state[rule.variable] = rule.operand
            elif rule.operation == "toggle":
                self._state[rule.variable] = not bool(current)
        return self.observe(), max(0.0, 0.25 - float(action.cost))

    def action_metadata(self, action: Action | str) -> dict[str, Any]:
        action_name = action.name if isinstance(action, Action) else action
        action_spec = next((candidate for candidate in self.spec.actions if candidate.name == action_name), None)
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
