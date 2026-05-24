from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from darwin.causal import CausalModel, Prediction
from darwin.types import Action, State


@dataclass
class ChainNode:
    step: int
    action: str
    state_before: State
    state_after: State
    confidence: float
    uncertainty: float
    expected_reward: float
    rationale: list[str]

    def to_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "expected_reward": self.expected_reward,
            "rationale": self.rationale,
        }


@dataclass
class CausalChain:
    """A simulated sequence of causal steps with propagated uncertainty."""

    nodes: list[ChainNode] = field(default_factory=list)
    final_state: State = field(default_factory=dict)
    chain_confidence: float = 1.0
    chain_uncertainty: float = 0.0
    total_expected_reward: float = 0.0

    @property
    def length(self) -> int:
        return len(self.nodes)

    def actions(self) -> list[str]:
        return [node.action for node in self.nodes]

    def describe(self) -> str:
        if not self.nodes:
            return "empty causal chain"
        pieces = []
        for node in self.nodes:
            pieces.append(
                f"step {node.step}: {node.action} (conf {node.confidence:.2f}, "
                f"unc {node.uncertainty:.2f})"
            )
        return " -> ".join(pieces)

    def to_record(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_record() for node in self.nodes],
            "final_state": self.final_state,
            "chain_confidence": self.chain_confidence,
            "chain_uncertainty": self.chain_uncertainty,
            "total_expected_reward": self.total_expected_reward,
        }


@dataclass
class CausalEdge:
    source_action: str
    variable: str
    effect: str
    confidence: float
    samples: int


@dataclass
class CausalGraph:
    """A bipartite action->variable graph distilled from learned beliefs."""

    edges: list[CausalEdge] = field(default_factory=list)
    actions: set[str] = field(default_factory=set)
    variables: set[str] = field(default_factory=set)

    def add_edge(self, edge: CausalEdge) -> None:
        self.edges.append(edge)
        self.actions.add(edge.source_action)
        self.variables.add(edge.variable)

    def neighbors_of_action(self, action: str) -> list[CausalEdge]:
        return [edge for edge in self.edges if edge.source_action == action]

    def actions_affecting(self, variable: str) -> list[CausalEdge]:
        return [edge for edge in self.edges if edge.variable == variable]

    def to_record(self) -> dict[str, Any]:
        return {
            "actions": sorted(self.actions),
            "variables": sorted(self.variables),
            "edges": [
                {
                    "action": edge.source_action,
                    "variable": edge.variable,
                    "effect": edge.effect,
                    "confidence": edge.confidence,
                    "samples": edge.samples,
                }
                for edge in self.edges
            ],
        }


class CausalChainEngine:
    """Builds multi-step causal chains and graphs on top of CausalModel."""

    def __init__(self, model: CausalModel) -> None:
        self.model = model

    def simulate_chain(
        self,
        state: Mapping[str, Any],
        actions: Iterable[Action | str],
    ) -> CausalChain:
        chain = CausalChain(final_state=dict(state))
        rolling_confidence = 1.0
        rolling_uncertainty = 0.0
        total_reward = 0.0

        current_state: State = dict(state)
        for step, action in enumerate(actions, start=1):
            action_name = action.name if isinstance(action, Action) else action
            prediction = self.model.predict(current_state, action_name)
            reward = self.model.expected_reward(current_state, action_name)
            step_uncertainty = max(prediction.uncertainty, self.model.uncertainty_for(current_state, action_name))
            rationale = [
                f"{estimate.variable} <- {estimate.predicted_value!r} "
                f"(conf {estimate.confidence:.2f}, {estimate.reason})"
                for estimate in prediction.estimates
            ] or ["no grounded effect known"]

            node = ChainNode(
                step=step,
                action=action_name,
                state_before=dict(current_state),
                state_after=dict(prediction.state),
                confidence=prediction.confidence,
                uncertainty=step_uncertainty,
                expected_reward=reward.mean,
                rationale=rationale,
            )
            chain.nodes.append(node)
            current_state = dict(prediction.state)
            total_reward += reward.mean
            rolling_confidence *= max(0.05, prediction.confidence)
            rolling_uncertainty = 1.0 - (1.0 - rolling_uncertainty) * (1.0 - step_uncertainty)

        chain.final_state = current_state
        chain.chain_confidence = rolling_confidence
        chain.chain_uncertainty = rolling_uncertainty
        chain.total_expected_reward = total_reward
        return chain

    def explore_chains(
        self,
        state: Mapping[str, Any],
        actions: Iterable[Action],
        depth: int = 3,
        beam: int = 4,
    ) -> list[CausalChain]:
        action_list = list(actions)
        if not action_list:
            return []

        beams: list[CausalChain] = [
            CausalChain(final_state=dict(state), chain_confidence=1.0)
        ]
        for _ in range(max(1, depth)):
            expanded: list[CausalChain] = []
            for chain in beams:
                for action in action_list:
                    extended = self.simulate_chain(
                        state, [*[node.action for node in chain.nodes], action.name]
                    )
                    expanded.append(extended)
            expanded.sort(
                key=lambda c: (c.chain_confidence * (1.0 + c.total_expected_reward)),
                reverse=True,
            )
            beams = expanded[: max(1, beam)]
        return beams

    def chain_for_goal(
        self,
        state: Mapping[str, Any],
        actions: Iterable[Action],
        goal_variables: Iterable[str],
        depth: int = 3,
        beam: int = 4,
    ) -> CausalChain | None:
        target = set(goal_variables)
        candidates = self.explore_chains(state, actions, depth=depth, beam=beam)
        scored: list[tuple[float, CausalChain]] = []
        for chain in candidates:
            satisfied = sum(
                1
                for variable in target
                if chain.final_state.get(variable) is not None
                and chain.final_state.get(variable) != state.get(variable)
            )
            score = satisfied * (1.0 + chain.chain_confidence) - chain.chain_uncertainty
            scored.append((score, chain))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else None

    def graph(self, min_confidence: float = 0.0, limit: int = 80) -> CausalGraph:
        graph = CausalGraph()
        for belief in self.model.beliefs(limit=limit):
            if belief.confidence < min_confidence:
                continue
            graph.add_edge(
                CausalEdge(
                    source_action=belief.action,
                    variable=belief.variable,
                    effect=belief.effect,
                    confidence=belief.confidence,
                    samples=belief.samples,
                )
            )
        return graph
