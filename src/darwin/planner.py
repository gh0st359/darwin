from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from darwin.causal import CausalModel, Prediction, RewardEstimate
from darwin.types import Action, Goal, State


@dataclass
class PlanCandidate:
    action: Action
    predicted: Prediction
    expected_reward: RewardEstimate
    goal_progress: float
    exploration_value: float
    score: float

    def explain(self) -> str:
        return (
            f"{self.action.name}: score={self.score:.3f}, "
            f"reward={self.expected_reward.mean:.3f}, "
            f"progress={self.goal_progress:.3f}, "
            f"curiosity={self.exploration_value:.3f}"
        )


@dataclass
class MultiStepPlan:
    actions: list[Action]
    final_state: State
    score: float
    total_expected_reward: float
    goal_score: float
    uncertainty: float
    trace: list[str]

    def explain(self) -> str:
        names = " -> ".join(action.name for action in self.actions) or "no-op"
        return (
            f"{names}: score={self.score:.3f}, reward={self.total_expected_reward:.3f}, "
            f"goal={self.goal_score:.3f}, uncertainty={self.uncertainty:.3f}"
        )


class CausalPlanner:
    """Ranks actions by predicted consequences and learning value."""

    def __init__(self, model: CausalModel) -> None:
        self.model = model

    def rank(self, state: Mapping[str, Any], actions: Iterable[Action], goal: Goal) -> list[PlanCandidate]:
        current_score = goal_satisfaction(state, goal)
        candidates: list[PlanCandidate] = []

        for action in actions:
            predicted = self.model.predict(state, action.name)
            expected_reward = self.model.expected_reward(state, action.name)
            predicted_goal_score = goal_satisfaction(predicted.state, goal)
            progress = predicted_goal_score - current_score
            exploration_value = self.model.uncertainty_for(state, action.name)

            score = (
                goal.reward_weight * expected_reward.mean
                + goal.progress_weight * progress
                + goal.exploration_weight * exploration_value
                - action.cost
            )

            candidates.append(
                PlanCandidate(
                    action=action,
                    predicted=predicted,
                    expected_reward=expected_reward,
                    goal_progress=progress,
                    exploration_value=exploration_value,
                    score=score,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

    def choose(self, state: Mapping[str, Any], actions: Iterable[Action], goal: Goal) -> PlanCandidate:
        ranked = self.rank(state, actions, goal)
        if not ranked:
            raise ValueError("Darwin cannot act without available actions.")
        return ranked[0]

    def plan_sequence(
        self,
        state: Mapping[str, Any],
        actions: Iterable[Action],
        goal: Goal,
        horizon: int = 3,
        beam_width: int = 5,
    ) -> MultiStepPlan:
        action_list = list(actions)
        if not action_list:
            raise ValueError("Darwin cannot plan without available actions.")

        beams: list[MultiStepPlan] = [
            MultiStepPlan(
                actions=[],
                final_state=dict(state),
                score=goal_satisfaction(state, goal),
                total_expected_reward=0.0,
                goal_score=goal_satisfaction(state, goal),
                uncertainty=0.0,
                trace=[],
            )
        ]

        for _depth in range(max(1, horizon)):
            expanded: list[MultiStepPlan] = []
            for beam in beams:
                for action in action_list:
                    prediction = self.model.predict(beam.final_state, action.name)
                    reward = self.model.expected_reward(beam.final_state, action.name)
                    uncertainty = self.model.uncertainty_for(beam.final_state, action.name)
                    total_reward = beam.total_expected_reward + reward.mean - action.cost
                    goal_score = goal_satisfaction(prediction.state, goal)
                    average_uncertainty = (
                        beam.uncertainty * len(beam.actions) + uncertainty
                    ) / max(1, len(beam.actions) + 1)
                    score = (
                        goal.progress_weight * goal_score
                        + goal.reward_weight * total_reward
                        + goal.exploration_weight * average_uncertainty
                    )
                    expanded.append(
                        MultiStepPlan(
                            actions=[*beam.actions, action],
                            final_state=prediction.state,
                            score=score,
                            total_expected_reward=total_reward,
                            goal_score=goal_score,
                            uncertainty=average_uncertainty,
                            trace=[
                                *beam.trace,
                                (
                                    f"{action.name} -> goal={goal_score:.2f} "
                                    f"reward={reward.mean:.2f} uncertainty={uncertainty:.2f}"
                                ),
                            ],
                        )
                    )
            expanded.sort(key=lambda plan: plan.score, reverse=True)
            beams = expanded[: max(1, beam_width)]

        return beams[0]


def goal_satisfaction(state: Mapping[str, Any], goal: Goal) -> float:
    if not goal.desired:
        return 0.0

    total = 0.0
    total_weight = 0.0
    for variable, desired in goal.desired.items():
        weight = float(goal.weights.get(variable, 1.0))
        actual = state.get(variable)
        total += weight * _match_score(actual, desired)
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return total / total_weight


def _match_score(actual: Any, desired: Any) -> float:
    if isinstance(actual, (int, float)) and isinstance(desired, (int, float)):
        scale = max(1.0, abs(float(desired)))
        distance = abs(float(actual) - float(desired))
        return max(0.0, 1.0 - min(1.0, distance / scale))
    return 1.0 if actual == desired else 0.0
