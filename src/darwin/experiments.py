from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from darwin.causal import CausalModel
from darwin.types import Action, Goal, State, Transition


@dataclass
class ExperimentProposal:
    action: Action
    state: State
    predicted_state: State
    uncertainty: float
    expected_reward: float
    question: str
    rationale: str

    def to_record(self, status: str = "proposed", result: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": status,
            "action": self.action.name,
            "uncertainty": self.uncertainty,
            "prediction": {
                "state": self.state,
                "predicted_state": self.predicted_state,
                "expected_reward": self.expected_reward,
                "question": self.question,
                "rationale": self.rationale,
            },
            "result": dict(result or {}),
        }


@dataclass
class ExperimentResult:
    proposal: ExperimentProposal
    transition: Transition
    surprises: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return not self.surprises

    def to_record(self) -> dict[str, Any]:
        return self.proposal.to_record(
            status="confirmed" if self.confirmed else "surprising",
            result={
                "transition": {
                    "before": dict(self.transition.before),
                    "action": self.transition.action,
                    "after": dict(self.transition.after),
                    "reward": self.transition.reward,
                    "t": self.transition.t,
                },
                "surprises": {
                    variable: {"predicted": values[0], "observed": values[1]}
                    for variable, values in self.surprises.items()
                },
            },
        )


class ExperimentEngine:
    """Chooses interventions that should teach Darwin the most."""

    def __init__(self, causal_model: CausalModel) -> None:
        self.causal_model = causal_model
        self.completed: list[ExperimentResult] = []

    def propose(
        self,
        state: Mapping[str, Any],
        actions: Iterable[Action],
        goal: Goal | None = None,
        limit: int = 5,
    ) -> list[ExperimentProposal]:
        proposals: list[ExperimentProposal] = []
        for action in actions:
            prediction = self.causal_model.predict(state, action.name)
            reward = self.causal_model.expected_reward(state, action.name)
            uncertainty = self.causal_model.uncertainty_for(state, action.name)
            question = self._question_for(action, prediction.state, uncertainty)
            rationale = (
                f"uncertainty={uncertainty:.2f}, samples={self.causal_model.action_count(action.name)}, "
                f"expected_reward={reward.mean:.2f}"
            )
            if goal is not None:
                rationale += f", goal_variables={','.join(goal.desired)}"
            proposals.append(
                ExperimentProposal(
                    action=action,
                    state=dict(state),
                    predicted_state=prediction.state,
                    uncertainty=uncertainty,
                    expected_reward=reward.mean,
                    question=question,
                    rationale=rationale,
                )
            )

        proposals.sort(key=lambda item: (item.uncertainty, -abs(item.expected_reward)), reverse=True)
        return proposals[:limit]

    def evaluate(self, proposal: ExperimentProposal, transition: Transition) -> ExperimentResult:
        surprises: dict[str, tuple[Any, Any]] = {}
        predicted = dict(proposal.predicted_state)
        observed = dict(transition.after)

        for variable, value in observed.items():
            if variable in predicted and predicted[variable] != value:
                surprises[variable] = (predicted[variable], value)

        result = ExperimentResult(proposal=proposal, transition=transition, surprises=surprises)
        self.completed.append(result)
        return result

    def _question_for(self, action: Action, predicted_state: Mapping[str, Any], uncertainty: float) -> str:
        if uncertainty >= 0.75:
            return f"What does {action.name} actually cause from the current state?"
        changed = ", ".join(f"{key}={value!r}" for key, value in sorted(predicted_state.items())[:3])
        return f"Will {action.name} reliably produce {changed}?"

