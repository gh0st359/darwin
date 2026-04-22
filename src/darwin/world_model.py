from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from darwin.causal import CausalBelief, CausalModel
from darwin.types import Action, State, Transition


@dataclass
class EntityModel:
    name: str
    variables: set[str] = field(default_factory=set)
    observations: int = 0


@dataclass
class PredictionError:
    action: str
    variable: str
    predicted: Any
    observed: Any
    t: int


@dataclass
class WorldPrediction:
    action: str
    before: State
    after: State
    confidence: float
    uncertainty: float
    expected_reward: float
    reasons: list[str]


@dataclass
class Hypothesis:
    name: str
    description: str
    confidence: float
    evidence: int
    open_question: str = ""


class WorldModel:
    """A structured, self-updating model of Darwin's experienced reality."""

    def __init__(self) -> None:
        self.entities: dict[str, EntityModel] = {}
        self.variables: Counter[str] = Counter()
        self.state_value_counts: dict[str, Counter[Any]] = defaultdict(Counter)
        self.action_counts: Counter[str] = Counter()
        self.prediction_errors: list[PredictionError] = []
        self.hidden_factors: Counter[str] = Counter()
        self.last_state: State = {}

    def learn(self, transition: Transition) -> None:
        before = dict(transition.before)
        after = dict(transition.after)
        self.last_state = after
        self.action_counts[transition.action] += 1

        for variable, value in after.items():
            self.variables[variable] += 1
            self.state_value_counts[variable][self._freeze(value)] += 1
            entity = self._entity_for(variable)
            entity.variables.add(variable)
            entity.observations += 1

        predicted = dict(transition.metadata.get("predicted_state", {}))
        if predicted:
            for variable, observed in after.items():
                if variable not in predicted:
                    continue
                expected = predicted[variable]
                if expected != observed:
                    error = PredictionError(
                        action=transition.action,
                        variable=variable,
                        predicted=expected,
                        observed=observed,
                        t=transition.t,
                    )
                    self.prediction_errors.append(error)
                    self.hidden_factors[f"{transition.action}:{variable}"] += 1

    def predict(self, state: Mapping[str, Any], action: Action | str, causal_model: CausalModel) -> WorldPrediction:
        action_name = action.name if isinstance(action, Action) else action
        prediction = causal_model.predict(state, action_name)
        reward = causal_model.expected_reward(state, action_name)
        reasons = [
            f"{estimate.variable} <- {estimate.predicted_value!r} ({estimate.reason})"
            for estimate in prediction.estimates
        ]
        if not reasons:
            reasons.append("no grounded transition data yet")
        return WorldPrediction(
            action=action_name,
            before=dict(state),
            after=prediction.state,
            confidence=prediction.confidence,
            uncertainty=prediction.uncertainty,
            expected_reward=reward.mean,
            reasons=reasons,
        )

    def hypotheses(self, causal_model: CausalModel, limit: int = 10) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        for belief in causal_model.beliefs(limit=limit * 2):
            hypotheses.append(self._belief_to_hypothesis(belief))

        for key, count in self.hidden_factors.most_common(limit):
            action, variable = key.split(":", 1)
            hypotheses.append(
                Hypothesis(
                    name=f"hidden_factor:{key}",
                    description=(
                        f"Prediction errors suggest {action} has an unmodeled condition "
                        f"affecting {variable}."
                    ),
                    confidence=min(0.95, count / 5.0),
                    evidence=count,
                    open_question=f"What state feature controls {action}'s effect on {variable}?",
                )
            )

        hypotheses.sort(key=lambda item: (item.confidence, item.evidence), reverse=True)
        return hypotheses[:limit]

    def summary(self, causal_model: CausalModel) -> str:
        top_beliefs = causal_model.beliefs(limit=3)
        unstable = self.hidden_factors.most_common(1)
        lines = [
            f"variables={len(self.variables)} actions={len(self.action_counts)}",
            f"prediction_errors={len(self.prediction_errors)}",
        ]
        if top_beliefs:
            belief = top_beliefs[0]
            lines.append(
                f"strongest={belief.action}->{belief.variable} {belief.effect} "
                f"conf={belief.confidence:.2f}"
            )
        if unstable:
            lines.append(f"least_understood={unstable[0][0]} errors={unstable[0][1]}")
        return " | ".join(lines)

    def _belief_to_hypothesis(self, belief: CausalBelief) -> Hypothesis:
        return Hypothesis(
            name=f"causal:{belief.action}:{belief.variable}:{belief.condition}",
            description=(
                f"When {belief.condition}, action {belief.action} tends to make "
                f"{belief.variable} change as {belief.effect}."
            ),
            confidence=belief.confidence,
            evidence=belief.samples,
        )

    def _entity_for(self, variable: str) -> EntityModel:
        name = variable.split("_", 1)[0] if "_" in variable else "world"
        entity = self.entities.get(name)
        if entity is None:
            entity = EntityModel(name=name)
            self.entities[name] = entity
        return entity

    def _freeze(self, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((key, self._freeze(inner)) for key, inner in value.items()))
        if isinstance(value, list):
            return tuple(self._freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze(item) for item in value))
        return value

