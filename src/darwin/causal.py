from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Iterable, Mapping

from darwin.types import State, Transition


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(inner)) for key, inner in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass
class RewardEstimate:
    mean: float
    confidence: float
    samples: int


@dataclass
class EffectEstimate:
    variable: str
    predicted_value: Any
    confidence: float
    samples: int
    changed_probability: float
    reason: str


@dataclass
class Prediction:
    action: str
    state: State
    estimates: list[EffectEstimate]
    confidence: float
    uncertainty: float


@dataclass
class CausalBelief:
    action: str
    variable: str
    effect: str
    confidence: float
    samples: int
    condition: str = "always"


@dataclass
class RewardStats:
    samples: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.samples += 1
        delta = value - self.mean
        self.mean += delta / self.samples
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        if self.samples < 2:
            return 0.0
        return self.m2 / (self.samples - 1)

    def confidence(self, min_samples: int) -> float:
        sample_confidence = min(1.0, self.samples / max(1, min_samples))
        stability = 1.0 / (1.0 + sqrt(self.variance))
        return sample_confidence * stability


@dataclass
class EffectStats:
    samples: int = 0
    changed: int = 0
    numeric_samples: int = 0
    delta_mean: float = 0.0
    delta_m2: float = 0.0
    after_values: Counter[Any] = field(default_factory=Counter)
    transitions: Counter[tuple[Any, Any]] = field(default_factory=Counter)

    def update(self, before_value: Any, after_value: Any) -> None:
        before_key = _freeze(before_value)
        after_key = _freeze(after_value)

        self.samples += 1
        self.after_values[after_key] += 1
        self.transitions[(before_key, after_key)] += 1

        if before_value != after_value:
            self.changed += 1

        if _is_number(before_value) and _is_number(after_value):
            delta = float(after_value) - float(before_value)
            self.numeric_samples += 1
            diff = delta - self.delta_mean
            self.delta_mean += diff / self.numeric_samples
            self.delta_m2 += diff * (delta - self.delta_mean)

    @property
    def changed_probability(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.changed / self.samples

    @property
    def delta_variance(self) -> float:
        if self.numeric_samples < 2:
            return 0.0
        return self.delta_m2 / (self.numeric_samples - 1)

    @property
    def outcome_consistency(self) -> float:
        if self.samples == 0 or not self.after_values:
            return 0.0
        return self.after_values.most_common(1)[0][1] / self.samples

    @property
    def change_consistency(self) -> float:
        if self.samples == 0:
            return 0.0
        change_rate = self.changed_probability
        return max(change_rate, 1.0 - change_rate)

    def confidence(self, min_samples: int) -> float:
        if self.samples == 0:
            return 0.0
        sample_confidence = min(1.0, self.samples / max(1, min_samples))
        consistency = (self.outcome_consistency + self.change_consistency) / 2.0
        if self.numeric_samples:
            consistency *= 1.0 / (1.0 + sqrt(self.delta_variance))
        return sample_confidence * consistency

    def predict_value(self, current_value: Any) -> Any:
        if self.samples == 0:
            return current_value

        if self.numeric_samples and _is_number(current_value):
            return float(current_value) + self.delta_mean

        return self.after_values.most_common(1)[0][0]

    def describe_effect(self) -> str:
        if self.samples == 0:
            return "unknown"
        before_value, after_value = self.transitions.most_common(1)[0][0]
        if self.numeric_samples:
            return f"+= {self.delta_mean:.3g}"
        return f"{before_value!r} -> {after_value!r}"


class CausalModel:
    """Learns action consequences from observed interventions."""

    def __init__(self, min_samples: int = 3) -> None:
        self.min_samples = min_samples
        self._effects: dict[tuple[str, str], EffectStats] = defaultdict(EffectStats)
        self._conditioned_effects: dict[tuple[str, str, str, Any], EffectStats] = defaultdict(EffectStats)
        self._rewards: dict[str, RewardStats] = defaultdict(RewardStats)
        self._conditioned_rewards: dict[tuple[str, str, Any], RewardStats] = defaultdict(RewardStats)
        self._action_counts: Counter[str] = Counter()

    def learn(self, transition: Transition) -> None:
        action = transition.action
        before = dict(transition.before)
        after = dict(transition.after)
        variables = sorted(set(before) | set(after))

        self._action_counts[action] += 1
        self._rewards[action].update(float(transition.reward))

        for feature, value in before.items():
            self._conditioned_rewards[(action, feature, _freeze(value))].update(float(transition.reward))

        for variable in variables:
            before_value = before.get(variable)
            after_value = after.get(variable)
            self._effects[(action, variable)].update(before_value, after_value)

            for feature, value in before.items():
                self._conditioned_effects[(action, variable, feature, _freeze(value))].update(
                    before_value,
                    after_value,
                )

    def predict(self, state: Mapping[str, Any], action: str) -> Prediction:
        predicted = dict(state)
        estimates: list[EffectEstimate] = []

        variables = sorted(variable for known_action, variable in self._effects if known_action == action)
        for variable in variables:
            stats, reason = self._select_effect_stats(action, variable, state)
            if stats.samples == 0:
                continue

            predicted_value = stats.predict_value(state.get(variable))
            confidence = stats.confidence(self.min_samples)
            predicted[variable] = predicted_value
            estimates.append(
                EffectEstimate(
                    variable=variable,
                    predicted_value=predicted_value,
                    confidence=confidence,
                    samples=stats.samples,
                    changed_probability=stats.changed_probability,
                    reason=reason,
                )
            )

        confidence = _mean(estimate.confidence for estimate in estimates)
        if not estimates and self._action_counts[action] == 0:
            confidence = 0.0

        return Prediction(
            action=action,
            state=predicted,
            estimates=estimates,
            confidence=confidence,
            uncertainty=1.0 - confidence,
        )

    def expected_reward(self, state: Mapping[str, Any], action: str) -> RewardEstimate:
        stats, _reason = self._select_reward_stats(action, state)
        return RewardEstimate(
            mean=stats.mean,
            confidence=stats.confidence(self.min_samples),
            samples=stats.samples,
        )

    def uncertainty_for(self, state: Mapping[str, Any], action: str) -> float:
        prediction = self.predict(state, action)
        count = self._action_counts[action]
        underexplored = 1.0 - min(1.0, count / max(1, self.min_samples))
        return max(prediction.uncertainty, underexplored)

    def action_count(self, action: str) -> int:
        return self._action_counts[action]

    def known_actions(self) -> list[str]:
        actions = set(self._action_counts)
        actions.update(action for action, _variable in self._effects)
        return sorted(actions)

    def variables_for_action(self, action: str) -> list[str]:
        return sorted(variable for known_action, variable in self._effects if known_action == action)

    def total_observations(self) -> int:
        return sum(self._action_counts.values())

    def beliefs(self, limit: int = 20) -> list[CausalBelief]:
        beliefs: list[CausalBelief] = []

        for (action, variable), stats in self._effects.items():
            if stats.samples == 0 or stats.changed == 0:
                continue
            beliefs.append(
                CausalBelief(
                    action=action,
                    variable=variable,
                    effect=stats.describe_effect(),
                    confidence=stats.confidence(self.min_samples),
                    samples=stats.samples,
                )
            )

        for (action, variable, feature, value), stats in self._conditioned_effects.items():
            if stats.samples < 2 or stats.changed == 0:
                continue
            beliefs.append(
                CausalBelief(
                    action=action,
                    variable=variable,
                    effect=stats.describe_effect(),
                    confidence=stats.confidence(self.min_samples),
                    samples=stats.samples,
                    condition=f"{feature} == {value!r}",
                )
            )

        beliefs.sort(key=lambda belief: (belief.confidence, belief.samples), reverse=True)
        return beliefs[:limit]

    def _select_effect_stats(
        self,
        action: str,
        variable: str,
        state: Mapping[str, Any],
    ) -> tuple[EffectStats, str]:
        global_stats = self._effects[(action, variable)]
        best = global_stats
        best_reason = "global action effect"
        best_score = global_stats.confidence(self.min_samples) * 0.95

        for feature, value in state.items():
            conditioned = self._conditioned_effects[(action, variable, feature, _freeze(value))]
            if conditioned.samples == 0:
                continue
            specificity_bonus = 0.35 * (
                conditioned.outcome_consistency + conditioned.change_consistency
            ) / 2.0
            score = conditioned.confidence(self.min_samples) + specificity_bonus
            if score > best_score:
                best = conditioned
                best_score = score
                best_reason = f"conditioned on {feature} == {_freeze(value)!r}"

        return best, best_reason

    def _select_reward_stats(self, action: str, state: Mapping[str, Any]) -> tuple[RewardStats, str]:
        global_stats = self._rewards[action]
        best = global_stats
        best_reason = "global reward"
        best_score = global_stats.confidence(self.min_samples) * 0.95

        for feature, value in state.items():
            conditioned = self._conditioned_rewards[(action, feature, _freeze(value))]
            if conditioned.samples == 0:
                continue
            score = conditioned.confidence(self.min_samples) + 0.05
            if score > best_score:
                best = conditioned
                best_score = score
                best_reason = f"conditioned reward on {feature} == {_freeze(value)!r}"

        return best, best_reason
