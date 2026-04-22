from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from darwin.causal import CausalModel
from darwin.memory import Memory
from darwin.types import Transition
from darwin.world_model import WorldModel


@dataclass
class Competence:
    action: str
    samples: int = 0
    reward_mean: float = 0.0
    surprise_mean: float = 0.0

    @property
    def score(self) -> float:
        familiarity = min(1.0, self.samples / 10.0)
        return familiarity * (1.0 + self.reward_mean) / (1.0 + self.surprise_mean)

    def update(self, reward: float, surprise: float) -> None:
        self.samples += 1
        self.reward_mean += (reward - self.reward_mean) / self.samples
        self.surprise_mean += (surprise - self.surprise_mean) / self.samples


@dataclass
class SelfReport:
    observations: int
    known_actions: int
    known_variables: int
    strongest_belief: str
    weakest_area: str
    learning_priority: str
    competence: list[Competence]

    def lines(self) -> list[str]:
        action_line = ", ".join(
            f"{item.action}:{item.score:.2f}" for item in sorted(self.competence, key=lambda c: c.score, reverse=True)[:5]
        )
        return [
            f"observations={self.observations}",
            f"known_actions={self.known_actions}",
            f"known_variables={self.known_variables}",
            f"strongest_belief={self.strongest_belief}",
            f"weakest_area={self.weakest_area}",
            f"learning_priority={self.learning_priority}",
            f"competence={action_line or 'none'}",
        ]


class SelfModel:
    """Darwin's internal estimate of its knowledge, limits, and priorities."""

    def __init__(self) -> None:
        self.competence_by_action: dict[str, Competence] = {}
        self.known_variables: Counter[str] = Counter()
        self.prediction_failures: Counter[str] = Counter()
        self.reflections: list[str] = []
        self._last_prediction_error_count = 0

    def learn(self, transition: Transition) -> None:
        after = dict(transition.after)
        for variable in after:
            self.known_variables[variable] += 1

        predicted = dict(transition.metadata.get("predicted_state", {}))
        surprises = 0
        for variable, observed in after.items():
            if variable in predicted and predicted[variable] != observed:
                surprises += 1
                self.prediction_failures[f"{transition.action}:{variable}"] += 1

        competence = self.competence_by_action.get(transition.action)
        if competence is None:
            competence = Competence(action=transition.action)
            self.competence_by_action[transition.action] = competence
        competence.update(float(transition.reward), float(surprises))

    def reflect(self, memory: Memory, causal_model: CausalModel, world_model: WorldModel) -> str:
        report = self.report(memory, causal_model, world_model)
        reflection = (
            f"I have {report.observations} grounded transitions. "
            f"My strongest belief is {report.strongest_belief}. "
            f"My weakest area is {report.weakest_area}. "
            f"My next learning priority is {report.learning_priority}."
        )
        self.reflections.append(reflection)
        return reflection

    def report(self, memory: Memory, causal_model: CausalModel, world_model: WorldModel) -> SelfReport:
        strongest = causal_model.beliefs(limit=1)
        strongest_belief = "none"
        if strongest:
            belief = strongest[0]
            strongest_belief = (
                f"if {belief.condition}: {belief.action}->{belief.variable} "
                f"{belief.effect} conf={belief.confidence:.2f}"
            )

        active_failures = self._active_prediction_failures(causal_model)
        weakest_area = "no active prediction errors"
        if active_failures:
            weakest_area = active_failures.most_common(1)[0][0]
        elif world_model.hidden_factors:
            weakest_area = world_model.hidden_factors.most_common(1)[0][0]

        learning_priority = self._learning_priority(causal_model, world_model)

        return SelfReport(
            observations=len(memory.episodes),
            known_actions=len(causal_model.known_actions()),
            known_variables=len(self.known_variables),
            strongest_belief=strongest_belief,
            weakest_area=weakest_area,
            learning_priority=learning_priority,
            competence=list(self.competence_by_action.values()),
        )

    def _learning_priority(self, causal_model: CausalModel, world_model: WorldModel) -> str:
        active_failures = self._active_prediction_failures(causal_model)
        if active_failures:
            key = active_failures.most_common(1)[0][0]
            action, variable = key.split(":", 1)
            if causal_model.action_count(action) < causal_model.min_samples:
                return f"retest {action} to stabilize its effect on {variable}"
            return f"find hidden conditions for {key}"
        if causal_model.total_observations() < 5:
            return "collect more interventions"
        if world_model.hidden_factors:
            return f"test hidden factor hypothesis {world_model.hidden_factors.most_common(1)[0][0]}"
        weakest = sorted(self.competence_by_action.values(), key=lambda item: item.score)
        if weakest:
            return f"improve competence with {weakest[0].action}"
        return "expand the environment with new actions and variables"

    def _active_prediction_failures(self, causal_model: CausalModel) -> Counter[str]:
        resolved: set[str] = set()
        for belief in causal_model.beliefs(limit=200):
            if belief.confidence >= 0.6 and belief.samples >= 2:
                resolved.add(f"{belief.action}:{belief.variable}")

        active: Counter[str] = Counter()
        for key, count in self.prediction_failures.items():
            if key not in resolved:
                active[key] = count
        return active

    def to_record(self, memory: Memory, causal_model: CausalModel, world_model: WorldModel) -> dict[str, Any]:
        report = self.report(memory, causal_model, world_model)
        return {
            "observations": report.observations,
            "known_actions": report.known_actions,
            "known_variables": report.known_variables,
            "strongest_belief": report.strongest_belief,
            "weakest_area": report.weakest_area,
            "learning_priority": report.learning_priority,
            "competence": [
                {
                    "action": item.action,
                    "samples": item.samples,
                    "reward_mean": item.reward_mean,
                    "surprise_mean": item.surprise_mean,
                    "score": item.score,
                }
                for item in report.competence
            ],
        }
