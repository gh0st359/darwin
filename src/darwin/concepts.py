from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from darwin.types import Transition


@dataclass
class Concept:
    name: str
    kind: str
    level: int = 0
    support: int = 0
    reward_total: float = 0.0
    parents: set[str] = field(default_factory=set)
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reward_mean(self) -> float:
        if self.support == 0:
            return 0.0
        return self.reward_total / self.support

    @property
    def salience(self) -> float:
        return self.support * (1.0 + abs(self.reward_mean)) * (1.0 + 0.15 * self.level)

    def add(self, reward: float, example: dict[str, Any], parents: set[str] | None = None) -> None:
        self.support += 1
        self.reward_total += reward
        if parents:
            self.parents.update(parents)
        if len(self.examples) < 5:
            self.examples.append(example)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "level": self.level,
            "support": self.support,
            "reward_mean": self.reward_mean,
            "parents": sorted(self.parents),
            "examples": self.examples,
            "salience": self.salience,
        }


class ConceptIndex:
    """Forms grounded concepts from state, effects, and outcomes."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}

    def learn(self, transition: Transition) -> None:
        before = dict(transition.before)
        after = dict(transition.after)
        reward = float(transition.reward)

        for variable, value in after.items():
            self._record(
                name=f"state:{variable}={value!r}",
                kind="state",
                level=0,
                reward=reward,
                example={"variable": variable, "value": value, "action": transition.action},
            )

        for variable in sorted(set(before) | set(after)):
            before_value = before.get(variable)
            after_value = after.get(variable)
            if before_value == after_value:
                continue
            self._record(
                name=f"effect:{transition.action}:{variable}",
                kind="effect",
                level=1,
                reward=reward,
                example={
                    "action": transition.action,
                    "variable": variable,
                    "before": before_value,
                    "after": after_value,
                },
                parents={
                    f"state:{variable}={before_value!r}",
                    f"state:{variable}={after_value!r}",
                },
            )

            self._record(
                name=f"affordance:{transition.action}:can_set:{variable}={after_value!r}",
                kind="affordance",
                level=2,
                reward=reward,
                example={
                    "action": transition.action,
                    "variable": variable,
                    "target_value": after_value,
                },
                parents={f"effect:{transition.action}:{variable}"},
            )

            if reward > 0:
                self._record(
                    name=f"strategy:seek:{variable}={after_value!r}:via:{transition.action}",
                    kind="strategy",
                    level=3,
                    reward=reward,
                    example={
                        "action": transition.action,
                        "variable": variable,
                        "target_value": after_value,
                    },
                    parents={f"affordance:{transition.action}:can_set:{variable}={after_value!r}"},
                )

        if reward > 0:
            self._record(
                name="outcome:positive_reward",
                kind="outcome",
                level=1,
                reward=reward,
                example={"action": transition.action, "reward": reward},
            )
            self._record(
                name=f"meta:reliable_action:{transition.action}",
                kind="meta",
                level=4,
                reward=reward,
                example={"action": transition.action, "reward": reward},
                parents={"outcome:positive_reward"},
            )
        elif reward < 0:
            self._record(
                name="outcome:negative_reward",
                kind="outcome",
                level=1,
                reward=reward,
                example={"action": transition.action, "reward": reward},
            )
            self._record(
                name=f"meta:risky_action:{transition.action}",
                kind="meta",
                level=4,
                reward=reward,
                example={"action": transition.action, "reward": reward},
                parents={"outcome:negative_reward"},
            )

    def salient(self, limit: int = 10) -> list[Concept]:
        concepts = list(self._concepts.values())
        concepts.sort(key=lambda concept: (concept.salience, concept.support), reverse=True)
        return concepts[:limit]

    def consolidate(self) -> dict[str, Any]:
        """Memory-consolidation pass.

        Forms cluster-of-affordance higher-order concepts from frequently
        co-occurring strategies and decays the support of very stale,
        low-salience concepts. Returns a summary of what was merged or
        decayed so background loops can report on it.
        """

        clusters: dict[str, list[Concept]] = defaultdict(list)
        for concept in self._concepts.values():
            if concept.kind == "affordance" and concept.parents:
                for parent in concept.parents:
                    clusters[parent].append(concept)

        merges: list[str] = []
        for parent, children in clusters.items():
            if len(children) < 2:
                continue
            children.sort(key=lambda concept: concept.salience, reverse=True)
            cluster_name = f"cluster:via:{parent}"
            cluster = self._concepts.get(cluster_name)
            if cluster is None:
                cluster = Concept(name=cluster_name, kind="cluster", level=5)
                self._concepts[cluster_name] = cluster
            for child in children[:5]:
                cluster.parents.add(child.name)
                cluster.support += 1
                cluster.reward_total += child.reward_mean
            merges.append(cluster_name)

        decayed: list[str] = []
        for name, concept in list(self._concepts.items()):
            if concept.support <= 1 and concept.level <= 1 and concept.salience < 0.5:
                # Soft decay: shrink support so it can naturally fade
                concept.reward_total *= 0.9
                if concept.salience < 0.1:
                    decayed.append(name)

        return {"clusters_formed": merges, "concepts_decayed": decayed}

    def by_kind(self) -> dict[str, list[Concept]]:
        groups: dict[str, list[Concept]] = defaultdict(list)
        for concept in self._concepts.values():
            groups[concept.kind].append(concept)
        for group in groups.values():
            group.sort(key=lambda concept: concept.salience, reverse=True)
        return dict(groups)

    def hierarchy(self, limit: int = 50) -> list[Concept]:
        concepts = list(self._concepts.values())
        concepts.sort(key=lambda concept: (concept.level, -concept.salience, concept.name))
        return concepts[:limit]

    def get(self, name: str) -> Concept | None:
        return self._concepts.get(name)

    def _record(
        self,
        name: str,
        kind: str,
        level: int,
        reward: float,
        example: dict[str, Any],
        parents: set[str] | None = None,
    ) -> None:
        concept = self._concepts.get(name)
        if concept is None:
            concept = Concept(name=name, kind=kind, level=level)
            self._concepts[name] = concept
        concept.level = max(concept.level, level)
        concept.add(reward, example, parents)
