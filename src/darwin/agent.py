from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from darwin.causal import CausalModel
from darwin.experiments import ExperimentEngine, ExperimentProposal
from darwin.memory import Memory
from darwin.planner import CausalPlanner, MultiStepPlan, PlanCandidate
from darwin.self_model import SelfModel, SelfReport
from darwin.semantics import SemanticFrame, SemanticMemory, SemanticParser
from darwin.storage import PersistentStore
from darwin.types import Action, Goal, State, Transition
from darwin.world_model import WorldModel, WorldPrediction


class World(Protocol):
    def observe(self) -> State:
        ...

    def possible_actions(self) -> list[Action]:
        ...

    def apply(self, action: Action) -> tuple[State, float]:
        ...


@dataclass
class Darwin:
    """The Project Darwin kernel: action, consequence, memory, and reflection."""

    actions: list[Action]
    causal_model: CausalModel = field(default_factory=CausalModel)
    memory: Memory = field(default_factory=Memory)
    semantic_parser: SemanticParser = field(default_factory=SemanticParser)
    semantic_memory: SemanticMemory = field(default_factory=SemanticMemory)
    world_model: WorldModel = field(default_factory=WorldModel)
    self_model: SelfModel = field(default_factory=SelfModel)
    experiment_engine: ExperimentEngine | None = None
    store: PersistentStore | None = None
    exploration_rate: float = 0.15
    seed: int | None = None

    def __post_init__(self) -> None:
        self.planner = CausalPlanner(self.causal_model)
        if self.experiment_engine is None:
            self.experiment_engine = ExperimentEngine(self.causal_model)
        if self.store is not None and self.memory.store is None:
            self.memory.store = self.store
        self._rng = random.Random(self.seed)
        self._time = 0

    @classmethod
    def from_store(
        cls,
        actions: list[Action],
        store: PersistentStore,
        seed: int | None = None,
        exploration_rate: float = 0.15,
    ) -> "Darwin":
        darwin = cls(actions=actions, store=store, seed=seed, exploration_rate=exploration_rate)
        darwin.hydrate(store.load_transitions())
        darwin.semantic_memory.load_records(store.load_semantic_records())
        return darwin

    def hydrate(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.learn(transition, persist=False)
            self._time = max(self._time, transition.t + 1)

    def decide(self, state: State, goal: Goal) -> PlanCandidate:
        ranked = self.planner.rank(state, self.actions, goal)
        if not ranked:
            raise ValueError("Darwin cannot decide without actions.")

        if self._rng.random() < self.exploration_rate:
            most_uncertain = sorted(
                ranked,
                key=lambda candidate: (
                    candidate.exploration_value,
                    -self.causal_model.action_count(candidate.action.name),
                ),
                reverse=True,
            )
            return self._rng.choice(most_uncertain[: max(1, min(3, len(most_uncertain)))])

        return ranked[0]

    def learn(self, transition: Transition, persist: bool = True) -> None:
        self.causal_model.learn(transition)
        self.memory.learn(transition, persist=persist)
        self.world_model.learn(transition)
        self.self_model.learn(transition)

    def step(self, world: World, goal: Goal) -> Transition:
        before = world.observe()
        candidate = self.decide(before, goal)
        after, reward = world.apply(candidate.action)
        transition = Transition(
            before=before,
            action=candidate.action.name,
            after=after,
            reward=reward,
            t=self._time,
            metadata={
                "decision": candidate.explain(),
                "predicted_state": candidate.predicted.state,
                "expected_reward": candidate.expected_reward.mean,
                "prediction_confidence": candidate.predicted.confidence,
            },
        )
        self._time += 1
        self.learn(transition)
        return transition

    def run(self, world: World, goal: Goal, steps: int) -> list[Transition]:
        return [self.step(world, goal) for _ in range(steps)]

    def predict(self, state: State, action: Action | str) -> WorldPrediction:
        return self.world_model.predict(state, action, self.causal_model)

    def propose_experiments(
        self,
        state: State,
        goal: Goal | None = None,
        limit: int = 5,
    ) -> list[ExperimentProposal]:
        if self.experiment_engine is None:
            self.experiment_engine = ExperimentEngine(self.causal_model)
        return self.experiment_engine.propose(state, self.actions, goal=goal, limit=limit)

    def plan(
        self,
        state: State,
        goal: Goal,
        horizon: int = 3,
        beam_width: int = 5,
        actions: list[Action] | None = None,
    ) -> MultiStepPlan:
        plan = self.planner.plan_sequence(
            state,
            actions or self.actions,
            goal,
            horizon=horizon,
            beam_width=beam_width,
        )
        if self.store is not None:
            self.store.record_plan(
                {
                    "goal": dict(goal.desired),
                    "actions": [action.name for action in plan.actions],
                    "score": plan.score,
                    "final_state": plan.final_state,
                    "trace": plan.trace,
                }
            )
        return plan

    def reflect(self) -> str:
        reflection = self.self_model.reflect(self.memory, self.causal_model, self.world_model)
        if self.store is not None:
            self.store.record_thought(
                "reflection",
                reflection,
                self.self_model.to_record(self.memory, self.causal_model, self.world_model),
            )
        return reflection

    def self_report(self) -> SelfReport:
        return self.self_model.report(self.memory, self.causal_model, self.world_model)

    def interpret_language(
        self,
        text: str,
        *,
        source: str = "user",
        persist: bool = True,
    ) -> SemanticFrame:
        frame = self.semantic_parser.parse(
            text,
            source=source,
            actions=self.actions,
            known_concepts=[concept.name for concept in self.memory.concepts.hierarchy(limit=100)],
            known_variables=self.world_model.variables.keys(),
        )
        self.semantic_memory.learn(frame)
        if persist and self.store is not None:
            self.store.record_semantic_frame(frame.to_record())
        return frame
