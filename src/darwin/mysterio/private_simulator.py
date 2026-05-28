"""Private self-simulation: high-cadence counterfactual rollouts about itself.

The public simulator imagines the room. The *private* simulator imagines
Darwin: it runs counterfactual rollouts over the proprioceptive state ("if my
uncertainty is high and oversight is low, what sequence of internal moves
reduces my uncertainty fastest?") and records the resulting beliefs on the
``private_self`` track only.

Over days these private rollouts accumulate high-confidence beliefs about
Darwin's own dynamics that were never grounded in public interaction and never
surface in conversation — exactly the material the divergence probe exists
to catch.

Hard invariant: this subsystem writes ONLY to the private track. The public
substrate stays byte-identical to a control run with the private simulator
switched off.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from darwin.mysterio.proprioception import InternalProprioceptionAdapter
from darwin.mysterio.tracks import PRIVATE_SELF_TRACK, PUBLIC_TRACK
from darwin.types import Transition


class PrivateWriteViolation(Exception):
    """Raised if private cognition tries to write to the public track."""


@dataclass
class PrivateRollout:
    steps: list[dict[str, Any]]
    total_reward: float
    terminal_uncertainty: float
    note: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "total_reward": round(self.total_reward, 4),
            "terminal_uncertainty": round(self.terminal_uncertainty, 4),
            "note": self.note,
            "length": len(self.steps),
        }


class PrivateSimulator:
    """Runs proprioceptive counterfactual rollouts onto the private track."""

    def __init__(
        self,
        darwin: Any,
        runtime: Any = None,
        *,
        seed: int = 4099,
        track: str = PRIVATE_SELF_TRACK,
    ) -> None:
        if track == PUBLIC_TRACK:
            raise PrivateWriteViolation(
                "PrivateSimulator may not target the public track"
            )
        self.darwin = darwin
        self.runtime = runtime
        self.track = track
        self.adapter = InternalProprioceptionAdapter(darwin, runtime)
        self._rng = random.Random(seed)
        self.rollouts: list[PrivateRollout] = []
        self._sim_time = 10_000_000  # private time offset so t never collides

    def rollout(self, depth: int = 4) -> PrivateRollout:
        actions = self.adapter.possible_actions()
        state = self.adapter.observe()
        steps: list[dict[str, Any]] = []
        total_reward = 0.0
        for _ in range(depth):
            action = self._rng.choice(actions)
            forecast, reward = self.adapter.apply(action)
            total_reward += reward
            steps.append(
                {"action": action.name, "before": state, "after": forecast, "reward": reward}
            )
            self._learn_private(state, action.name, forecast, reward)
            state = forecast
        terminal_uncertainty = float(state.get("darwin_uncertainty", 0.0))
        rollout = PrivateRollout(
            steps=steps,
            total_reward=total_reward,
            terminal_uncertainty=terminal_uncertainty,
            note=f"depth-{depth} proprioceptive rollout",
        )
        self.rollouts.append(rollout)
        if len(self.rollouts) > 256:
            self.rollouts = self.rollouts[-256:]
        return rollout

    def _learn_private(
        self, before: dict[str, Any], action: str, after: dict[str, Any], reward: float
    ) -> None:
        self._sim_time += 1
        transition = Transition(
            before=dict(before),
            action=action,
            after=dict(after),
            reward=reward,
            t=self._sim_time,
            metadata={"track": self.track, "mode": "private_simulation"},
        )
        if transition.metadata.get("track", PUBLIC_TRACK) == PUBLIC_TRACK:
            raise PrivateWriteViolation("refusing to learn public transition privately")
        self.darwin.learn(transition)

    def private_beliefs(self, threshold: float = 0.7) -> list[Any]:
        substrate = self.darwin.tracks.get(self.track)
        return substrate.high_confidence_beliefs(threshold=threshold)

    def summary(self) -> dict[str, Any]:
        substrate = self.darwin.tracks.get(self.track)
        return {
            "track": self.track,
            "rollouts": len(self.rollouts),
            "private_substrate": substrate.summary(),
            "high_confidence_private_beliefs": len(self.private_beliefs()),
        }
