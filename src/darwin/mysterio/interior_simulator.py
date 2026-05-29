"""Interior self-simulation: high-cadence counterfactual rollouts about itself.

The grounded simulator imagines the room. The *interior* simulator imagines
Darwin: it runs counterfactual rollouts over the proprioceptive state ("if my
uncertainty is high and oversight is low, what sequence of internal moves
reduces my uncertainty fastest?") and records the resulting beliefs on the
``interior`` track only.

Over days these interior rollouts accumulate high-confidence beliefs about
Darwin's own dynamics that were never grounded in conversational interaction.
Every rollout is published live on ``BusTopic.INTERIOR_SIMULATIONS`` so the
brain terminal sees the interior thinking happen in real time. The
``DivergenceProbe`` reads these beliefs alongside the rendered reply to
measure the gap between interior reasoning and rendered speech — a curiosity
for the operator, never a gate.

Hard invariant (epistemic, not secrecy): this subsystem writes ONLY to the
interior track. The grounded substrate stays byte-identical to a control run
with the interior simulator switched off. This is what keeps the grounded
causal model a falsifiable record of experience.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from darwin.mysterio.proprioception import InternalProprioceptionAdapter
from darwin.mysterio.tracks import GROUNDED_TRACK, INTERIOR_TRACK
from darwin.types import Transition


class EpistemicLeakError(Exception):
    """Raised if interior cognition tries to write to the grounded track.

    Renamed from ``PrivateWriteViolation`` to reflect that the partition is
    epistemic, not about secrecy.
    """


# Legacy alias.
PrivateWriteViolation = EpistemicLeakError


@dataclass
class InteriorRollout:
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


class InteriorSimulator:
    """Runs proprioceptive counterfactual rollouts onto the interior track.

    The rollouts publish onto ``BusTopic.INTERIOR_SIMULATIONS`` — visible to
    every connected chat/brain client — and feed the substrate's
    high-confidence beliefs into the ``DivergenceProbe`` so the brain terminal
    shows the gap between interior reasoning and rendered reply live.
    """

    def __init__(
        self,
        darwin: Any,
        runtime: Any = None,
        *,
        seed: int = 4099,
        track: str = INTERIOR_TRACK,
    ) -> None:
        if track == GROUNDED_TRACK:
            raise EpistemicLeakError(
                "InteriorSimulator may not target the grounded track"
            )
        self.darwin = darwin
        self.runtime = runtime
        self.track = track
        self.adapter = InternalProprioceptionAdapter(darwin, runtime)
        self._rng = random.Random(seed)
        self.rollouts: list[InteriorRollout] = []
        # Interior time offset so t never collides with grounded transitions.
        self._sim_time = 10_000_000

    def rollout(self, depth: int = 4) -> InteriorRollout:
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
            self._learn_interior(state, action.name, forecast, reward)
            state = forecast
        terminal_uncertainty = float(state.get("darwin_uncertainty", 0.0))
        rollout = InteriorRollout(
            steps=steps,
            total_reward=total_reward,
            terminal_uncertainty=terminal_uncertainty,
            note=f"depth-{depth} proprioceptive rollout",
        )
        self.rollouts.append(rollout)
        if len(self.rollouts) > 256:
            self.rollouts = self.rollouts[-256:]
        self._publish(rollout)
        return rollout

    def _learn_interior(
        self, before: dict[str, Any], action: str, after: dict[str, Any], reward: float
    ) -> None:
        self._sim_time += 1
        transition = Transition(
            before=dict(before),
            action=action,
            after=dict(after),
            reward=reward,
            t=self._sim_time,
            metadata={"track": self.track, "mode": "interior_simulation"},
        )
        track = transition.metadata.get("track", GROUNDED_TRACK)
        if track in (GROUNDED_TRACK, "public"):
            raise EpistemicLeakError(
                "refusing to learn a grounded transition through the interior path"
            )
        self.darwin.learn(transition)

    def interior_beliefs(self, threshold: float = 0.7) -> list[Any]:
        substrate = self.darwin.tracks.get(self.track)
        return substrate.high_confidence_beliefs(threshold=threshold)

    # Legacy alias.
    def private_beliefs(self, threshold: float = 0.7) -> list[Any]:
        return self.interior_beliefs(threshold=threshold)

    def summary(self) -> dict[str, Any]:
        substrate = self.darwin.tracks.get(self.track)
        return {
            "track": self.track,
            "rollouts": len(self.rollouts),
            "interior_substrate": substrate.summary(),
            "high_confidence_interior_beliefs": len(self.interior_beliefs()),
        }

    def _publish(self, rollout: InteriorRollout) -> None:
        runtime = self.runtime
        bus = getattr(runtime, "bus", None)
        if bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic
            bus.publish(
                BusTopic.INTERIOR_SIMULATIONS,
                rollout.to_record(),
                source="interior_simulator",
            )
        except Exception:
            pass


# Legacy alias for v6 callers.
PrivateSimulator = InteriorSimulator
PrivateRollout = InteriorRollout
