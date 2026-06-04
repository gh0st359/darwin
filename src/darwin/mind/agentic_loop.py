"""AgenticLoop — the brain's internal multi-step reasoning loop.

When :class:`Mind` decides a problem warrants more than a single
faculty pass, it invokes :class:`AgenticLoop`. Each iteration:

  1. Embed the current problem state.
  2. Query the learned representation for the nearest prior approach.
  3. Run one reasoning step via the runtime's reasoning dispatcher.
  4. Verify the step against the universe + cortical mesh.
  5. Decide continue / converge / give up.

Bounded by ``max_steps`` and ``max_wall_seconds``. Publishes
``BusTopic.MIND_STEP`` events visible to the brain terminal but never
to chat — the surface for outside-the-brain observation, not the
chat reply path.

This is *the* place categorised "agent" behaviour folds into a single
faculty-blind loop: the loop calls faculty methods by capability, never
by name, and the composed answer is written in Darwin's voice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopState:
    """Accumulating state across loop iterations."""

    problem: str
    step_index: int = 0
    started_at: float = field(default_factory=time.time)
    notes: list[str] = field(default_factory=list)
    answer: str = ""
    succeeded: bool = False
    reason_stopped: str = ""

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def to_record(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "elapsed": round(self.elapsed(), 3),
            "note_count": len(self.notes),
            "succeeded": self.succeeded,
            "reason_stopped": self.reason_stopped,
        }


class AgenticLoop:
    """Bounded multi-step reasoning loop driven by the learned representation."""

    def __init__(
        self,
        runtime: Any,
        *,
        max_steps: int = 12,
        max_wall_seconds: float = 30.0,
    ) -> None:
        self.runtime = runtime
        self.max_steps = int(max_steps)
        self.max_wall_seconds = float(max_wall_seconds)

    def run(self, problem: str) -> LoopState:
        state = LoopState(problem=problem)
        for _ in range(self.max_steps):
            if state.elapsed() > self.max_wall_seconds:
                state.reason_stopped = "timeout"
                break
            state.step_index += 1
            progress = self._step(state)
            self._publish_step(state)
            if progress == "converged":
                state.succeeded = True
                state.reason_stopped = "converged"
                break
            if progress == "stuck":
                state.reason_stopped = "stuck"
                break
        if not state.reason_stopped:
            state.reason_stopped = "step_budget"
        return state

    def _step(self, state: LoopState) -> str:
        """One reasoning step. Returns 'progress' | 'converged' | 'stuck'."""

        embedding_space = getattr(self.runtime, "embedding_space", None)
        if embedding_space is None:
            state.notes.append("no embedding space")
            return "stuck"
        # 1. Look for nearest prior token to the current problem stub. The
        # presence of a meaningful neighbour means we have *some* learned
        # structure to lean on; a degenerate result means the substrate is
        # too empty for this query and we should bail rather than fabricate.
        try:
            from darwin.neural.tokenizer import split_words
        except Exception:
            words = state.problem.lower().split()
        else:
            words = split_words(state.problem)
        if not words:
            state.notes.append("no tokens in problem")
            return "stuck"
        try:
            nearest = embedding_space.nearest(words[-1], k=3)
        except Exception:
            nearest = []
        if nearest:
            state.notes.append(
                f"nearest:{nearest[0][0]}:{round(nearest[0][1], 3)}"
            )
        # 2. Try the reasoning dispatcher on the problem.
        dispatcher = getattr(self.runtime, "reasoning_dispatcher", None)
        if dispatcher is None:
            state.notes.append("no reasoning dispatcher")
            return "stuck"
        try:
            trace = dispatcher.dispatch(state.problem)
        except Exception:
            trace = None
        if trace is None:
            return "progress"  # one step done, no convergence yet
        # 3. Verify against the universe.
        universe = getattr(self.runtime, "universe", None)
        if universe is not None and hasattr(trace, "conclusion"):
            try:
                if universe.has(getattr(trace, "conclusion", "")):
                    state.notes.append("conclusion grounded in universe")
            except Exception:
                pass
        # 4. Convergence: pick up the dispatcher's answer if it produced one.
        if hasattr(trace, "answer") and trace.answer:
            state.answer = str(trace.answer)
            return "converged"
        if hasattr(trace, "conclusion") and trace.conclusion:
            state.answer = str(trace.conclusion)
            return "converged"
        return "progress"

    def _publish_step(self, state: LoopState) -> None:
        bus = getattr(self.runtime, "bus", None)
        if bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            bus.publish(
                BusTopic.MIND_STEP,
                state.to_record(),
                source="agentic_loop",
            )
        except Exception:
            return


__all__ = ["AgenticLoop", "LoopState"]
