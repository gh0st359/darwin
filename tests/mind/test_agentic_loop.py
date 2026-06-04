"""AgenticLoop — bounded multi-step reasoning."""

from __future__ import annotations

from darwin.mind.agentic_loop import AgenticLoop
from darwin.mysterio.bus import BusTopic, CognitionBus
from darwin.neural.learned_space import LearnedCausalSpace


class _FakeDispatcher:
    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls = 0

    def dispatch(self, problem):
        self.calls += 1
        if not self.answers:
            return None

        class _Trace:
            pass

        t = _Trace()
        t.answer = self.answers.pop(0)
        t.conclusion = ""
        return t


class _FakeRuntime:
    def __init__(self, *, dispatcher=None):
        self.embedding_space = LearnedCausalSpace(dim=8, seed=1)
        self.embedding_space.train_tokens(["solve", "problem", "step"])
        self.bus = CognitionBus()
        self.reasoning_dispatcher = dispatcher
        self.universe = None


def test_loop_converges_when_dispatcher_returns_answer():
    runtime = _FakeRuntime(dispatcher=_FakeDispatcher(answers=["forty-two"]))
    loop = AgenticLoop(runtime, max_steps=5)
    state = loop.run("what is the answer to life?")
    assert state.succeeded is True
    assert state.answer == "forty-two"
    assert state.reason_stopped == "converged"
    assert state.step_index >= 1


def test_loop_stops_on_step_budget():
    runtime = _FakeRuntime(dispatcher=_FakeDispatcher(answers=[]))
    loop = AgenticLoop(runtime, max_steps=3)
    state = loop.run("an unsolved problem")
    assert state.succeeded is False
    assert state.reason_stopped == "step_budget"
    assert state.step_index == 3


def test_loop_stuck_without_dispatcher():
    runtime = _FakeRuntime(dispatcher=None)
    loop = AgenticLoop(runtime, max_steps=4)
    state = loop.run("hello?")
    assert state.succeeded is False
    assert state.reason_stopped == "stuck"


def test_loop_publishes_mind_step_events():
    runtime = _FakeRuntime(dispatcher=_FakeDispatcher(answers=["x"]))
    events: list = []
    runtime.bus.subscribe(BusTopic.MIND_STEP, lambda e: events.append(e))
    loop = AgenticLoop(runtime, max_steps=3)
    loop.run("anything")
    assert len(events) >= 1
    assert "step_index" in events[0].payload


def test_loop_handles_empty_problem_text():
    runtime = _FakeRuntime(dispatcher=_FakeDispatcher(answers=[]))
    loop = AgenticLoop(runtime, max_steps=2)
    state = loop.run("")
    # Empty input → no tokens → stuck.
    assert state.reason_stopped in ("stuck", "step_budget")
