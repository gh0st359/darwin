from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from darwin.agent import Darwin
from darwin.embodiment import ConversationAdapter, EnvironmentAdapter
from darwin.experiments import ExperimentResult
from darwin.language import LanguageCortex
from darwin.storage import PersistentStore
from darwin.types import Action, Goal, Transition


@dataclass
class RuntimeEvent:
    kind: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)


class DarwinRuntime:
    """Always-on local cognition loop for Darwin."""

    def __init__(
        self,
        darwin: Darwin,
        adapter: EnvironmentAdapter,
        goal: Goal,
        store: PersistentStore | None = None,
        interval: float = 2.0,
        event_sink: Callable[[RuntimeEvent], None] | None = None,
    ) -> None:
        self.darwin = darwin
        self.adapter = adapter
        self.goal = goal
        self.store = store or darwin.store
        self.interval = interval
        self.conversation = ConversationAdapter()
        self.language = LanguageCortex()
        self.events: list[RuntimeEvent] = []
        self.event_sink = event_sink
        self.stream_enabled = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="darwin-runtime", daemon=True)
        self._thread.start()
        self._event("runtime", "Darwin's continuous cognition loop is running.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2.0))
        self._event("runtime", "Darwin's continuous cognition loop stopped.")

    def cognition_cycle(self) -> RuntimeEvent:
        with self._lock:
            state = self.adapter.observe()
            proposals = self.darwin.experiment_engine.propose(
                state,
                self.adapter.possible_actions(),
                goal=self.goal,
                limit=3,
            )
            proposal = proposals[0] if proposals else None

            if proposal is not None and proposal.uncertainty >= 0.25:
                before = self.adapter.observe()
                after, reward = self.adapter.apply(proposal.action)
                transition = Transition(
                    before=before,
                    action=proposal.action.name,
                    after=after,
                    reward=reward,
                    t=self._next_time(),
                    metadata={
                        "mode": "active_experiment",
                        "question": proposal.question,
                        "predicted_state": proposal.predicted_state,
                        "expected_reward": proposal.expected_reward,
                    },
                )
                self.darwin.learn(transition)
                result = self.darwin.experiment_engine.evaluate(proposal, transition)
                if self.store is not None:
                    self.store.record_experiment(result.to_record())
                return self._experiment_event(result)

            reflection = self.darwin.reflect()
            return self._event("reflection", reflection)

    def dream(self) -> RuntimeEvent:
        reflection = self.darwin.reflect()
        concepts = self.darwin.memory.concepts.salient(limit=5)
        concept_line = ", ".join(concept.name for concept in concepts) or "no concepts yet"
        return self._event(
            "dream",
            f"{reflection} Consolidated salient concepts: {concept_line}.",
            {"concepts": [concept.to_record() for concept in concepts]},
        )

    def chat(self, message: str) -> str:
        with self._lock:
            if self.store is not None:
                self.store.record_chat("user", message)

            user_frame = self.darwin.interpret_language(message, source="user")
            response = self._respond(message, user_frame)
            darwin_frame = self.darwin.interpret_language(response, source="darwin")
            transition = self.conversation.make_transition(message, response, t=self._next_time())
            transition = Transition(
                before=transition.before,
                action=transition.action,
                after=transition.after,
                reward=transition.reward,
                t=transition.t,
                metadata={
                    **dict(transition.metadata),
                    "user_semantics": user_frame.to_record(),
                    "darwin_semantics": darwin_frame.to_record(),
                },
            )
            self.darwin.learn(transition)

            if self.store is not None:
                self.store.record_chat("darwin", response)

            self._event(
                "chat",
                response,
                {"message_signal": transition.before, "response_signal": transition.after},
            )
            return response

    def recent_events(self, limit: int = 20) -> list[RuntimeEvent]:
        return self.events[-limit:]

    def set_streaming(self, enabled: bool) -> None:
        self.stream_enabled = enabled

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.cognition_cycle()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self._event("error", f"runtime cycle failed: {exc!r}")

    def _respond(self, message: str, semantic_frame) -> str:
        return self.language.respond(
            message=message,
            darwin=self.darwin,
            adapter=self.adapter,
            goal=self.goal,
            recent_events=self.recent_events(limit=5),
            conversation=self.conversation,
            semantic_frame=semantic_frame,
        )

    def _experiment_event(self, result: ExperimentResult) -> RuntimeEvent:
        if result.confirmed:
            content = f"Experiment confirmed: {result.proposal.question}"
        else:
            surprise_list = ", ".join(result.surprises)
            content = f"Experiment produced surprise in {surprise_list}: {result.proposal.question}"
        return self._event("experiment", content, result.to_record())

    def _event(self, kind: str, content: str, payload: dict[str, Any] | None = None) -> RuntimeEvent:
        event = RuntimeEvent(kind=kind, content=content, payload=payload or {})
        self.events.append(event)
        if len(self.events) > 500:
            self.events = self.events[-500:]
        if self.store is not None and kind != "chat":
            self.store.record_thought(kind, content, payload or {})
        if self.event_sink is not None and self.stream_enabled:
            self.event_sink(event)
        return event

    def _next_time(self) -> int:
        value = getattr(self.darwin, "_time", 0)
        setattr(self.darwin, "_time", value + 1)
        return value


def ensure_chat_action(actions: list[Action]) -> list[Action]:
    if any(action.name == "chat_with_user" for action in actions):
        return actions
    return [
        *actions,
        Action("chat_with_user", cost=0.0, description="Exchange language with the user as experience."),
    ]
