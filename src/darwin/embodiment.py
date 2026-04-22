from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from darwin.types import Action, State, Transition
from darwin.worlds import AdaptiveRoomWorld


class EnvironmentAdapter(Protocol):
    name: str

    def observe(self) -> State:
        ...

    def possible_actions(self) -> list[Action]:
        ...

    def apply(self, action: Action) -> tuple[State, float]:
        ...


@dataclass
class RoomSimulationAdapter:
    """Embodiment adapter for Darwin's current simulation body."""

    world: AdaptiveRoomWorld
    name: str = "adaptive_room"

    def observe(self) -> State:
        return self.world.observe()

    def possible_actions(self) -> list[Action]:
        return self.world.possible_actions()

    def apply(self, action: Action) -> tuple[State, float]:
        return self.world.apply(action)


class ConversationAdapter:
    """Turns user conversation into structured experience Darwin can remember."""

    name = "conversation"

    def __init__(self) -> None:
        self.turn = 0

    def signal(self, message: str) -> State:
        lowered = message.lower()
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", lowered)
        topic = self._topic(tokens)
        intent = self._intent(lowered)
        return {
            "channel": "chat",
            "turn": self.turn,
            "topic": topic,
            "intent": intent,
            "message_length": len(message),
            "token_count": len(tokens),
            "mentions_darwin": "darwin" in tokens,
            "mentions_learning": any(token in {"learn", "learning", "teach", "evolve"} for token in tokens),
        }

    def make_transition(self, message: str, response: str, t: int) -> Transition:
        before = self.signal(message)
        self.turn += 1
        after = {
            **before,
            "turn": self.turn,
            "response_length": len(response),
            "response_mode": self._response_mode(response),
            "conversation_active": True,
        }
        reward = 0.2
        if before["mentions_learning"]:
            reward += 0.2
        if before["mentions_darwin"]:
            reward += 0.1
        return Transition(
            before=before,
            action="chat_with_user",
            after=after,
            reward=reward,
            t=t,
            metadata={"user_message": message, "darwin_response": response},
        )

    def _topic(self, tokens: list[str]) -> str:
        topics = {
            "architecture": {"architecture", "system", "kernel", "model", "design"},
            "memory": {"memory", "remember", "dream", "consolidate"},
            "planning": {"plan", "planning", "goal", "future", "simulate"},
            "experiments": {"experiment", "test", "uncertain", "hypothesis"},
            "self": {"self", "metacognition", "aware", "status", "belief"},
            "tools": {"tool", "browser", "robot", "file", "api"},
        }
        token_set = set(tokens)
        for name, keywords in topics.items():
            if token_set & keywords:
                return name
        return "general"

    def _intent(self, lowered: str) -> str:
        if lowered.startswith("/"):
            return "command"
        if "?" in lowered:
            return "question"
        if any(word in lowered for word in ["teach", "remember", "learn this"]):
            return "teaching"
        if any(word in lowered for word in ["do", "run", "start", "build"]):
            return "directive"
        return "conversation"

    def _response_mode(self, response: str) -> str:
        if "I am" in response or "My " in response:
            return "introspective"
        if "I will" in response:
            return "directive_ack"
        return "conversation"

