from __future__ import annotations

import random
from dataclasses import dataclass, field

from darwin.types import Action, State


ROOM_ACTIONS = [
    Action("open_curtains", cost=0.03, description="Let daylight into the room."),
    Action("close_curtains", cost=0.03, description="Block daylight from the room."),
    Action("toggle_switch", cost=0.05, description="Toggle the electric light switch."),
    Action("replace_fuse", cost=0.15, description="Restore the fuse if the circuit is broken."),
    Action("overload_circuit", cost=0.3, description="Stress the circuit and likely break the fuse."),
    Action("wait", cost=0.01, description="Do nothing and let the world continue."),
]


@dataclass
class AdaptiveRoomWorld:
    """A deterministic room where actions have conditional consequences."""

    seed: int | None = None
    state: State = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not self.state:
            self.reset()

    def reset(self) -> State:
        self.state = {
            "switch_on": False,
            "fuse_intact": True,
            "curtains_open": False,
            "daylight": True,
            "room_bright": False,
            "battery_charge": 4,
        }
        self._recompute()
        return self.observe()

    def observe(self) -> State:
        return dict(self.state)

    def possible_actions(self) -> list[Action]:
        return list(ROOM_ACTIONS)

    def apply(self, action: Action) -> tuple[State, float]:
        if action.name == "open_curtains":
            self.state["curtains_open"] = True
        elif action.name == "close_curtains":
            self.state["curtains_open"] = False
        elif action.name == "toggle_switch":
            self.state["switch_on"] = not self.state["switch_on"]
            if self.state["switch_on"] and self.state["battery_charge"] > 0:
                self.state["battery_charge"] -= 1
        elif action.name == "replace_fuse":
            self.state["fuse_intact"] = True
        elif action.name == "overload_circuit":
            self.state["fuse_intact"] = False
            self.state["switch_on"] = False
        elif action.name == "wait":
            if self.state["switch_on"] and self.state["battery_charge"] > 0:
                self.state["battery_charge"] -= 1
            if self._rng.random() < 0.05:
                self.state["daylight"] = not self.state["daylight"]
        else:
            raise ValueError(f"Unknown action: {action.name}")

        self._recompute()
        reward = self._reward(action)
        return self.observe(), reward

    def _recompute(self) -> None:
        electric_light = (
            self.state["switch_on"]
            and self.state["fuse_intact"]
            and self.state["battery_charge"] > 0
        )
        daylight = self.state["curtains_open"] and self.state["daylight"]
        self.state["room_bright"] = bool(electric_light or daylight)

    def _reward(self, action: Action) -> float:
        reward = -action.cost
        if self.state["room_bright"]:
            reward += 1.0
        if not self.state["fuse_intact"]:
            reward -= 0.5
        if self.state["battery_charge"] <= 1:
            reward -= 0.1
        return reward

