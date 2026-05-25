from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from darwin.types import Action, State


ROOM_ACTIONS = [
    Action("open_curtains", cost=0.03, description="Let daylight into the room.", metadata={"domain": "room"}),
    Action("close_curtains", cost=0.03, description="Block daylight from the room.", metadata={"domain": "room"}),
    Action("toggle_switch", cost=0.05, description="Toggle the electric light switch.", metadata={"domain": "room"}),
    Action("replace_fuse", cost=0.15, description="Restore the fuse if the circuit is broken.", metadata={"domain": "room"}),
    Action("overload_circuit", cost=0.3, description="Stress the circuit and likely break the fuse.", metadata={"domain": "room"}),
    Action("wait", cost=0.01, description="Do nothing and let the world continue.", metadata={"domain": "time"}),
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


UNIVERSE_ACTIONS = [
    Action(
        "room/open_curtains",
        cost=0.03,
        description="Let daylight into the room facet.",
        metadata={"domain": "room", "vocabulary": {"room", "light", "curtains", "brightness", "daylight"}},
    ),
    Action(
        "room/close_curtains",
        cost=0.03,
        description="Block daylight in the room facet.",
        metadata={"domain": "room", "vocabulary": {"room", "dark", "curtains", "daylight"}},
    ),
    Action(
        "room/toggle_switch",
        cost=0.05,
        description="Toggle the electric light in the room facet.",
        metadata={"domain": "room", "vocabulary": {"room", "switch", "electric", "light", "power"}},
    ),
    Action(
        "room/replace_fuse",
        cost=0.15,
        description="Repair the room circuit.",
        metadata={"domain": "room", "vocabulary": {"room", "fuse", "circuit", "repair", "electricity"}},
    ),
    Action(
        "room/overload_circuit",
        cost=0.3,
        description="Stress the room circuit.",
        metadata={"domain": "room", "vocabulary": {"room", "fuse", "circuit", "overload", "electricity"}},
    ),
    Action(
        "math/add_1",
        cost=0.02,
        description="Add one to the numeric state.",
        metadata={"domain": "math", "vocabulary": {"math", "number", "numbers", "arithmetic", "addition", "add", "plus", "increase"}},
    ),
    Action(
        "math/add_2",
        cost=0.02,
        description="Add two to the numeric state.",
        metadata={"domain": "math", "vocabulary": {"math", "number", "numbers", "arithmetic", "addition", "add", "plus", "increase"}},
    ),
    Action(
        "math/subtract_1",
        cost=0.02,
        description="Subtract one from the numeric state.",
        metadata={"domain": "math", "vocabulary": {"math", "number", "numbers", "arithmetic", "subtraction", "subtract", "minus", "decrease"}},
    ),
    Action(
        "math/multiply_2",
        cost=0.03,
        description="Double the numeric state.",
        metadata={"domain": "math", "vocabulary": {"math", "number", "numbers", "arithmetic", "multiply", "times", "double"}},
    ),
    Action(
        "math/multiply_0",
        cost=0.03,
        description="Multiply the numeric state by zero.",
        metadata={"domain": "math", "vocabulary": {"math", "number", "numbers", "arithmetic", "multiply", "zero"}},
    ),
    Action(
        "math/reset",
        cost=0.04,
        description="Reset the numeric state.",
        metadata={"domain": "math", "vocabulary": {"math", "number", "numbers", "arithmetic", "reset", "zero"}},
    ),
    Action(
        "space/push_a_left",
        cost=0.04,
        description="Push block a left.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "push", "left", "position", "physics", "motion"}},
    ),
    Action(
        "space/push_a_right",
        cost=0.04,
        description="Push block a right.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "push", "right", "position", "physics", "motion"}},
    ),
    Action(
        "space/push_b_left",
        cost=0.04,
        description="Push block b left.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "push", "left", "position", "physics", "motion"}},
    ),
    Action(
        "space/push_b_right",
        cost=0.04,
        description="Push block b right.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "push", "right", "position", "physics", "motion"}},
    ),
    Action(
        "space/lift_a",
        cost=0.08,
        description="Lift block a.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "lift", "height", "physics", "gravity"}},
    ),
    Action(
        "space/drop_a",
        cost=0.02,
        description="Drop block a.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "drop", "fall", "floor", "physics", "gravity"}},
    ),
    Action(
        "space/lift_b",
        cost=0.08,
        description="Lift block b.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "lift", "height", "physics", "gravity"}},
    ),
    Action(
        "space/drop_b",
        cost=0.02,
        description="Drop block b.",
        metadata={"domain": "space", "vocabulary": {"space", "block", "blocks", "drop", "fall", "floor", "physics", "gravity"}},
    ),
    Action(
        "time/wait",
        cost=0.01,
        description="Let the shared universe continue.",
        metadata={"domain": "time", "vocabulary": {"time", "wait", "pause", "continue", "change"}},
    ),
]


@dataclass
class UniverseSimulation:
    """One shared environment with several causal facets.

    This is deliberately not a collection of isolated worlds. The state is a
    single flattened universe state, and action names carry facet prefixes so
    Darwin can learn that arithmetic, room dynamics, and spatial motion are
    different regions of one causal field.
    """

    seed: int | None = None
    state: State = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not self.state:
            self.reset()

    def reset(self) -> State:
        self.state = {
            "room.switch_on": False,
            "room.fuse_intact": True,
            "room.curtains_open": False,
            "room.daylight": True,
            "room.room_bright": False,
            "room.battery_charge": 4,
            "math.x": 0,
            "math.last_operand": 0,
            "math.last_operator": "none",
            "math.result_is_even": True,
            "math.result_is_zero": True,
            "space.a.x": 0,
            "space.a.y": 0,
            "space.b.x": 2,
            "space.b.y": 0,
            "space.held": "none",
            "time.ticks": 0,
        }
        self._recompute()
        return self.observe()

    def observe(self) -> State:
        return dict(self.state)

    def possible_actions(self) -> list[Action]:
        return list(UNIVERSE_ACTIONS)

    def apply(self, action: Action) -> tuple[State, float]:
        name = action.name
        if name.startswith("room/"):
            self._apply_room(name.removeprefix("room/"))
        elif name.startswith("math/"):
            self._apply_math(name.removeprefix("math/"))
        elif name.startswith("space/"):
            self._apply_space(name.removeprefix("space/"))
        elif name == "time/wait":
            self.state["time.ticks"] += 1
            if self._rng.random() < 0.04:
                self.state["room.daylight"] = not self.state["room.daylight"]
        else:
            raise ValueError(f"Unknown action: {action.name}")

        self._recompute()
        return self.observe(), self._reward(action)

    def action_metadata(self, action: Action | str) -> dict[str, Any]:
        action_name = action.name if isinstance(action, Action) else action
        for candidate in UNIVERSE_ACTIONS:
            if candidate.name == action_name:
                return {
                    "scope": "world",
                    "world": "universe",
                    "domain": candidate.metadata.get("domain", "universe"),
                }
        return {"scope": "world", "world": "universe", "domain": "universe"}

    def variables_for_domain(self, domain: str) -> list[str]:
        prefix = f"{domain}."
        return sorted(key for key in self.state if key.startswith(prefix))

    def actions_for_terms(self, terms: set[str]) -> list[Action]:
        if not terms:
            return []
        matches: list[tuple[int, Action]] = []
        for action in UNIVERSE_ACTIONS:
            vocabulary = set(action.metadata.get("vocabulary", set()))
            domain = str(action.metadata.get("domain", ""))
            score = len(terms & (vocabulary | {domain}))
            if score:
                matches.append((score, action))
        matches.sort(key=lambda item: (item[0], -item[1].cost), reverse=True)
        return [action for _score, action in matches]

    def _apply_room(self, local_name: str) -> None:
        if local_name == "open_curtains":
            self.state["room.curtains_open"] = True
        elif local_name == "close_curtains":
            self.state["room.curtains_open"] = False
        elif local_name == "toggle_switch":
            self.state["room.switch_on"] = not self.state["room.switch_on"]
            if self.state["room.switch_on"] and self.state["room.battery_charge"] > 0:
                self.state["room.battery_charge"] -= 1
        elif local_name == "replace_fuse":
            self.state["room.fuse_intact"] = True
        elif local_name == "overload_circuit":
            self.state["room.fuse_intact"] = False
            self.state["room.switch_on"] = False
        else:
            raise ValueError(f"Unknown room action: {local_name}")

    def _apply_math(self, local_name: str) -> None:
        x = int(self.state["math.x"])
        operand = 0
        operator = local_name
        if local_name == "add_1":
            operand = 1
            x += operand
        elif local_name == "add_2":
            operand = 2
            x += operand
        elif local_name == "subtract_1":
            operand = 1
            x -= operand
        elif local_name == "multiply_2":
            operand = 2
            x *= operand
        elif local_name == "multiply_0":
            operand = 0
            x *= operand
        elif local_name == "reset":
            x = 0
        else:
            raise ValueError(f"Unknown math action: {local_name}")
        self.state["math.x"] = x
        self.state["math.last_operand"] = operand
        self.state["math.last_operator"] = operator

    def _apply_space(self, local_name: str) -> None:
        parts = local_name.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unknown space action: {local_name}")
        verb = parts[0]
        block = parts[1]
        if block not in {"a", "b"}:
            raise ValueError(f"Unknown block: {block}")
        x_key = f"space.{block}.x"
        y_key = f"space.{block}.y"
        if verb == "push":
            direction = parts[2] if len(parts) > 2 else ""
            if self.state["space.held"] == block:
                return
            if direction == "left":
                self.state[x_key] -= 1
            elif direction == "right":
                self.state[x_key] += 1
            else:
                raise ValueError(f"Unknown push direction: {direction}")
        elif verb == "lift":
            self.state["space.held"] = block
            self.state[y_key] = 1
        elif verb == "drop":
            if self.state["space.held"] == block:
                self.state["space.held"] = "none"
            self.state[y_key] = 0
        else:
            raise ValueError(f"Unknown space action: {local_name}")

    def _recompute(self) -> None:
        electric_light = (
            self.state["room.switch_on"]
            and self.state["room.fuse_intact"]
            and self.state["room.battery_charge"] > 0
        )
        daylight = self.state["room.curtains_open"] and self.state["room.daylight"]
        self.state["room.room_bright"] = bool(electric_light or daylight)
        x = int(self.state["math.x"])
        self.state["math.result_is_even"] = x % 2 == 0
        self.state["math.result_is_zero"] = x == 0

    def _reward(self, action: Action) -> float:
        reward = -action.cost
        domain = action.metadata.get("domain")
        if domain == "room" and self.state["room.room_bright"]:
            reward += 0.7
        if domain == "math":
            reward += 0.15 if self.state["math.x"] != 0 else 0.05
            if abs(int(self.state["math.x"])) > 20:
                reward -= 0.2
        if domain == "space":
            reward += 0.1
            if self.state["space.a.y"] == 0 and self.state["space.b.y"] == 0:
                reward += 0.05
        if not self.state["room.fuse_intact"]:
            reward -= 0.25
        return reward
