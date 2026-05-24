# Embodiment Adapters

Darwin's mind is independent of *which* world it lives in. The
`EnvironmentAdapter` Protocol is the seam. Any object implementing
it can be plugged in as Darwin's body.

## The Protocol

```python
class EnvironmentAdapter(Protocol):
    name: str
    def observe(self) -> State: ...
    def possible_actions(self) -> list[Action]: ...
    def apply(self, action: Action) -> tuple[State, float]: ...
```

Three methods, no return types you can't see. `State` is just a
`dict[str, Any]`, `Action` is a tiny frozen dataclass, `reward` is a
float.

## `AdaptiveRoomWorld` + `RoomSimulationAdapter`

The default. A deterministic room with a switch, curtains, a fuse,
daylight, a battery, and a `room_bright` outcome. Action set:

| Action | Cost | Effect |
| --- | --- | --- |
| `open_curtains` | 0.03 | sets `curtains_open=True` |
| `close_curtains` | 0.03 | sets `curtains_open=False` |
| `toggle_switch` | 0.05 | toggles `switch_on`, drains battery if on |
| `replace_fuse` | 0.15 | sets `fuse_intact=True` |
| `overload_circuit` | 0.30 | sets `fuse_intact=False`, `switch_on=False` |
| `wait` | 0.01 | passes time; daylight flips occasionally |

`room_bright` is recomputed each step from
`(switch_on AND fuse_intact AND battery_charge > 0) OR (curtains_open
AND daylight)`. Reward is `-action.cost + (1.0 if room_bright) +
(-0.5 if not fuse_intact) + (-0.1 if battery_charge <= 1)`.

This world is *intentionally* small. It has just enough conditional
structure that the causal model has to learn real dependencies
(toggling the switch does nothing without a fuse; the room is bright
either via electricity or via curtains+daylight), and just enough
goal complexity that planning matters.

## `ConversationAdapter`

A second adapter used for chat. It does not implement the full
`EnvironmentAdapter` protocol (it does not have a meaningful
`observe()`); instead it provides `signal(message)` and
`make_transition(message, response, t)` helpers used by
`DarwinRuntime.chat(...)` to turn each chat turn into a `Transition`
so language exchanges become *experience* the rest of the system can
learn from.

This is why "explain a concept" in chat shows up as observations in
`/status` — every conversation is a transition.

## Writing your own adapter

```python
from darwin.types import Action, State

class MyWorldAdapter:
    name = "my_world"

    def __init__(self, my_world):
        self.world = my_world

    def observe(self) -> State:
        # any dict of strings -> primitives / nested dicts
        return self.world.snapshot()

    def possible_actions(self) -> list[Action]:
        return [
            Action("primary_button", cost=0.01, description="Press the primary button."),
            Action("secondary_button", cost=0.01, description="Press the other button."),
            Action("wait", cost=0.0, description="Do nothing."),
        ]

    def apply(self, action: Action) -> tuple[State, float]:
        self.world.do(action.name)
        reward = self.world.score()
        return self.observe(), reward
```

Then plug it in:

```python
from darwin.agent import Darwin
from darwin.runtime import DarwinRuntime, ensure_chat_action

adapter = MyWorldAdapter(my_world)
actions = ensure_chat_action(adapter.possible_actions())
darwin = Darwin(actions=actions, seed=42)
runtime = DarwinRuntime(darwin, adapter, goal=Goal(desired={...}))
runtime.start()
```

The 5 background loops, the causal chain engine, the planner, the
self-modification engine, the dream loop — none of them know or care
what world they're in. They all just see `Action` and `State`.

## Robotics / external policy use

`Darwin.plan(state, goal, horizon)` returns a `MultiStepPlan` with a
`causal_chain` attached. `ResponsePlan.to_dlm_payload()` returns the
language plan. A robotics integration can:

- consume `multi_step_plan.actions` as the action sequence for an
  external policy
- consume `multi_step_plan.causal_chain.chain_confidence` and
  `chain_uncertainty` as gating signals (abort below threshold)
- consume `response_plan.to_dlm_payload()` for human-facing
  explanations without ever instantiating a language renderer

The DLM is genuinely optional. Darwin can be a pure planning kernel
that outputs structured plans for downstream consumers, with no
language layer at all.

## Multiple bodies

Nothing in the architecture forbids running multiple adapters
concurrently (e.g. one simulated body + one conversation adapter,
which is exactly what `DarwinRuntime` already does). For more
ambitious setups (multiple physical agents sharing one mind), you'd
want to extend `EnvironmentAdapter` with an embodiment identifier and
tag transitions with which body they came from. That is a v3
direction; v2 keeps it simple with a primary world adapter + the
conversation adapter for chat-as-experience.
