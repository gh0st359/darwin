# Testing

Darwin has 44 tests across the cognitive stack. They all use the
stdlib `unittest` runner, run in under 6 seconds, and have no
external dependencies.

```bash
python -m unittest discover tests
# Ran 44 tests in <6s. OK
```

## What is covered

| File | Tests | What they assert |
| --- | --- | --- |
| `tests/test_agent.py` | 2 | the agent picks learned brightening actions; experience is recorded |
| `tests/test_causal.py` | 2 | conditional toggle effects are learned; expected reward uses observed payoff |
| `tests/test_semantics.py` | 4 | parser extracts hypotheses, goals, groundings; runtime chat stores both user and self semantics; semantic memory persists |
| `tests/test_language_cognition.py` | 3 | retrieved semantic memory is used without parser leaks; streaming speaker works; thought trace + critic happen before responding |
| `tests/test_v02.py` | 7 | persistent SQLite store round-trips; hierarchical concepts form; experiment engine prefers underexplored; runtime chat is recorded as experience; event sink fires; prediction-failure priority resolves; long-horizon plan returns a sequence |
| `tests/test_v2.py` | 22 | Phase 0–5 of v2 (structured plans, JSONL logging, causal chains, advanced retrieval, self-modification, multi-threaded runtime, persistent state, DLM rendering, faithfulness validation, fallback, training-data collection) |
| `tests/test_brain_daemon.py` | 4 | client chats over the socket; subscribed clients receive background events; **unsubscribed clients receive zero events**; two clients share one brain |

## Patterns worth knowing

### Building a seeded Darwin

```python
def _seed_basic_world():
    world = AdaptiveRoomWorld(seed=11)
    adapter = RoomSimulationAdapter(world)
    actions = ensure_chat_action(adapter.possible_actions())
    darwin = Darwin(actions=actions, seed=11, exploration_rate=0.1)
    darwin.learn(Transition(
        before={..., "room_bright": False, ...},
        action="open_curtains",
        after={..., "room_bright": True, ...},
        reward=1.0, t=0,
    ))
    return darwin, adapter, Goal(desired={"room_bright": True})
```

This pattern (see `tests/test_v2.py`) gives you a Darwin with one
grounded causal belief, which is enough to exercise discourse
planning, retrieval, the DLM payload, and the validator without
needing a real run.

### Running background loops in a test

```python
runtime = DarwinRuntime(
    darwin=darwin, adapter=adapter, goal=goal,
    interval=0.1,
    state_path=None,
    loop_intervals={
        "experiment": 0.1, "simulation": 0.15, "dream": 0.2,
        "self_modification": 0.3, "uncertainty": 0.15,
    },
)
runtime.start()
time.sleep(0.6)
self.assertTrue(runtime.running)
self.assertGreaterEqual(len(runtime._threads), 5)
runtime.stop()
```

The runtime constructor accepts `loop_intervals` so tests can sub-second
the cadences and confirm all five threads actually fire within a
short window.

### Asserting the chat window is clean

This is the critical regression test for the v2 UX:

```python
def test_unsubscribed_client_receives_no_events(self) -> None:
    """Default chat client must not receive the background firehose."""
    daemon = _build_daemon(tmpdir, port)
    daemon.start()
    try:
        client = DarwinClient(host="127.0.0.1", port=port)
        received = []
        client.connect(received.append)
        # NOTE: do NOT call subscribe_events. Background loops fire at
        # <0.5s intervals so 1.5s gives them many chances to leak.
        time.sleep(1.5)
        client.close()
        event_messages = [m for m in received if m.get("type") == "event"]
        self.assertEqual(event_messages, [])
    finally:
        daemon.stop()
```

If this ever fails, the chat REPL is broken — the user should not see
brain background events leak into their conversation.

### Faithfulness validator unit tests

```python
plan = ResponsePlan(mode="belief_answer", intent="x", thesis="y",
                    confidence=0.7, causal_claims=[
    CausalClaim(action="open_curtains", variable="room_bright",
                effect="True", confidence=0.9, samples=4),
])
valid, notes = FaithfulnessValidator().validate(plan, "Just some unrelated commentary.")
assert not valid
assert any("causal claim" in note for note in notes)
```

These tests pin down the contract that the renderer must obey, and
make any regression in the validator's strictness immediately
visible.

## Running a single file

```bash
python -m unittest tests.test_brain_daemon
python -m unittest tests.test_brain_daemon.BrainDaemonTests.test_unsubscribed_client_receives_no_events
python -m unittest tests.test_v2.Phase3DLMTests -v
```

## When to add a test

- a new mode in `DiscoursePlanner`
- a new validator rule
- a new background loop or change to an existing loop's cadence
- a new field on `ResponsePlan` that downstream code reads
- any change to the daemon's wire protocol

Tests in v2 are written to be readable, fast, and have minimal
fixtures. Keep that style.
