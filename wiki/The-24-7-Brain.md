# The 24/7 Brain

`darwin/runtime.py` turns Darwin from a request-response system into a
mind that keeps thinking. The runtime owns five concurrent background
threads, each with its own cadence, each producing visible thoughts
that get logged to disk.

## The five loops

| Loop | Default interval | Method | What it does |
| --- | --- | --- | --- |
| `experiment` | `interval` | `_loop_experiment` | proposes the highest-uncertainty intervention via `ExperimentEngine`, applies it to the embodiment (`adapter.apply`), creates a `Transition`, learns from it, evaluates the proposal against the result. |
| `simulation` | `1.5 × interval` | `_loop_simulation` | runs `CausalChainEngine.explore_chains(...)` with depth 3 and beam 4. Stores the best chain. **Cross-feed**: the highest-uncertainty step in the imagined chain registers as a prediction-failure signal in `SelfModel.prediction_failures` so `learning_priority` reflects what the mind is unsure about even when it has not acted on it yet. |
| `dream` | `4 × interval` | `_loop_dream` | calls `Darwin.reflect()` + `ConceptIndex.consolidate()`. Consolidation clusters affordances into level-5 `cluster` concepts and decays low-support stale ones. Salient concepts are reported. |
| `self_modification` | `6 × interval` | `_loop_self_modification` | `SelfModificationEngine.run_cycle()` proposes up to 3 small tweaks, tests each against held-out recent transitions, accepts only those that reduce prediction error. Persists outcomes via `PersistentStore.record_self_modification(...)`. |
| `uncertainty` | `3 × interval` | `_loop_uncertainty` | per-action `causal_model.uncertainty_for(...)` scan, sorted. Stored in `runtime.last_uncertainty_scan` and exposed via `/uncertainty`. |

Each wake-up does work under a single `RLock` so the shared Darwin
agent never races with itself. Each wake-up emits a `RuntimeEvent`
with `event.loop` set, plus a structured `BackgroundLogEntry` to
`training_logs/background.jsonl`.

## The main thread

While the background threads run, a chat message from a client comes
in through `DarwinDaemon` and is handled by `DarwinRuntime.chat(...)`,
which:

1. parses the user text with the `SemanticParser`,
2. produces a response via the structured cognitive cycle (retrieve →
   plan → render → critique),
3. learns from the conversation as if it were any other transition,
4. emits a `thought` event to subscribed clients.

The chat handler also holds the `RLock`, so background loops briefly
wait while a chat turn is processed. The interval defaults are tuned
so this is invisible.

## Cadence design

You can change all five intervals via `loop_intervals`:

```python
runtime = DarwinRuntime(
    darwin=darwin,
    adapter=adapter,
    goal=goal,
    interval=2.0,
    loop_intervals={
        "experiment":        2.0,
        "simulation":        3.0,
        "dream":             8.0,
        "self_modification": 12.0,
        "uncertainty":       6.0,
    },
)
```

Or from the CLI:

```bash
darwin brain --interval 2.0
# experiment=2.0  simulation=3.0  dream=8.0  self_modification=12.0  uncertainty=6.0
```

The defaults form a 1× / 1.5× / 4× / 6× / 3× progression so the fast
loops dominate while the slow loops do heavier work less often.

## Persistence and resumption

`DarwinRuntime` checkpoints to `darwin_runtime_state.json` on
`stop()`. The snapshot contains:

- per-loop state (last event kind, last timestamp)
- `darwin._time` (logical clock)
- `darwin.exploration_rate`
- `darwin.causal_model.min_samples`
- `darwin._planner_overrides` (the self-modification engine's record
  of which planner tweaks have been accepted)

On construction with the same `state_path`, the runtime restores all
of this, so the brain wakes up in the posture it was last in. Combined
with `Darwin.from_store(...)` rehydrating transitions and semantic
frames from SQLite, the entire mind resumes coherently.

## What a brain-window session looks like

```
[experiment] Experiment confirmed: Will wait reliably produce battery_charge=3.0, …
[simulation] Mental simulation: step 1: open_curtains (conf 0.33, unc 0.67) -> step 2: …
[uncertainty] Uncertainty scan: open_curtains=0.67; toggle_switch=0.67; replace_fuse=0.67
[experiment] Experiment produced surprise in switch_on, room_bright: Will replace_fuse …
[dream] Dreaming. I have 7 grounded transitions. My strongest belief is if always: …
[self_modification] Self-modification proposed but rejected: causal.min_samples, …
[thought] parse: ... | retrieve: ... | plan: belief_answer ... | critic: passed
```

These are real, durable thoughts, not heartbeats. Each one corresponds
to actual model updates in memory and persisted records on disk.
