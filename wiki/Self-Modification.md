# Self-Modification

`SelfModificationEngine` (`darwin/self_modification.py`) is the part
of Darwin that rewrites its own parameters when doing so reduces
prediction error on held-out experience. It is one of the five
background loops and runs on the longest cadence.

## The contract

A proposal is **only** accepted if it measurably lowers Darwin's
average prediction error on the most recent held-out transitions.
Anything else is reverted. Nothing about this contract requires (or
allows) an external model.

## What can be modified

The engine generates `ProposedModification`s in four families:

| Kind | Target | Direction | Why |
| --- | --- | --- | --- |
| `causal.min_samples` | `causal_model.min_samples` | ±1 | makes the model demand more or less evidence per belief |
| `exploration.rate` | `darwin.exploration_rate` | ±0.05 | trades exploration vs exploitation |
| `concept.prune` | `memory.concepts` | drop up to 5 low-support, low-level concepts | removes likely noise from the hierarchy |
| `planner.exploration_bias` | `darwin._planner_overrides["exploration_bias"]` | ±0.2 | scales the planner's curiosity weighting |

Each proposal carries:
- `apply(darwin)` and `revert(darwin)` closures
- `rationale` (human-readable)
- `payload` (the before/after values)
- `proposal_id`

## Evaluation

```python
def evaluate(proposal):
    sample = darwin.memory.episodes.recent(holdout_size)  # default 12
    baseline_error = _prediction_error(darwin, sample)
    proposal.apply(darwin)
    candidate_error = _prediction_error(darwin, sample)
    if candidate_error < baseline_error:
        # accept; persist outcome
    else:
        proposal.revert(darwin)
```

`_prediction_error` averages per-variable absolute deltas (for
numerics) or 0/1 mismatches (for non-numerics) of
`causal_model.predict(before, action).state` vs the actual `after`,
across all variables of the sample.

A snapshot of the affected fields is taken before applying so we can
restore on exception too, not only on rejection.

`run_cycle()` returns up to 3 `ModificationOutcome`s. Each is stored
via `PersistentStore.record_self_modification(...)` so there is a
durable audit trail.

## Cross-feed from simulation

The simulation loop registers the highest-uncertainty step of its
imagined chain as a prediction-failure entry in
`SelfModel.prediction_failures`. That changes Darwin's
`learning_priority`, which (a) reshapes the experiment loop's
preferences and (b) is surfaced in `/status` and in
`ResponsePlan.self_reflection`. So a mental simulation that exposes a
weakness can sharpen real learning even without acting.

## How to watch it in action

```bash
darwin connect
you> /loops
- self_modification  interval=18.0s last=self_modification
you> /selfmod
- [rejected] causal.min_samples baseline=0.0042 candidate=0.0050 gain=-0.0008
    rationale: lower min_samples for faster belief crystallization
- [rejected] exploration.rate baseline=0.0042 candidate=0.0048 gain=-0.0006
    rationale: less exploration based on current uncertainty
- [accepted] planner.exploration_bias baseline=0.0042 candidate=0.0029 gain=0.0013
    rationale: increase curiosity bias
```

Accepted changes persist across restarts via
`darwin_runtime_state.json` (which carries `exploration_rate`,
`min_samples`, and `planner_overrides`).

## What this is not

- It is not "the model edits its own weights" — there are no neural
  weights to edit. The mind is symbolic.
- It is not "the model reasons about itself in natural language" —
  there is no LLM in this loop.
- It is not "neural architecture search" — the search space is small,
  the evaluation is concrete (held-out prediction error), and every
  decision is logged.

It is a tight, durable, empirically-validated tuning loop. Boring on
purpose. Boring is how it stays trustworthy.
