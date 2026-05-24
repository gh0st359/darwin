# Planning and Simulation

Darwin has two modes of imagining the future:

1. **Planning** — pick a sequence of actions that best advances a
   `Goal`. Produces a `MultiStepPlan`.
2. **Mental simulation** — explore "what if I did this?" without any
   particular goal, just to learn where the model is most uncertain.

Both rely on the `CausalChainEngine`.

## Goals

A `Goal(desired, weights, reward_weight, progress_weight,
exploration_weight)` describes a *target state*. Variables in
`desired` are scored by match (1.0 if equal for categorical; a
distance-based score for numeric). `weights` lets you prioritize
variables. `progress_weight` controls how much the planner cares
about *moving* toward the desired state; `reward_weight` controls
how much it cares about extrinsic reward; `exploration_weight`
controls how much it values reducing uncertainty.

## `CausalPlanner.rank(state, actions, goal)`

Returns `PlanCandidate`s sorted by

```
score = reward_weight * expected_reward
      + progress_weight * (predicted_goal_score - current_goal_score)
      + exploration_weight * uncertainty_for(state, action)
      - action.cost
```

The chosen action is the highest-scoring candidate. The agent's
`decide(state, goal)` adds an ε-greedy exploration: with probability
`exploration_rate`, sample from the top-uncertainty candidates instead
of always taking the top score.

## `CausalPlanner.plan_sequence(state, actions, goal, horizon, beam_width)`

Beam search across horizons. At each depth, each surviving beam is
expanded by every available action. Each expansion is scored by
`progress_weight * goal_score + reward_weight * total_reward +
exploration_weight * avg_uncertainty`. Beams are pruned to
`beam_width`.

After the search, the best plan is paired with a
`CausalChainEngine.simulate_chain(...)` rollout so the plan reports
a true `chain_confidence` (multiplicative) and the maximum of its
average uncertainty and the chain uncertainty.

The result is a `MultiStepPlan` carrying:
- `actions: list[Action]`
- `final_state: State`
- `total_expected_reward`, `goal_score`
- `uncertainty`, `chain_confidence`
- `trace: list[str]` (per-step human-readable trace)
- `causal_chain: CausalChain`

`PersistentStore.record_plan(...)` stores every plan that
`Darwin.plan(...)` produces.

## Mental simulation (the background loop)

```python
chains = darwin.planner.reason_chain(state, actions, depth=3, beam=4)
best = chains[0]
```

The simulation loop calls this every `1.5 × interval` seconds. The
best chain by `chain_confidence × (1 + total_expected_reward)` is
stored in `runtime.last_simulation` and emitted as a `simulation`
event.

The simulation is **not** treated as evidence — it is imagined, not
experienced — but the highest-uncertainty step is registered in
`SelfModel.prediction_failures` so `learning_priority` reflects what
the mind is unsure about. Imagination shapes attention; it does not
contaminate the causal model.

## Why uncertainty propagation matters

Without it, a planner can produce a four-step plan where every step
is 25% likely to behave as predicted and confidently report it as
"the best plan." With it:

```
chain_confidence = 0.25 ** 4 ≈ 0.004
chain_uncertainty ≈ 0.996
```

That number flows into the plan and into `ResponsePlan.uncertainty_levels`,
into the composer's wording ("I am uncertain about this plan at level
1.00"), and into the critic's enforcement that high uncertainty is
disclosed. The behavior of "confidently saying low-confidence things"
becomes a structurally enforced impossibility.

## Where you see this

```bash
darwin connect
you> /simulate
chain confidence=0.06
chain uncertainty=0.94
total expected reward=1.66
- step 1: open_curtains conf=0.33
- step 2: open_curtains conf=0.33
- step 3: close_curtains conf=0.56

you> /plan        # (live mode only)
open_curtains -> wait -> toggle_switch: score=2.81, reward=0.21,
                                        goal=0.95, uncertainty=0.42,
                                        chain_confidence=0.21
```
