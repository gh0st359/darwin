# Causal Reasoning

Darwin's beliefs are grounded in interventions. The `CausalModel`
learns what actions do; the `CausalChainEngine` rolls those beliefs
forward across sequences with propagated uncertainty; the
`CausalPlanner` ranks candidate plans by their predicted consequence.

## CausalModel (`darwin/causal.py`)

Every observed `Transition(before, action, after, reward, t,
metadata)` updates two families of statistics:

1. **Effect stats** per `(action, variable)`:
   - count of times the variable changed when the action was taken
   - per-outcome frequency counter (`after_values`)
   - per-(before,after) transition counter
   - mean and variance of the numeric delta (online via Welford)
   - confidence = sample_confidence × outcome_consistency
2. **Conditional effect stats** per `(action, variable, feature, value)`:
   - the same stats, restricted to transitions where `state[feature] == value`
   - used to spot conditional rules (e.g. "toggle_switch *only* changes
     `switch_on` if `fuse_intact == True`")

Predictions:

```python
prediction = causal_model.predict(state, "open_curtains")
# Prediction.state: predicted next state
# Prediction.estimates: per-variable EffectEstimate with confidence & reason
# Prediction.confidence: average of estimate confidences
# Prediction.uncertainty: 1 - confidence
```

The model picks the *most specific* conditional that beats the global
estimate (with a specificity bonus), and falls back to the global one
when no condition wins. This is what makes the room-world example
work: the model learns that flipping the switch is conditional on the
fuse being intact, instead of producing a noisy unconditional rule.

Beliefs:

```python
for belief in causal_model.beliefs(limit=10):
    # belief: action, variable, effect, condition, confidence, samples
```

Beliefs are sorted by confidence × samples. They are what flows into
`ResponsePlan.causal_claims` for the DLM and the critic.

## Chains (`darwin/causal_chain.py`)

`CausalChainEngine.simulate_chain(state, [action, action, ...])`
rolls the model forward step by step. Each step produces a
`ChainNode` with `state_before`, `state_after`, `confidence`,
`uncertainty`, `expected_reward`, and a list of human-readable
rationale lines.

Uncertainty propagation is multiplicative on confidence and additive
on uncertainty:

```
chain_confidence  *= max(0.05, step_confidence)
chain_uncertainty  = 1 - (1 - prev_uncertainty) * (1 - step_uncertainty)
```

This naturally penalizes long, unsupported chains: each speculative
step pushes the chain's overall confidence toward zero and its
uncertainty toward one.

`explore_chains(state, actions, depth, beam)` runs beam search over
action sequences. Each beam is itself a fully-simulated chain.

`chain_for_goal(state, actions, goal_variables, depth, beam)` returns
the chain that most increases the number of goal variables changed
away from their current values, weighted by chain confidence and
penalized by chain uncertainty.

## CausalGraph

`CausalChainEngine.graph(min_confidence)` distills the learned
beliefs into an action→variable graph with edges carrying confidence
and sample counts. Useful for understanding what Darwin has learned at
a glance:

```bash
darwin connect
you> /causal-graph
actions=7 variables=6 edges=12
- open_curtains -> curtains_open effect=False -> True conf=0.86 n=4
- open_curtains -> room_bright   effect=False -> True conf=0.86 n=4
- close_curtains -> curtains_open effect=True -> False conf=0.86 n=4
- toggle_switch -> switch_on     effect=False -> True conf=0.66 n=3
- toggle_switch -> battery_charge effect=+= -1 conf=0.66 n=3
- replace_fuse -> fuse_intact     effect=False -> True conf=0.6 n=3
- overload_circuit -> fuse_intact effect=True -> False conf=0.6 n=3
...
```

## Planner (`darwin/planner.py`)

`CausalPlanner.rank(state, actions, goal)` returns `PlanCandidate`s
sorted by score, where score is:

```
score = goal.reward_weight   * expected_reward
      + goal.progress_weight * (predicted_goal_score - current_goal_score)
      + goal.exploration_weight * uncertainty_for(state, action)
      - action.cost
```

`plan_sequence(...)` does beam search across horizons and attaches a
`CausalChain` to the chosen `MultiStepPlan`. The chain's confidence
flows into the plan's reported `chain_confidence`, and the planner
records it via `PersistentStore.record_plan(...)`.

## What this gives you

When Darwin says "if I open the curtains, then I expect the room to be
bright with confidence 0.86", that statement is:

- a real `(action, variable, effect, confidence, samples, condition)`
  tuple from the model,
- preserved in `ResponsePlan.causal_claims`,
- enforced into the rendered text by `FaithfulnessValidator` and
  `ResponseCritic`,
- inspectable at any time via `/beliefs` and `/causal-graph`.

If Darwin doesn't have evidence, it says so. Confidence and sample
counts are first-class citizens, not afterthoughts.
