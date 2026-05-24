# Self-Model and Metacognition

`SelfModel` (`darwin/self_model.py`) is Darwin's running self-estimate
of "what I know, what I don't, what I should try to learn next." It
is built from the same intervention stream that feeds the causal
model, but it tracks meta-quantities: per-action competence,
prediction failures, weakest areas, and a `learning_priority` string
that drives the rest of the system.

## What it tracks

Per action it keeps `Competence`:

```python
Competence(action, samples, reward_mean, surprise_mean)
score = familiarity * (1 + reward_mean) / (1 + surprise_mean)
```

Across actions it keeps:
- `known_variables` (counter)
- `prediction_failures` keyed by `f"{action}:{variable}"`
- a log of reflection strings

## `learning_priority`

Computed each time `self_report()` is called. The priority is the
first rule below that applies:

1. **Active prediction failures**: a `(action, variable)` key still
   accumulating mismatches and not yet "resolved" (a resolved key
   has a high-confidence belief with ≥2 samples).
   - If `action_count(action) < min_samples`: `"retest <action> to
     stabilize its effect on <variable>"`
   - Else: `"find hidden conditions for <action>:<variable>"`
2. **Too few observations**: `"collect more interventions"`
3. **Hidden factor hypothesis**: `"test hidden factor hypothesis
   <action>:<variable>"`
4. **Weakest action by competence score**: `"improve competence with
   <action>"`
5. Default: `"expand the environment with new actions and variables"`

This is what drives:

- `ResponsePlan.next_actions[0]` (the "next learning pressure" line
  that the composer surfaces)
- the experiment loop's preferences (low-competence actions get more
  attention)
- the `_loop_simulation` cross-feed (it injects synthetic prediction
  failure keys so the priority reflects imagined weaknesses too)

## `reflect()`

```python
reflection = self_model.reflect(memory, causal_model, world_model)
# "I have N grounded transitions. My strongest belief is …. My weakest area is …. My next learning priority is …."
```

The reflection string is what the dream loop emits, what `/think`
returns, and what gets persisted via `PersistentStore.record_thought`
under kind `"reflection"`. Every reflection is durable.

## `report()`

Returns a `SelfReport`:

```python
SelfReport(
    observations,
    known_actions,
    known_variables,
    strongest_belief,
    weakest_area,
    learning_priority,
    competence,    # list[Competence]
)
```

`/status` prints it directly. `ResponsePlan.self_reflection` carries
a condensed version that the DLM is allowed to ground its tone in
("I am tentative; my learning priority is X").

## Metacognition without an LLM

The point is that Darwin's report on itself is computed from the
*same* statistics that drive its behaviour — not narrated by a
separate model that could lie. When `/status` says
"`strongest_belief = if always: open_curtains -> room_bright False ->
True conf=0.86`", that string is a serialization of the actual
strongest entry in `causal_model.beliefs(limit=1)`, with the
confidence and condition pulled from the same struct the planner
consults to choose actions. There is no parallel narrator. Darwin's
self-knowledge and Darwin's behavior come from one ground truth.
