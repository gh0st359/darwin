# Glossary

Definitions of every important term in v2, in alphabetical order.

### Action
A frozen dataclass: `Action(name, cost, description, metadata)`. The
unit of intervention. Actions are what Darwin does; everything Darwin
learns is keyed by which action did it.

### Adapter
See **EnvironmentAdapter**.

### Background loop
One of five concurrent threads in `DarwinRuntime` that keep Darwin
thinking 24/7: `experiment`, `simulation`, `dream`,
`self_modification`, `uncertainty`. See [The 24/7 Brain](The-24-7-Brain.md).

### Brain (`darwin brain`)
The daemon process that owns Darwin, runs the background loops, and
serves chat clients over TCP.

### `CausalChain`
A simulated sequence of action consequences with propagated
confidence and uncertainty. See [Causal Reasoning](Causal-Reasoning.md).

### `CausalClaim`
A structured record on a `ResponsePlan`: `(action, variable, effect,
condition, confidence, samples)`. The DLM is required to preserve
these in meaning; the validator enforces it.

### `CausalGraph`
The action → variable graph distilled from the causal model's
learned beliefs. Visible via `/causal-graph`.

### `CausalModel`
The core learner. Updates per-`(action, variable)` and per-`(action,
variable, feature, value)` effect statistics from every observed
transition.

### Chat client (`darwin connect`)
A thin TCP client that gives you a clean `you>` / `darwin>` REPL.
Does not subscribe to the brain's event firehose by default. See
[Two-Terminal Workflow](Two-Terminal-Workflow.md).

### Competence
Per-action self-estimate inside `SelfModel`. Familiarity times reward
mean divided by surprise mean.

### Composer
The deterministic `NaturalLanguageComposer`. Produces fluent English
from a `ResponsePlan` using fixed sentence templates and the plan's
structured fields. Faithful by construction; always passes
validation.

### Concept
A learned hierarchical structure: state, effect, affordance,
strategy, meta, cluster. Salience is `support * (1 + |reward|) * (1
+ 0.15 * level)`.

### `ConceptIndex`
The container for concepts. `consolidate()` is what runs during the
dream loop.

### Daemon
See **Brain**.

### DLM (Darwin Language Module)
The thin renderer: `StubDLM` (composer) or `GemmaDLM` (gemma-3-270m).
The only place a neural model is allowed in the stack, and only for
rendering already-formed structured plans. See
[The Darwin Language Module](The-Darwin-Language-Module.md).

### `DiscoursePlanner`
Picks a mode (`clarify`, `answer`, `belief_answer`, `learn`,
`experiment`, `self_report`, `memory_summary`, `unknown_terms`,
`conversation`) and produces a `ResponsePlan`.

### Dream loop
The background loop that runs `Darwin.reflect()` +
`ConceptIndex.consolidate()`. Forms cluster concepts; decays stale
ones; emits a reflection.

### Embodiment
Anything that satisfies the `EnvironmentAdapter` protocol.

### EnvironmentAdapter
`Protocol` with `observe`, `possible_actions`, `apply`. See
[Embodiment](Embodiment.md).

### Episodic memory
The bounded deque of all observed `Transition`s, with indices by
action and by variable.

### ExperimentEngine
Proposes uncertainty-reducing interventions. The experiment loop runs
one whenever there is a sufficiently uncertain candidate.

### `FaithfulnessValidator`
The audit layer for every DLM output. Rejects renderings that drop
high-confidence claims, fail to surface high-impact uncertainties,
leak parser notation, etc. See
[Faithfulness Validation](Faithfulness-Validation.md).

### Goal
A target state plus weights: `Goal(desired, weights, reward_weight,
progress_weight, exploration_weight)`.

### `learning_priority`
A short string computed by `SelfModel` that tells the rest of the
system what Darwin should focus on next. Flows into
`ResponsePlan.next_actions[0]` and into the experiment loop's
preferences.

### `MultiStepPlan`
The output of `CausalPlanner.plan_sequence(...)`. Carries actions,
predicted final state, score, total expected reward, uncertainty,
chain confidence, and the simulated `CausalChain`.

### `ResponsePlan`
The strictly-shaped data structure that every renderer consumes. See
[Discourse Planning](Discourse-Planning.md).

### `RuntimeEvent`
A single observable thought or background-loop wake-up:
`RuntimeEvent(kind, content, payload, loop, timestamp)`.

### `SelfModel`
Darwin's metacognition. Tracks competence, prediction failures,
learning priority.

### `SelfModificationEngine`
The background-loop component that proposes small tweaks to Darwin's
own knobs, tests them on held-out experience, and accepts only those
that reduce prediction error.

### `SemanticFrame`
A structured parse of a user message. Speech act, topic, intent,
confidence, groundings, propositions, goals, values, unknown terms.

### `SemanticMemory`
The container for `SemanticFrame`s, plus counters over propositions,
goals, values, and unknown terms.

### Simulation loop
The background loop that runs `CausalChainEngine.explore_chains(...)`
to imagine action sequences. Feeds the highest-uncertainty step back
to `SelfModel.prediction_failures`.

### State
A `dict[str, Any]`. Whatever the environment adapter says it is.

### `StructuredLogger`
The JSONL logger for plans, background events, and metrics.

### Subscribe / Unsubscribe
Wire commands `{"cmd": "subscribe"}` and `{"cmd": "unsubscribe"}`.
Control whether a connected client receives the background event
firehose. The default chat client does *not* subscribe.

### `ThoughtTrace`
The live inspectable trace of one cognitive cycle. Steps: `parse`,
`retrieve`, `plan`, `dlm`, `dlm_fallback`, `critic`.

### `Transition`
The atomic learning unit: `Transition(before, action, after, reward,
t, metadata)`.

### Uncertainty loop
The background loop that scans `causal_model.uncertainty_for(...)`
per action.

### `WorldModel`
A structured running model of variables, entities, hypotheses,
prediction errors, and possible hidden factors.
