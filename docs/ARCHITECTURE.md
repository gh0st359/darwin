# Project Darwin Architecture

## Mission

Project Darwin is a causal-adaptive AI system. Its core claim is that
intelligence should be grounded in interventions, consequences, goals, and
self-updating models of the world, instead of word prediction alone.

Darwin's first version is built as a kernel: a compact set of components that
can be expanded into richer environments, robotics interfaces, long-horizon
agency, and human-facing interaction.

## Core commitments

1. Darwin learns from action, not text alone.
2. Darwin represents cause and effect as first-class structure.
3. Darwin predicts consequences before acting.
4. Darwin keeps memory of what happened, what changed, and what paid off.
5. Darwin forms concepts from stable patterns in state, action, effect, and
   reward.
6. Darwin uses uncertainty as fuel for curiosity, not as a reason to freeze.
7. Darwin can be embodied in a simulated world, software tools, or robots
   through the same perception-action-transition loop.

## System loop

```text
state_t
  -> Darwin perceives state
  -> planner asks causal model what actions are expected to change
  -> action is selected by reward, goal progress, cost, and uncertainty
  -> world applies the intervention
  -> state_t+1 and payoff are observed
  -> causal model, memory, and concepts update
```

The important unit is not a prompt or a completion. It is a transition:

```text
(before_state, action, after_state, reward)
```

That transition is the atom Darwin uses to learn reality.

## Components

### Darwin agent

`darwin.agent.Darwin` owns the runtime loop. It binds perception, planning,
memory, concepts, and causal learning into one adaptive system.

### Causal model

`darwin.causal.CausalModel` learns from interventions. For every action and
state variable, it estimates:

- whether the variable changes
- what value it tends to become
- whether the effect depends on a prior condition
- how confident Darwin should be in that belief
- how much reward the action tends to produce

The conditional learner lets Darwin represent facts like:

```text
if switch_on is false, toggle_switch causes switch_on to become true
if switch_on is true, toggle_switch causes switch_on to become false
```

This matters because many real actions are not one-way commands. Their effects
depend on the current world.

### Planner

`darwin.planner.CausalPlanner` scores each possible action by predicted
consequence. It does not select an action because text "sounds right"; it asks:

- What state do I expect after this action?
- Does that state move me toward the goal?
- What payoff has this action produced before?
- How costly is the action?
- How much would I learn by trying it?

### World model

`darwin.world_model.WorldModel` tracks the structured reality Darwin has seen:
variables, simple entities, action counts, prediction errors, hypotheses, and
possible hidden factors. This is where Darwin asks what it thinks will happen
before acting.

### Active experiments

`darwin.experiments.ExperimentEngine` turns uncertainty into proposed tests.
It asks which intervention would be most informative, predicts the result, and
marks the outcome as confirmed or surprising afterward.

### Memory

`darwin.memory.EpisodicMemory` stores raw transitions. This is Darwin's
experience stream.

`darwin.storage.PersistentStore` makes that stream durable in SQLite, including
transitions, concepts, thoughts, chat messages, experiments, and plans.

### Concepts

`darwin.concepts.ConceptIndex` turns experience into compact semantic units:

- state concepts: stable facts like `room_bright=true`
- effect concepts: interventions like `toggle_switch changes switch_on`
- outcome concepts: reward-bearing patterns

Concepts are hierarchical. Darwin now records primitive state concepts, action
effects, affordances, strategies, and meta-concepts such as reliable or risky
actions. This gives Darwin a grounded ladder from raw facts to higher structure.

### Self-model and metacognition

`darwin.self_model.SelfModel` tracks what Darwin knows, where its predictions
fail, which actions it is competent with, and what it should learn next.

### Long-horizon planning

`darwin.planner.CausalPlanner.plan_sequence` simulates action sequences through
the causal model. It scores delayed reward, goal satisfaction, and uncertainty.

### Always-on runtime

`darwin.runtime.DarwinRuntime` runs Darwin as a living local process. It can
respond to chat, think when idle, run active experiments, consolidate memory,
and expose its internal state through the CLI.

### Language cortex

`darwin.language.LanguageCortex` is Darwin's current speech layer. It is not an
LLM. It generates language from Darwin's own self-model, beliefs, concepts,
plans, experiment proposals, and recent runtime events. The language is still
small, but it is grounded in Darwin's state rather than detached text
prediction.

### Semantic language engine

`darwin.semantics.SemanticParser` turns arbitrary conversation into structured
meaning frames. Each frame records:

- speech act: question, goal, teaching, correction, hypothesis, directive, or
  statement
- grounded symbols: actions, variables, known concepts, and frontier concepts
- propositions: definitions, claims, and cause/effect hypotheses
- goals and values extracted from user language
- instructions and questions
- unknown terms Darwin should learn later
- confidence and uncertainty

`darwin.semantics.SemanticMemory` stores these frames separately from raw chat
text, so Darwin can remember what was meant instead of only what was typed.
Darwin also parses its own responses, giving it a first version of knowing what
it said.

### Cognitive response pipeline

`darwin.runtime.DarwinRuntime` now routes each chat turn through a response
cycle instead of directly selecting a canned sentence:

```text
user language
  -> SemanticParser creates a meaning frame
  -> ContextRetriever finds relevant semantic, causal, concept, and runtime memory
  -> ThoughtTrace records parse, retrieval, planning, and critique steps
  -> DiscoursePlanner decides the communicative intent and answer structure
  -> NaturalLanguageComposer realizes the plan as original text for this state
  -> ResponseCritic checks for leaks, ignored memory, and overconfident claims
  -> Darwin stores the trace, plan, critique, and conversation transition
```

The CLI exposes this process with `/thoughts`, `/reason`, `/retrieved`, and
`/critic`. Those commands can show compact internal notation. Darwin's spoken
answers are kept separate from that debug layer and are composed in natural
language from the current plan, retrieved memory, and uncertainty.

The speech layer still contains grammar and wording machinery because a
non-LLM system needs a surface realizer. It does not contain prompt-to-reply
tables. The generated answer depends on the parsed meaning, memory retrieved at
that moment, confidence, self-model state, and response critique.

### Worlds and embodiment

`darwin.embodiment` contains adapters for the current room simulation and
conversation. The same environment interface can later attach to:

- browser automation
- desktop tools
- file systems
- robotics middleware
- simulators
- real sensors and actuators

## What makes this not an LLM wrapper

The current kernel has no language model dependency. It does not call any model
API. It learns from structured experience and chooses actions through causal
prediction. A language interface can be added later, but language would be one
input/output layer around Darwin, not Darwin itself.

## Near-term roadmap

1. Expand causal structure beyond one-step effects into multi-step chains.
2. Add latent-state inference for hidden causes.
3. Add active experiment design so Darwin can choose tests that disambiguate
   competing causal hypotheses.
4. Add persistent storage for memory, concepts, and causal beliefs.
5. Add richer world adapters: local computer actions, browser tasks, and robot
   simulators.
6. Add self-modification boundaries: Darwin can propose model updates, run
   regression tests against them, and adopt only updates that improve measured
   world performance.
7. Expand the response pipeline with richer grammar, analogy, causal
   explanation, and multi-turn concept repair while keeping the causal kernel as
   the center of cognition.
