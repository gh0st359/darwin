# Project Darwin

Darwin is an experimental causal-adaptive AI system. It is not an LLM, not a
prompt chain, and not an API wrapper. The first kernel in this repository is
built around direct experience:

- perceive the current world state
- intervene with an action
- observe the resulting state transition
- update causal beliefs from the intervention
- choose future actions by predicted consequences, payoff, and uncertainty

The long-term vision is an AI that can self-adapt, self-learn, and self-evolve
through grounded cause and effect. This repository starts with the machinery
that makes that vision executable: causal transition learning, concept
formation, memory, and consequence-aware planning.

## What exists now

Darwin V0.2 includes:

- a causal model that learns action effects from intervention traces
- conditional effect learning, such as "toggle changes the switch differently
  depending on whether it is already on"
- persistent SQLite memory for transitions, concepts, chat, thoughts,
  experiments, and plans
- a world model that tracks variables, entities, hypotheses, prediction errors,
  and possible hidden factors
- an active experiment engine that chooses interventions to reduce uncertainty
- hierarchical concept formation: states, effects, affordances, strategies, and
  meta-concepts
- a long-horizon planner that simulates action sequences
- a self-model that tracks competence, uncertainty, failures, and learning
  priorities
- a symbolic semantic language engine that parses conversation into speech
  acts, goals, values, claims, hypotheses, corrections, instructions,
  questions, unknown terms, and grounded internal symbols
- embodiment adapters, currently for the room simulation and conversation
- an always-on runtime loop that can think, experiment, dream, and chat
- a small deterministic world for testing causal adaptation

This is the seed of the system, not the ceiling.

## Run it

```powershell
python -m pip install -e .
python -m unittest discover -s tests
python -m darwin.cli run --steps 40 --seed 7
python -m darwin.cli live
```

If you prefer not to install the package, set `PYTHONPATH` for the current
shell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m darwin.cli run --steps 40 --seed 7
```

## Live CLI

```powershell
python -m darwin.cli live --memory darwin_memory.sqlite3
```

Commands:

```text
/status       show Darwin's self-model
/beliefs      show strongest causal beliefs
/concepts     show concept hierarchy
/semantics    show recent parsed meanings
/experiments  show active experiment proposals
/think        run one cognition cycle now
/dream        consolidate memory and concepts
/run N        run N cognition cycles
/plan         show the current multi-step plan
/trace        show recent runtime events
/stream on|off show or hide live background thoughts
/exit         shut down cleanly
```

Normal text is treated as conversation experience. Darwin records it, extracts
simple signals from it, and folds it into memory and concepts without using an
LLM.

When background cognition is enabled, Darwin streams live cognition events such
as experiments and reflections into the terminal. Ask "what are you thinking?"
to get a natural-language summary of the recent thought thread.
Ask "what did you understand?" or run `/semantics` to inspect Darwin's parsed
meaning frames.

## Repository map

```text
docs/
  ARCHITECTURE.md      System architecture and roadmap
src/darwin/
  agent.py             Darwin orchestration loop
  causal.py            Causal transition learner
  cli.py               Command line entrypoint
  concepts.py          Concept formation from experience
  embodiment.py        Simulation and conversation adapters
  experiments.py       Active experiment proposal/evaluation
  language.py          State-grounded natural-language cortex
  memory.py            Episodic and semantic memory
  planner.py           Consequence-aware action ranking
  runtime.py           Always-on cognition loop
  self_model.py        Metacognition and learning priorities
  semantics.py         Symbolic/conceptual language parser and semantic memory
  storage.py           SQLite durable memory
  types.py             Shared data structures
  world_model.py       Structured world model and hypotheses
  worlds.py            Test environments
tests/
  test_agent.py
  test_causal.py
  test_semantics.py
  test_v02.py
```
