# Project Darwin

Darwin is an experimental causal-adaptive AI system. Not an LLM. Not a prompt
chain. Not an API wrapper. Darwin is built around direct experience, learned
cause and effect, and a 24/7 symbolic mind that can hold a conversation
without ever delegating its reasoning to a pre-trained model.

The current release is **v2: Eternal Causal Mind**.

## What v2 is

A continuously-thinking causal intelligence with five concurrent background
cognition loops, a daemon/client split so the mind runs 24/7 in one
terminal while you chat from another, a self-modification engine that
rewrites its own knobs when doing so reduces prediction error, and an
optional thin language renderer (the **Darwin Language Module / DLM**)
that converts Darwin's structured plans into natural English without ever
being allowed to reason on its own.

### Non-negotiables

- Darwin's mind is the symbolic + causal engine, always on.
- The DLM is a thin renderer only. It receives a structured `ResponsePlan`
  from Darwin and renders it as prose. It does zero reasoning, holds zero
  external world knowledge, and never gets the last word over the
  `FaithfulnessValidator`.
- No external LLM ever generates Darwin's thinking, concepts, or causal
  rules.
- Every output passes a faithfulness validator + a response critic.
  Renderings that drift are rejected; the deterministic composer takes
  over.

## Architecture in one diagram

```
                   ┌───────────────────────────────────────────┐
                   │             darwin brain (24/7)           │
                   │                                           │
                   │  CausalModel ── CausalChain ── Planner    │
                   │       │              │             │      │
                   │       ▼              ▼             ▼      │
                   │   Episodic ── Concepts ── Self-modify     │
                   │   Memory       │            engine        │
                   │       │        ▼            │             │
                   │       └─►  WorldModel  ◄────┘             │
                   │                │                          │
                   │           SelfModel                       │
                   │                │                          │
                   │   ┌────────────┴───────────┐              │
                   │   │ 5 background loops:    │              │
                   │   │  experiment            │              │
                   │   │  simulation            │              │
                   │   │  dream (consolidation) │              │
                   │   │  self_modification     │              │
                   │   │  uncertainty           │              │
                   │   └────────────────────────┘              │
                   │                │                          │
                   │           DiscoursePlanner                │
                   │                │                          │
                   │           ResponsePlan ─────► DLM         │
                   │                │              (stub or   │
                   │           FaithfulnessValidator + Critic  │
                   │                │              gemma-3-270m)
                   └────────────────┼──────────────────────────┘
                                    │ TCP JSON-line protocol
                                    │
                  ┌─────────────────┴──────────────────┐
                  ▼                                    ▼
         darwin connect                       darwin connect
         (clean chat 1)                       (clean chat 2)
```

## Quick start

```bash
# Install
pip install -e .

# Run the test suite
python -m unittest discover tests   # 44 tests in <6s

# Two-terminal workflow
darwin brain          # terminal 1: 24/7 mind + thought stream
darwin connect        # terminal 2: clean chat
```

In the chat window you see only `you>` for your input and `darwin>` for
Darwin's response. Background thoughts stream in the brain terminal. Open
as many `darwin connect` windows as you want — they all share one mind
and one persistent memory.

Single-terminal mode is still available:

```bash
darwin live
```

## What's in the box

| Phase | What | File |
| --- | --- | --- |
| 0 | structured `ResponsePlan` + DLM payload, JSONL plan/background/metrics logs | `instrumentation.py`, `discourse.py`, `thought.py` |
| 1 | multi-step causal chains, causal graph, uncertainty propagation | `causal_chain.py`, `planner.py` |
| 1 | per-action / per-variable episodic indices, cross-source retrieval | `memory.py`, `retrieval.py` |
| 1 | self-modification engine (proposes + tests + accepts) | `self_modification.py` |
| 2 | 5-loop multi-threaded 24/7 mind, persistent runtime state | `runtime.py` |
| 2 | brain daemon + chat client (TCP, JSON-line) | `server.py` |
| 3 | DLM Protocol + StubDLM + GemmaDLM + FaithfulnessValidator | `dlm.py` |
| 4 | training-pair collector + JSONL export | `training_data.py` |
| 5 | clean two-terminal UX, full CLI, persistence | `cli.py`, `storage.py` |

## CLI reference (cheat sheet)

```text
darwin run --steps 40 --seed 7              # batch run in the room sim
darwin live                                 # single-terminal mind + chat
darwin brain                                # 24/7 daemon (terminal 1)
darwin connect                              # clean chat client (terminal 2)
darwin connect --watch-events               # connect AND mirror brain events
darwin export-training --min-quality 0.7    # export DLM training pairs
```

Chat commands (in `darwin connect` or `darwin live`):

```text
/status         self-model
/beliefs        strongest causal beliefs
/concepts       concept hierarchy
/experiments    active experiment proposals
/think          run one cognition cycle now
/dream          consolidate memory now
/simulate       run one mental simulation now
/selfmod        propose+test self-modifications now
/uncertainty    per-action uncertainty scan
/loops          background loop status
/causal-graph   distilled action -> variable graph
/dlm            DLM info and last render validation
/training       DLM training-data corpus summary
/metrics        structured-logger metrics
/thoughts       last internal thought trace
/retrieved      memories used for last response
/critic         self-critique of last response
/trace          recent runtime events
/exit           disconnect (brain keeps running)
/shutdown-brain stop the brain daemon
```

## Persistence

- `darwin_memory.sqlite3` — transitions, concepts, thoughts, chat,
  experiments, plans, semantic frames, self-modification proposals.
- `darwin_runtime_state.json` — runtime-loop posture (time, exploration
  rate, min_samples, planner overrides).
- `training_logs/*.jsonl` — plan logs, background-cognition logs,
  metrics, DLM training pairs.

Kill the brain and restart — Darwin wakes up with the same internal
posture.

## Optional: gemma-3-270m as the DLM mouth

```bash
ollama pull gemma3:270m
darwin brain --dlm gemma --dlm-backend ollama
```

If the renderer's output fails `FaithfulnessValidator`, the runtime
silently falls back to the deterministic composer. The DLM is therefore
strictly optional — Darwin's mouth is never required to produce its mind.

## Repository map

```text
docs/
  ARCHITECTURE.md      System architecture (v0.3 era)
  V2_ARCHITECTURE.md   v2 architecture deep-dive
wiki/                  In-depth Darwin documentation (mirror of GitHub Wiki)
src/darwin/
  agent.py             Darwin orchestration
  causal.py            Causal transition learner
  causal_chain.py      Multi-step causal chains and graph
  cli.py               Command-line entrypoint
  composer.py          Deterministic natural-language realizer
  concepts.py          Concept formation + consolidation
  critic.py            Response critique
  discourse.py         Discourse planning -> ResponsePlan
  dlm.py               Darwin Language Module (StubDLM, GemmaDLM, validator)
  embodiment.py        Adapters for simulated / conversational embodiment
  experiments.py       Active experiment proposal / evaluation
  instrumentation.py   Structured plan + background + metrics logging
  language.py          State-grounded language cortex (legacy v0.3 surface)
  memory.py            Indexed episodic + semantic memory
  planner.py           Consequence-aware planner + chain integration
  retrieval.py         Cross-source memory retrieval
  runtime.py           5-loop 24/7 cognition loop
  self_model.py        Metacognition and learning priorities
  self_modification.py Self-modification engine
  semantics.py         Symbolic language parser + semantic memory
  server.py            DarwinDaemon + DarwinClient (TCP JSON-line)
  storage.py           SQLite durable memory
  streaming.py         Incremental text output
  thought.py           Inspectable thought traces
  training_data.py     DLM (plan -> rendering) pair collection
  types.py             Shared data structures
  world_model.py       Structured world model + hypotheses
  worlds.py            Test environments (adaptive room)
tests/
  test_agent.py
  test_brain_daemon.py
  test_causal.py
  test_language_cognition.py
  test_semantics.py
  test_v02.py
  test_v2.py
```

## In-depth docs

The **`wiki/`** directory is a complete mirror of the GitHub Wiki. Start
with `wiki/Home.md`. Each page corresponds 1:1 to a GitHub Wiki page so
you can copy them straight in.
