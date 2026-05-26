# Darwin v4: Generative Universe Kernel

Darwin v4 adds a corpus-to-world path around the existing symbolic/causal
kernel. It is not an LLM, not a prompt chain, and not an API wrapper. The
kernel still reasons through structured transitions, causal hypotheses, memory,
world state, and response plans.

The new piece is the substrate: Darwin can ingest curated offline knowledge,
turn explicit causal claims into sandboxed world specifications, act in those
worlds, and promote support only after generated experience backs a claim.

## What exists in this branch

| Area | Current implementation |
| --- | --- |
| Corpus ingestion | `darwin ingest-corpus --source wikipedia|wikidata|wikidump --path PATH --memory PATH` |
| v4 brain mode | `darwin brain --kernel v4 --workers auto --accelerator auto` |
| Chat client | `darwin connect` is unchanged |
| Knowledge model | `src/darwin/knowledge.py` |
| Generated worlds | `src/darwin/generative.py` |
| v4 scheduler surface | `src/darwin/kernel.py` |
| Dormant live research | `src/darwin/research.py` |
| Runtime integration | `src/darwin/cli.py`, `src/darwin/runtime.py`, `src/darwin/server.py`, `src/darwin/storage.py`, `src/darwin/discourse.py` |
| Tests | `tests/test_v4_generative_universe.py` |

This is a real foundation, not finished universal sentience. The implementation
has a generated-world path and promotion boundary; it does not yet have a fully
complete actor-runtime replacement, high-scale dump ingestion, live web
research, or broad autonomous world-building.

## High-level architecture

```mermaid
flowchart LR
    Corpus["curated corpus<br/>Wikipedia text, Wikidata JSONL, local wikidump text"]
    Ingestor["CorpusIngestor<br/>deterministic extractors"]
    Atom["KnowledgeAtom<br/>definition / relation / quantity / alias / causal_hypothesis"]
    Graph["KnowledgeGraph<br/>persisted query surface"]
    Generator["WorldSpecGenerator<br/>causal hypotheses -> WorldSpec"]
    Compiler["SandboxedWorldCompiler<br/>validate data-only worlds"]
    Adapter["GenerativeUniverseAdapter<br/>EnvironmentAdapter-compatible"]
    Runtime["DarwinRuntime<br/>experiment / simulation / dream / self_modification / uncertainty"]
    Plan["ResponsePlan<br/>closed plan payload"]
    DLM["DLM / Gemma<br/>prose renderer only"]
    User["user"]

    Corpus --> Ingestor --> Atom --> Graph --> Generator --> Compiler --> Adapter --> Runtime --> Plan --> DLM --> User
```

The DLM sits after the plan. It is the mouth, not the source of intelligence.
The reasoning path is:

```mermaid
flowchart TB
    State["observed state"]
    Action["intervention"]
    Transition["Transition(before, action, after, reward)"]
    Causal["CausalModel"]
    World["WorldModel"]
    Memory["Episodic + semantic memory"]
    Discourse["DiscoursePlanner"]
    Plan["ResponsePlan"]

    State --> Action --> Transition
    Transition --> Causal
    Transition --> World
    Transition --> Memory
    Causal --> Discourse
    World --> Discourse
    Memory --> Discourse
    Discourse --> Plan
```

## Runtime comparison: v3 vs v4

```mermaid
flowchart TB
    subgraph V3["v3 UniverseSimulation"]
        V3World["hand-built adapters<br/>room / math / space / time"]
        V3Actions["fixed action set"]
        V3State["fixed state variables"]
    end

    subgraph V4["v4 GenerativeUniverse"]
        V4Atoms["KnowledgeAtom records"]
        V4Specs["WorldSpec records"]
        V4Compiler["SandboxedWorldCompiler"]
        V4Adapter["GenerativeUniverseAdapter"]
    end

    Kernel["same Darwin symbolic/causal kernel"]

    V3World --> V3Actions --> V3State --> Kernel
    V4Atoms --> V4Specs --> V4Compiler --> V4Adapter --> Kernel
```

v3 remains useful as the legacy/default hand-built universe path. v4 swaps the
environment substrate: generated sandbox worlds expose actions through the same
adapter protocol Darwin already uses.

## Belief promotion boundary

Corpus claims do not become causal beliefs automatically. They can become
hypotheses. A generated experiment can then promote provenance after Darwin acts
in the sandbox world and observes a transition.

```mermaid
flowchart LR
    Claim["corpus claim<br/>Force causes acceleration"]
    Atom["KnowledgeAtom<br/>kind=causal_hypothesis<br/>promoted=false"]
    Spec["WorldSpec<br/>generated/force_acceleration"]
    Action["generated/apply_force"]
    Transition["observed transition<br/>force.acceleration increases"]
    Promotion["promoted=true<br/>support_kind=generated_experiment"]
    Belief["CausalModel belief<br/>learned from transition"]

    Claim --> Atom --> Spec --> Action --> Transition
    Transition --> Promotion
    Transition --> Belief
    Claim -. "not automatically trusted" .-> Atom
```

The promotion implementation is in `DarwinRuntime._loop_experiment()`:

1. The adapter supplies generated action metadata with `provenance_ids`.
2. Darwin applies the generated action and learns from the transition.
3. If provenance IDs are present, `PersistentStore.promote_knowledge_atoms(...)`
   marks those atoms as `promoted` with `support_kind="generated_experiment"`.

## Data model

`PersistentStore` creates the v4 tables alongside the older runtime tables. Some
tables are active in this branch; others are reserved schema surface for the v4
pipeline as it grows.

```mermaid
erDiagram
    knowledge_atoms ||--o{ world_specs : "provenance_ids"
    world_specs ||--o{ generated_experiments : "world_name"
    world_specs ||--o{ validation_results : "target"
    knowledge_atoms ||--o{ generated_experiments : "provenance_ids"
    research_events ||--o{ knowledge_atoms : "future curated claims"

    knowledge_atoms {
        integer id
        string atom_id
        string kind
        string subject
        string relation
        string object
        float confidence
        boolean promoted
        string support_kind
        json provenance
        json payload
        timestamp created_at
    }

    world_specs {
        integer id
        string name
        string status
        json payload
        timestamp created_at
    }

    generated_experiments {
        integer id
        string world_name
        string action
        json provenance_ids
        json payload
        timestamp created_at
    }

    validation_results {
        integer id
        string target
        boolean valid
        json payload
        timestamp created_at
    }

    research_events {
        integer id
        string status
        string url
        json payload
        timestamp created_at
    }
```

Current writes:

- `knowledge_atoms`: written by `CorpusIngestor` through
  `PersistentStore.record_knowledge_atom`.
- `world_specs`: written by `darwin ingest-corpus` and by v4 brain startup when
  it generates missing specs.
- `experiments`: existing experiment table used for evaluated experiment
  outcomes.
- `generated_experiments`, `validation_results`, `research_events`: schema
  exists, but this branch does not yet use them as the primary write path.

## Commands

Create a tiny corpus:

```bash
cat > /tmp/darwin-force.txt <<'EOF'
== Force ==
Force is an interaction that changes motion.
Force causes acceleration.
Aliases: push, pull
EOF
```

Ingest it into a separate memory file:

```bash
darwin ingest-corpus \
  --source wikidump \
  --path /tmp/darwin-force.txt \
  --memory /tmp/darwin-v4.sqlite3
```

Start the v4 brain:

```bash
darwin brain \
  --kernel v4 \
  --workers auto \
  --accelerator auto \
  --memory /tmp/darwin-v4.sqlite3
```

Connect from another terminal:

```bash
darwin connect
```

Useful v4 chat commands:

```text
/knowledge force
/hypotheses
/worlds
/mind
/research status
/why Force causes acceleration
```

## Current limits and future work

Implemented now:

- deterministic curated corpus ingestion
- `KnowledgeAtom` plus `Provenance`
- persisted `KnowledgeGraph` search
- generated data-only `WorldSpec` records
- `SandboxedWorldCompiler` validation
- `GenerativeUniverseAdapter` through the existing adapter protocol
- v4 brain startup path
- corpus-answering through the discourse planner
- promotion after generated experiments
- dormant disabled live research surface
- v4 scheduler/metrics surface

Not implemented yet:

- full-scale Wikipedia/Wikidata dump processing
- robust entity linking, contradiction handling, or poisoning checks
- live web research activation
- a complete actor-runtime replacement for the background loop runtime
- rich world generation with invariants and experiment templates
- hardware accelerator execution beyond the current CLI/scheduler surface
- universal sentience or unconstrained autonomous world growth
