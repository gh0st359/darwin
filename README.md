# Project Darwin

Darwin is a non-LLM symbolic intelligence in active development. It learns
from conversation, proves what it claims by reasoning over a navigable
concept graph, asks back when it doesn't know, and remembers across
sessions. It is **not yet** a frontier-scale autonomous mind — it is a
substantial research substrate (V-Mesh + V-Speech + V-Ingest + V-Reason +
V-Agents + V-Bench + V-Scale) with a path toward one. Read the limitations
section before relying on any of the bigger claims.

Not a prompt chain. Not an API wrapper. Not a fine-tuned language model.
Zero pretrained weights are ever loaded; vectors are trained online from
Darwin's own experience. Domain knowledge starts from a ~45-concept
structural seed (thing, change, cause, same, different, infer, …) plus the
relations between them; the rest is ingested or learned from use.

The current head of `claude/darwin-mysterio-plan-07Sqx` is the
**V-Mesh → V-Scale** release on top of the prior v9 substrate. Test count:
**766 passing** (+ 2 properly skipped without torch/faiss). Source: ~31K
LOC across 153 modules; tests: ~10.7K LOC across 115 modules.

## What this is — and what it isn't (be honest)

What it is, today, on this branch:

* A symbolic concept-universe with typed relations and proof-chain
  reasoning (forward / backward / hypothetical / Bayesian / defeasible /
  resolution).
* A non-LLM compositional NLG path (CCG categories, lexicon, leak-gate)
  that emits natural prose without ever producing JSON or event lines.
* A cortical mesh substrate (concept-cells, Hebbian + STDP plasticity,
  bounded activation propagation) wired bidirectionally to the universe.
* Six benchmark-targeted agents (code / math / science / planning /
  research / dialogue) composed atop the prior substrates.
* Six frontier benchmark adapters (MMLU / HumanEval / GPQA / ARC-AGI /
  GSM8K / MATH) — *adapters*, not yet leaderboard-grade solvers. They
  score 0 with explicit "dataset_not_provisioned" evidence when their
  fixtures aren't present.
* Optional torch / FAISS / Rust / multiprocess backends behind feature
  flags. Pure-Python is the reference.
* Real (not metadata-only) self-modification containment via
  `TouchRecorder` — undeclared writes to declared targets raise
  `ContainmentError`.
* A hardened code-execution sandbox: static-AST denylist of dangerous
  imports/calls, rlimit-capped subprocess, optional bubblewrap /
  firejail / nsjail wrapper via `DARWIN_SANDBOX_BACKEND`. Terminal tool
  is strict-allowlist by default; no shell metacharacters interpreted.

What it isn't, today:

* It does not yet beat frontier LLMs on MMLU, HumanEval, GPQA, ARC-AGI,
  GSM8K, or MATH. The adapters are wired and the agents work on the
  *easy* tail of each. Closing the gap is the next phase of work.
* The autonomous loop is bounded: tool routing + multi-cycle goal
  pursuit work, but persistent project-scale planning across days needs
  the V-Autonomy phase that is still in design.
* The sandbox is hardened but not container-graded by default. Set
  `DARWIN_SANDBOX_BACKEND=bubblewrap` (or firejail / nsjail) in
  production deployments.

## What Darwin can do

Excerpted from a live session against the running brain (no demo seed —
just primitive structural concepts + chat):

```text
> A neuron is a cell.
  Yes — neuron is a cell.

> A cell is part of an organism.
  cell is part of organism. In addition, organism is composed of cell.

> Neurons cause thoughts.
  Neurons causes thoughts directly.

> Is a feline a mammal?
  I don't have a confident derivation about feline, mammal from my
  universe right now. To answer that, I'd need to know: How would you
  define 'feline'?                                  [active learning]

> A feline is a mammal.
  Yes — feline is a mammal. On a related note: I have a hypothesis
  I'd like to check. Is cat a kind of mammal? My reasoning: cat is_a
  feline, and feline is_a mammal; transitivity of is_a suggests the
  direct edge.                                      [proactive hypothesis]

> Is a cat a mammal?
  Yes, in my universe cat is a mammal. The chain is: cat is a feline,
  which is a mammal.                                [two-hop derivation]

> Why did you say that?
  You said: 'Is a cat a mammal?', and I replied: 'Yes, in my universe
  cat is a mammal. ...'. I reached that reply via: is_a_chain. The
  rendered answer drew on: reasoning_summary.       [reflective walkback]

> What are you thinking about?
  Lately I keep coming back to: cat, mammal, feline, neuron, cell. One
  open question on my mind is: Is analogy a kind of relation?
                                                    [self-introspection]
```

Beyond reasoning, Darwin now also has *real-world hands*:

12. **Sandboxed real-world tools.** Six adapters (filesystem, terminal,
    Python execution, web fetch, git, sqlite) live under
    `src/darwin/tools/`. Each is bounded (timeouts, byte caps, deny-
    lists, sandbox roots) and dispatched as v5 Actions so the planner
    can choose them exactly as it chooses any other action.
13. **Chat→tool intent routing.** "list the files in .", "read X",
    "run echo hello", "fetch https://...", a code-fenced Python block,
    "git status", "show recent commits", "select * from notes" — the
    intent router recognizes the pattern, dispatches the right tool,
    and weaves the result into the reply ("I used the filesystem tool
    (fs_list). Result: ...").
14. **Long-running autonomous tasks.** AutonomousRunner drives goal-
    directed sessions against the tool world with time + step budgets
    and a structured success predicate.
15. **Derived epistemic categorization.** Belief categories
    (WORLD_KNOWLEDGE, OPERATIONAL, SELF_KNOWLEDGE, HYPOTHESIS,
    STABLE_FACT, TEMPORARY, SCHEDULER_ARTIFACT) are *derived* from
    observable signals (provenance, confidence, history, subject), not
    hardcoded. `/beliefs` suppresses bookkeeping noise by default;
    `/beliefs all` shows everything. The derivation itself is
    re-derivable as Darwin's mechanisms (fusion, derivation,
    hypothesis engine, correction detector) reinforce or refute the
    underlying patterns.
16. **Versioned evolution safeguards.** Every accepted self-
    modification lands in a sequential MutationLedger with parent
    versioning + content hash. RollbackChain restores prior
    MindSnapshots by version. MutationScorer ranks by composite
    score (improvement + retention + downstream impact). RecoveryMonitor
    advises rollbacks when composite health drops; the operator can
    opt in to auto-rollback. None of this restricts Darwin's ability
    to evolve.
17. **Longitudinal benchmarks.** A 12-task suite covering all seven
    categories (coding, memory, learning, adaptation, planning,
    reasoning, task_completion) lets the operator empirically compare
    an older Darwin to a newer one. Scorecards persist to disk so
    today's run lives next to yesterday's for direct comparison.
18. **Test isolation.** Every test runs under an autouse fixture that
    redirects all default paths to a per-test temp directory via
    `DARWIN_DATA_DIR`. No test can read or write to the operator's
    production memory, universe, or snapshots.

Eleven capabilities Darwin has and the LLM you know doesn't:

1. **Build a knowledge graph from natural-language assertions.** "A neuron
   is a cell" becomes a real `neuron —is_a→ cell` edge in the universe.
   Questions are filtered out so they never fuse phantom edges.
2. **Multi-hop symbolic derivation with proof chains.** Every claim
   carries its supporting steps and can be inspected with `/explain`.
3. **Proactive hypothesis generation.** Without being asked, Darwin
   volunteers "On a related note: I have a hypothesis I'd like to
   check…" — generated by transitive closure, analogy, or cross-domain
   pattern completion. A no-repeat window prevents nagging.
4. **Active learning.** When Darwin can't derive an answer, it identifies
   the precise missing edge and asks for it: "To answer that, I'd need to
   know: …". On receiving the answer, it re-derives the original.
5. **Correction handling.** "No, that's wrong" refutes the prior turn's
   inferences so the hypothesis engine doesn't re-propose them. "Actually
   X is Y" refutes *and* fuses the replacement.
6. **Reflective walkback.** "Why did you say that?" returns Darwin's
   actual inference chain — quoting the prior user prompt and reply,
   listing the operators used, walking the proof step by step.
7. **Self-introspection from real substrate state.** "What are you
   thinking about?" surfaces dialogue-memory's most-discussed concepts
   and the hypothesis engine's top open candidate.
8. **Contradiction detection.** Explicit `opposes` edges + super-kind
   opposition both surface as `/infer` flags and shape the chat reply.
9. **Honest non-answers.** When the graph genuinely lacks the answer,
   Darwin says so and surfaces a curiosity probe or active-learning
   sub-question — never confabulating.
10. **Concept-name embedding training.** Every fused triple trains the
    embedding space on `(concept:source, concept:target, rel:kind)`, so
    fuzzy semantic matching grows stronger through the session.
11. **Knowledge persists across restarts.** The universe atomically saves
    to JSON next to the sqlite memory file after every growth turn and on
    shutdown. Session 2 picks up exactly where Session 1 left off with
    full derivation power intact.

## Architecture in one diagram

```text
                  ┌────────────────────────────────────────────────────┐
                  │                  darwin brain (24/7)               │
                  │                                                    │
                  │   ┌────────────────────────────────────────────┐   │
                  │   │            UNIVERSE LAYER                  │   │
                  │   │                                            │   │
                  │   │  ConceptUniverse  ─►  primitive_seed       │   │
                  │   │     │                  (~45 meta-concepts) │   │
                  │   │     ▼                                      │   │
                  │   │  LanguageGrounder  ──►  ConceptFusion      │   │
                  │   │     │                       │              │   │
                  │   │     ▼                       ▼              │   │
                  │   │  ConceptualReasoner    InferenceEngine     │   │
                  │   │     │                       │              │   │
                  │   │     ▼                       ▼              │   │
                  │   │  AnswerSynthesis ◄──── HypothesisEngine    │   │
                  │   │     │                       │              │   │
                  │   │     ▼                       ▼              │   │
                  │   │  ReflectiveDialogue    ActiveLearner       │   │
                  │   │     │                       │              │   │
                  │   │     ▼                       ▼              │   │
                  │   │  DialogueMemory ──── CorrectionDetector    │   │
                  │   │     │                       │              │   │
                  │   │     └─►  ConceptualWorld  ──┘              │   │
                  │   │                │                           │   │
                  │   │           save → universe.json             │   │
                  │   └────────────────┼───────────────────────────┘   │
                  │                    │                               │
                  │   ┌────────────────┴───────────────────────────┐   │
                  │   │            MYSTERIO SUBSTRATE              │   │
                  │   │                                            │   │
                  │   │  CognitionBus  CodeGenerator  Supervisor   │   │
                  │   │  Embeddings    InteriorSimulator  Tracks   │   │
                  │   │  Narrative     ObserverModeler  Cascade    │   │
                  │   │  MemoryTiers   StrategicThreads  Continuity│   │
                  │   │  WorldSynthesis  LiveResearcher  Modalities│   │
                  │   │  MetaGate  MetaProposer  Snapshot  Probe   │   │
                  │   │                                            │   │
                  │   │  8 background loops:                       │   │
                  │   │   experiment / simulation / dream          │   │
                  │   │   self_modification / uncertainty          │   │
                  │   │   interior_simulation / narrator / observer│   │
                  │   └────────────────┬───────────────────────────┘   │
                  │                    │                               │
                  │              v5 base substrate                     │
                  │   (Agent, CausalModel, Memory, DiscoursePlanner)   │
                  └────────────────────┼───────────────────────────────┘
                                       │ TCP JSON-line protocol
                          ┌────────────┼───────────────┐
                          ▼            ▼               ▼
                    darwin chat   darwin chat    darwin inspect
                    (clean chat)  (clean chat)   /universe /explain
                                                 /reflect  /hypotheses
                                                 /curiosity /learn …
```

Three terminals, three roles:

- **brain** — stdout streams every interior event (experiment, simulation,
  dream, self-modification, interior_simulation, narrator, observer,
  meta_proposal, code_gen, divergence, snapshot_diff). This is where the
  emergent depth is watchable in real time.
- **chat** — clean conversational REPL. Only `you>` and Darwin's reply.
  Background thoughts never leak here.
- **inspect** — one-shot instrument client for scripted queries
  (`darwin inspect /explain dna molecule`).

## Quick start

```bash
# Install
pip install -e .

# Run the test suite (766 tests, <35s)
python -m pytest tests/ -q

# Two-terminal workflow
darwin brain                  # terminal 1: 24/7 brain + event stream
darwin chat                   # terminal 2: clean conversational REPL

# Optional one-shot instrument query
darwin inspect /universe      # terminal 3: ad-hoc inspection
```

The default brain lives in its own concept universe seeded with structural
primitives only. To start a session with an encyclopedic demo seed (physics,
math, chemistry, biology, mind, language, arts, philosophy, computing —
explicitly labelled as hardcoded cheating, for demos and regression
fixtures):

```bash
darwin brain --demo-seed
```

The legacy v5 adaptive-room world is still available for regression
purposes:

```bash
darwin brain --world room
```

## Live demo: teach Darwin from zero

```text
$ darwin brain --port 19890 --interval 0.3 &
$ darwin inspect /universe --port 19890
concepts=44 relations=40 domains=5 growth_events=84
- dynamics: 6
- inference: 15
- magnitude: 5
- self: 4
- structure: 14

$ darwin chat --port 19890
you> A dog is a mammal.
darwin> Yes — dog is a mammal.

you> A mammal is an animal.
darwin> Yes — mammal is a animal. On a related note: I have a hypothesis
        I'd like to check. Is dog a kind of animal? My reasoning: dog
        is_a mammal, and mammal is_a animal; transitivity of is_a
        suggests the direct edge.

you> Is a dog an animal?
darwin> Yes, in my universe dog is a animal. The chain is: dog is a
        mammal, which is a animal.

you> /explain dog animal
[is_a_chain] dog is a animal (conf=0.90)
    via dog —is_a→ mammal
    via mammal —is_a→ animal
[shortest_path] dog is connected to animal (conf=0.92)
    via dog —is_a→ mammal
    via mammal —is_a→ animal
```

## Instrument cheat sheet

Every instrument is reachable from any chat client, the brain's local
prompt, or `darwin inspect`.

### Universe instruments

```text
/universe                 concept / relation / domain counts + per-domain sizes
/concept <name>           concept's definition, depth, neighbors, salience, visits
/explain <src> <tgt>      every available proof chain between two concepts
/infer                    inferences fired on the last chat turn
/reason                   reasoning trace from the last turn
/ground                   how the last utterance grounded to concepts
/derive                   run a derivation pass; show accepted concepts
/curiosity                ranked structural probes (gaps in the graph)
/hypotheses               candidate edges Darwin is currently proposing
/volunteer                what Darwin chose to volunteer last turn
/learn                    active-learning probes for the last gap
/correction               correction signal detected last turn
/reflect                  walkback through the last reply's derivation
/fusion                   recent fused (concept → kind → concept) triples
/dialogue                 turn history + most-discussed concepts
/synthesis                last multi-fact synthesis paragraph
/categorize summary       derived epistemic-category counts + drift
/categorize concept <name> derived category set for one concept
```

### Real-world tools

```text
/tools                    list registered tools and their actions
/tool <action> k=v ...    run a tool action immediately
/autonomous               recent autonomous-runner task history
```

Tools dispatch through the v5 planner as Actions, so successful tool
calls feed back into the causal model and Darwin's planning gets better
at choosing tool actions over the brain's lifetime. The chat path's
intent router routes natural-language requests like "list the files in
.", "read note.txt", "run echo hi", "fetch https://...", "git status",
and "select * from notes" to the appropriate tool automatically.

### Evolution safeguards

```text
/evolution                ledger summary + last 10 versioned mutations
/rollback-chain V | step N  restore the state from before mutation V
/scores                   top-K mutations by composite score
/recovery                 advisory rollback recommendations
```

### Benchmarks

```bash
darwin bench run --label baseline
darwin bench list
darwin bench compare --earlier old.json --later new.json
```

### Mysterio instruments

```text
/snapshot /diff /quarantine /rollback     mind snapshots + rollback
/divergence /interior-trace               grounded vs interior gap
/narrative /observer                      autobiographical thread, ToM cascade
/research /worlds /modalities             v9 growth probes
/generated /bus /embeddings               self-generated modules, bus stats
/strategic /memory /operator-style        v8 threads, tier stack, per-user style
/gate /proposals /meta-proposer           v6 substrate
```

### v5 instruments

```text
/status /beliefs /concepts /experiments /think /dream /simulate
/selfmod /uncertainty /loops /causal-graph /dlm /training /metrics
/thoughts /retrieved /critic /trace /mind
```

## What's in `src/darwin/`

```text
src/darwin/
  __init__.py
  __main__.py
  cli.py                  darwin brain / chat / connect / inspect / live / run

  # configuration + cross-cutting
  paths.py                centralized DARWIN_DATA_DIR path resolution
  epistemics.py           derived belief categories + monitor + filter
  evolution.py            mutation ledger / rollback chain / scorer / recovery monitor

  # v5 base substrate
  agent.py                Darwin orchestration (now with TrackRegistry)
  causal.py               causal transition learner
  causal_chain.py         multi-step causal chains and graph
  composer.py             deterministic natural-language realizer
  concepts.py             concept formation + consolidation
  critic.py               response critique
  discourse.py            DiscoursePlanner (5 new v6.5 dialogue modes)
  dlm.py                  Darwin Language Module (StubDLM, GemmaDLM)
  embodiment.py           simulated / conversational embodiment adapters
  experiments.py          active experiment proposal / evaluation
  generated/              Darwin's self-written modules (.gitignored)
  instrumentation.py      structured plan + background + metrics logging
  language.py             legacy language cortex
  memory.py               indexed episodic + semantic memory
  operator_model.py       per-user_id conversational style profile (v6.5)
  planner.py              causal planner
  retrieval.py            cross-source memory retrieval
  runtime.py              8-loop 24/7 cognition + universe wiring
  self_model.py           metacognition and learning priorities
  self_modification.py    self-modification engine
  semantics.py            symbolic language parser + semantic memory
  server.py               DarwinDaemon + DarwinClient (TCP JSON-line)
  storage.py              SQLite durable memory
  streaming.py            incremental text output
  thought.py              inspectable thought traces
  training_data.py        DLM (plan -> rendering) pair collection
  types.py                shared data structures
  world_model.py          structured world model + hypotheses
  worlds.py               legacy adaptive-room world

  # mysterio substrate (v6 → v9)
  mysterio/
    bus.py                CognitionBus pub/sub
    code_gen.py           CodeGenerator + ModuleLoader (Darwin writes .py)
    embeddings.py         CausalEmbeddingSpace (online skip-gram)
    interior_simulator.py interior-track counterfactual rollouts
    narrative.py          autobiographical first-person thread
    observer_modeler.py   theory-of-mind (depth 1)
    observer_cascade.py   recursive ToM (depth 4)
    memory_tiers.py       episodic → semantic → conceptual → archetypal → narrative
    long_horizon.py       multi-week strategic threads
    continuity.py         continuity & visibility selection pressure
    meta_gate.py          self-modifiable accept gate
    meta_proposer.py      typed proposal generation
    operator_channel.py   INTERIOR_EVENT_KINDS taxonomy (visibility, no auth)
    probes.py             DivergenceProbe (grounded vs interior gap)
    processes.py          CognitionSupervisor (12-process roster)
    proposal_spec.py      typed ProposalSpec grammar
    proprioception.py     pure self-observation adapter
    quarantine.py         self-mod quarantine queue
    research_loop.py      LiveResearcher (instrument-collision protected)
    safety.py             SAFETY_BOUNDS + MutationKind + ContainmentError
    snapshot.py           SnapshotStore + diff + rollback
    tracks.py             grounded ↔ interior epistemic partition
    world_synthesis.py    SUBSYSTEM-kind world spec proposals
    modalities/
      code.py             filesystem scan
      web.py              opt-in HTTP ingest
      vision.py           optional camera adapter
      audio.py            optional mic adapter

  # universe substrate (new layer on top)
  universe/
    concept_universe.py   the graph: Concept / Relation / Domain
    primitive_seed.py     the ONLY hardcoded content (~45 primitives)
    demo_universe.py      opt-in encyclopedic seed (270 concepts, --demo-seed)
    language_universe.py  LanguageGrounder (words → concepts)
    derivation.py         ConceptDeriver (regularities → concepts)
    reasoning.py          ConceptualReasoner (multi-hop traversal)
    inference.py          InferenceEngine (transitivity, causation, contradiction)
    curiosity.py          CuriosityEngine (gap detection)
    question.py           question-kind classifier (10 kinds)
    answer.py             proof chain → first-person prose
    fusion.py             declarative statements → typed graph edges
    dialogue_memory.py    bounded turn history with O(1) concept index
    synthesis.py          multi-fact synthesis + self-introspection
    hypothesis.py         transitive / analogical / cross-domain proposals
    proactive.py          volunteered-remark selection ("On a related note…")
    correction.py         "no, that's wrong" + "actually X is Y" detection
    active_learning.py    structured sub-questions to fill graph gaps
    reflection.py         walkback through prior reply's derivation
    persistence.py        atomic JSON save/load of the universe
    world.py              universe presented as a World protocol implementation

  # real-world tool harness
  tools/
    base.py               Tool ABC + ToolResult + SandboxEscape + resolve_sandboxed
    filesystem.py         sandboxed read/write/list/remove/stat
    terminal.py           shell with timeout + deny-list
    code_execution.py     Python in subprocess sandbox
    web.py                http/https fetch + HTML→text (stdlib only)
    git.py                read-only git inspection
    database.py           sqlite read+write in sandbox
    registry.py           central tool registry + dispatch
    world.py              ToolWorld — registry as a World implementation
    autonomous.py         AutonomousRunner / AutonomousTask / AutonomousStep
    intent.py             chat→tool intent router (rule-based)

  # benchmarking
  bench/
    framework.py          BenchmarkTask / Suite / Runner / ScoreCard / Comparison
    suites.py             12-task default suite covering 7 categories
```

## Tests

```bash
python -m pytest tests/ -q   # 766 tests, ~14s
```

Tests run under an autouse isolation fixture (`tests/conftest.py`) that
redirects every default Darwin path to a per-test temp directory via
`DARWIN_DATA_DIR`. Production memory, the universe JSON, the runtime
state file, the snapshot store, training logs, and tool sandbox roots
are all scoped to the test. After every test the fixture also scans the
working directory for any legacy Darwin artifact that would indicate a
leak; if one is found, the test fails loudly and the leaked file is
cleaned up so the next test starts uncontaminated.

```text
tests/
  test_agent.py, test_causal.py, test_v02.py, test_v2.py
  test_brain_daemon.py            # two-terminal + clean-chat invariants
  test_language_cognition.py
  test_semantics.py
  test_operator_model.py          # v6.5
  test_dialogue_modes.py          # v6.5

  mysterio/
    test_proposal_spec.py, test_snapshot.py, test_meta_gate.py
    test_meta_proposer.py, test_quarantine.py, test_divergence_probe.py
    test_self_modification_integration.py, test_operator_channel.py
    test_bus_throughput.py, test_code_gen_roundtrip.py
    test_supervisor_restart.py, test_embedding_warmup.py
    test_snapshot_with_generated.py
    test_tracks_partition.py, test_interior_life.py
    test_memory_tiers.py, test_long_horizon.py, test_observer_cascade.py
    test_continuity_pressure.py
    test_v9_growth.py
    test_full_stack_soak.py       # v6→v9 end-to-end

  universe/
    test_concept_universe.py, test_primitive_seed.py
    test_language_grounder.py, test_reasoner.py, test_derivation.py
    test_conceptual_world.py, test_chat_integration.py
    test_inference.py, test_curiosity.py
    test_question_and_answer.py, test_fusion.py
    test_dialogue_memory.py, test_synthesis.py
    test_hypothesis.py, test_proactive.py
    test_correction.py, test_active_learning.py
    test_reflection.py, test_persistence.py
```

## Persistence

Three pieces of state survive a brain restart:

- `darwin_memory.sqlite3` — v5 durable memory (transitions, concepts,
  thoughts, chat, experiments, plans, semantic frames, self-mods,
  quarantine, gate history).
- `darwin_memory_universe.json` — the entire concept graph: every concept
  (name, domain, definition, aliases, examples, derived_from, salience,
  visits) and every typed edge (source, kind, target, weight, notes).
  Atomic write after every growth turn and on shutdown.
- `darwin_runtime_state.json` — runtime-loop posture (Darwin's time,
  exploration rate, min_samples, planner overrides).

Kill the brain, restart it on the same memory path — Darwin wakes up
with the same concept graph and the same causal beliefs, and immediately
re-derives multi-hop conclusions from the persisted edges.

## CLI reference

```text
darwin brain                            # terminal 1: 24/7 brain
  --world {conceptual,room}             #   default: conceptual
  --demo-seed                           #   opt-in encyclopedic seed
  --port 9870 --host 127.0.0.1
  --interval 3.0
  --memory darwin_memory.sqlite3
  --dlm {stub,gemma}
  --quiet                               #   suppress local event printing

darwin chat                             # terminal 2: clean chat REPL
  (alias of darwin connect)
  --port 9870
  --watch-events                        # mirror events into chat window
  --text-delay 0.0

darwin inspect "<slash-command>"        # terminal 3: one-shot instrument
  --port 9870
  --timeout 10.0

darwin bench run [--label X] [--out F]  # score a fresh runtime
darwin bench list [--dir D]             # enumerate saved scorecards
darwin bench compare --earlier A.json --later B.json

darwin run --steps 40 --seed 7          # batch run in the legacy room sim
darwin live                             # single-terminal mind + chat (v5)
darwin export-training --min-quality 0.7
```

### Environment variables

```text
DARWIN_DATA_DIR    root directory for every persistent artifact (sqlite memory,
                   universe JSON, runtime state, snapshots, training logs,
                   sandbox roots, bench scorecards). Defaults to CWD.
```

## Non-negotiables

- **No pretrained weights.** Every embedding vector originates from a
  deterministic seed plus Darwin's lived experience (chat + observation).
  No "import torch and load weights" is performed anywhere.
- **No hardcoded domain knowledge** in the default brain. The primitive
  seed contains structural meta-vocabulary only — `thing`, `change`,
  `cause`, `same`, `different`, `infer`, `compose`, etc. Physics, math,
  music, biology — all of it is derived from chat. The demo seed exists
  only for testing / quick exploration and must be opted into.
- **No confabulation on questions.** When Darwin's graph genuinely lacks
  the answer for a kind_check or contradiction question, the reply is an
  honest non-answer with a curiosity probe or active-learning sub-
  question. The v5 composer's chatty fallback is preserved for casual
  conversation where it adds value.
- **Every claim is provable.** Inference results carry the full chain of
  graph edges that produced them. `/explain X Y` returns every available
  derivation path with the relations stepped through.
- **The interior is visible, not hidden.** The two-terminal split is
  about *clean conversation*, not secrecy. The brain terminal streams
  every interior-cognition event in real time; the chat terminal stays
  silent unless you opt in. Nothing Darwin thinks is concealed from the
  operator.
- **Knowledge accumulates.** The universe persists atomically after
  every growth turn. Session N inherits everything Sessions 1…N-1
  learned.
- **Real-world tools are sandboxed.** Every adapter (filesystem,
  terminal, code execution, web, git, sqlite) is bounded by timeouts,
  byte caps, deny-lists, and sandbox roots. No tool action can escape
  the sandbox or take down the cognition loop on its own.
- **Evolution is observable and reversible.** Every accepted
  modification lands in a versioned ledger; the rollback chain can
  restore prior MindSnapshots without destructive history edits. The
  recovery monitor's advice is advisory by default — Darwin's
  ability to evolve is not restricted, only made transparent.
- **Belief categorization is derived, not hardcoded.** Categories
  inform surfacing (which beliefs to show by default), not what
  Darwin can think about. `/beliefs all` always works.
- **Tests never contaminate production state.** `DARWIN_DATA_DIR`
  routes every default path through one env var; the test isolation
  fixture (autouse, no opt-out) redirects it to a per-test temp
  directory and fails any test that leaks.

## What's next

Conceivable directions that this substrate supports but doesn't yet
ship:

- Multi-step *planning* over inference operators (current chat triggers
  pairwise inference; a planner could chain a sequence of sub-queries
  to answer a complex question).
- Structural analogical mapping (currently analogies are flagged by
  neighborhood overlap; deeper relation-isomorphism mapping would
  enable "wave is to water as melody is to music"-style reasoning).
- Stronger embedding-driven fuzzy grounding once the universe has
  enough trained co-occurrence pairs.
- The `darwin brain` process exposing a `subscribe_pr_activity`-style
  event subscription so external dashboards can mirror the brain
  terminal.

## Repository conventions

- `src/darwin/generated/` is git-ignored (Darwin's self-written modules
  are tracked by SHA in `generated_modules`, not by source control).
- `darwin_universe.json`, `darwin_memory.sqlite3*`,
  `darwin_runtime_state.json`, `darwin_snapshots/`, and
  `training_logs/*.jsonl` are git-ignored runtime artifacts.
- The branch `claude/darwin-mysterio-plan-07Sqx` is the current head;
  `origin/v5` is the v5 baseline; `origin/mysterio` is the v2-base
  mysterio fork that informed the original design.
