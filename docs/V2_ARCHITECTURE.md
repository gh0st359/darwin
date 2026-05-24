# Darwin v2: Eternal Causal Mind

This document describes the v2 architecture and how each phase of the
plan is implemented. The high-level goal: turn Darwin into a true
continuously-thinking causal intelligence that runs 24/7, keeps improving
itself, and can hold natural, flowing conversations — without ever
delegating its actual reasoning to a pre-trained model.

## Non-negotiable principles

1. Darwin's mind is the always-on symbolic + causal engine.
2. The Darwin Language Module (DLM) is a thin renderer only. It takes a
   structured `ResponsePlan` from Darwin and renders it as prose. It does
   zero reasoning and holds zero world knowledge of its own.
3. No external LLM is ever allowed to generate Darwin's thinking,
   concepts, or causal rules.
4. Every output that touches a user passes through `FaithfulnessValidator`
   plus `ResponseCritic`; renderings that drift are rejected.

## Phase 0 — Foundation & instrumentation

Implemented in `darwin/instrumentation.py`, `darwin/thought.py`,
`darwin/discourse.py`.

* `ResponsePlan` carries explicit `causal_claims`, `uncertainty_levels`,
  `referenced_experiences`, `self_reflection`, `tone`, `target_length`,
  and a stable `plan_id`. The `to_dlm_payload()` method emits the
  strictly-shaped contract that the DLM is allowed to see.
* `ThoughtTrace` gets a `trace_id`, a per-step `payload`, timestamps and
  a `duration_ms` accessor — so every reasoning episode is fully
  machine-readable.
* `StructuredLogger` writes JSONL plan logs, JSONL background-cognition
  logs, and a metrics log. The plan log doubles as the raw corpus for
  phase 4 (DLM training data collection).

## Phase 1 — Core intelligence upgrades

### Multi-step causal reasoning

`darwin/causal_chain.py` adds `CausalChain`, `ChainNode`, `CausalGraph`,
and `CausalChainEngine`. The engine rolls the existing `CausalModel`
forward across an action sequence, propagating uncertainty
multiplicatively per step and accumulating expected reward. `CausalGraph`
distills the learned beliefs into an action→variable graph that the CLI
can inspect via `/causal-graph`.

`CausalModel.predict_chain()` and `CausalModel.chain_uncertainty()`
expose the chain primitives directly on the existing model.

`CausalPlanner.plan_sequence()` now attaches a `CausalChain` to each
`MultiStepPlan` so downstream consumers (DLM, instrumentation,
background loops) see the same chain as the planner used.

### Advanced memory retrieval

`darwin/memory.py` upgrades `EpisodicMemory` with index dictionaries by
action and variable, and helpers like `changed_variable()`,
`positive_reward()`, and `temporal_distance()` that are used both by
retrieval and by self-modification's holdout evaluation.

`darwin/retrieval.py` now ranks across:
* semantic-memory frames (already existed),
* concept index,
* causal beliefs (already existed),
* runtime events,
* episodic transitions (new), with grounding overlap × recency × reward,
* completed experiments (new).

A `retrieve_for_topic()` helper is used by background loops that don't
have a `SemanticFrame` to drive retrieval from.

### Self-modification engine

`darwin/self_modification.py` proposes small tweaks to Darwin's own
knobs: `causal_model.min_samples`, `darwin.exploration_rate`,
planner exploration bias, and pruning of low-support concepts.
Every proposal carries `apply` and `revert` closures and is evaluated
against held-out recent transitions; only changes that measurably
reduce prediction error are kept. The full outcome record is persisted
through `PersistentStore.record_self_modification()`.

## Phase 2 — True 24/7 always-on persistent mind

`darwin/runtime.py` becomes a multi-threaded background cognition
system. Each cognitive activity runs in its own thread on its own
cadence:

| Loop                | Default interval | What it does                                       |
| ------------------- | ---------------- | -------------------------------------------------- |
| `experiment`        | `interval`        | runs an active uncertainty-reducing intervention   |
| `simulation`        | `1.5 × interval`  | mental simulation across causal chains             |
| `dream`             | `4 × interval`    | memory consolidation + concept salience            |
| `self_modification` | `6 × interval`    | proposes + tests + accepts/rejects self changes    |
| `uncertainty`       | `3 × interval`    | per-action uncertainty scan from current state     |

The main thread handles user conversation. All loops share Darwin
through a single `RLock`. Every wake-up emits a visible `RuntimeEvent`
(`event.loop` carries which loop produced it) and a structured
`BackgroundLogEntry` in JSONL.

Runtime state (Darwin time, exploration rate, min_samples, planner
overrides, per-loop snapshots) is checkpointed to
`darwin_runtime_state.json` on stop and restored on construction, so
Darwin "wakes up" with the same internal posture it had when it last
slept.

## Phase 3 — Darwin Language Module (DLM)

`darwin/dlm.py` defines the `DarwinLanguageModule` Protocol and two
implementations:

* `StubDLM` — wraps the deterministic `NaturalLanguageComposer`. This is
  the default. It always returns a `DLMRenderResult` and lets the rest
  of the system treat the renderer as a first-class module.
* `GemmaDLM` — runs gemma-3-270m locally through one of three backends:
  Ollama HTTP (`OLLAMA_HOST`, `DARWIN_DLM_MODEL`), `llama-cpp-python`
  with a GGUF file (`DARWIN_DLM_GGUF`), or `transformers` pipeline
  (`DARWIN_DLM_HF_MODEL`).

The DLM's only allowed input is `ResponsePlan.to_dlm_payload()`. It
receives a very strict system prompt (no invention, preserve every
causal claim, surface every uncertainty, prose only, no parser
notation). The output is then run through `FaithfulnessValidator`
which checks for:
* leaked parser notation,
* forbidden "I am an AI / training" phrases,
* missing high-confidence causal claims,
* unsurfaced high-impact uncertainty levels,
* missing clarification questions,
* hallucinated numeric values,
* responses outside `target_length`.

If the renderer's output fails validation, the runtime silently falls
back to the deterministic composer. The DLM is therefore strictly
optional — Darwin's mouth is never required to produce its mind.

## Phase 4 — Training data strategy

`darwin/training_data.py` provides `TrainingDataCollector`. Every time
the runtime renders a plan it writes a `(plan_payload, rendering,
renderer, critique_passed, quality)` tuple to
`training_logs/dlm_training_pairs.jsonl`. The CLI's
`darwin export-training` subcommand exports an accepted subset for
fine-tuning the DLM (default min quality 0.7).

The collector is deliberately additive: it never affects Darwin's
behaviour. The intended workflow is:

1. Let Darwin run live and in simulation; thousands of `ResponsePlan`s
   collect with deterministic composer renderings.
2. Curate a small set of (plan → rendering) pairs by hand for
   fluency-target quality.
3. *Optionally and exactly once*, run one heavily filtered pass through
   a larger model to polish renderings; validate each candidate against
   the same `FaithfulnessValidator`. Never use that larger model again.
4. Fine-tune gemma-3-270m with LoRA on the resulting corpus.
5. Periodically retrain as Darwin's plans get richer.

The collector summary is visible at any time through `/training`.

## Phase 5 — Integration, testing, hardening

* The DLM is optional; with `--dlm stub` (default) Darwin uses the
  deterministic composer. With `--dlm gemma` it uses gemma-3-270m and
  silently falls back on validation failure.
* All output is double-gated: `FaithfulnessValidator` (renderer-side)
  then `ResponseCritic` (Darwin-side). Both produce structured
  critique records that get logged.
* New CLI commands:
  * `/simulate` — run a mental simulation now
  * `/selfmod` — propose + test self-modifications now
  * `/uncertainty` — latest per-action uncertainty scan
  * `/loops` — background loop status
  * `/causal-graph` — distilled action→variable graph
  * `/dlm` — current DLM and last render validation notes
  * `/training` — DLM training-data corpus summary
  * `/metrics` — structured-logger metrics snapshot
* New tests in `tests/test_v2.py` cover instrumentation, causal
  chains, advanced retrieval, self-modification, multi-threaded runtime
  loops, persistent state, DLM rendering, faithfulness validation,
  fallback, training-data collection, storage, and the chat→DLM
  payload round-trip.

## Build order followed

Phase 0 → Phase 1 → Phase 2 → Phase 3 + 4 (interleaved) → Phase 5.
Existing tests still pass; the new v2 tests exercise each phase.
