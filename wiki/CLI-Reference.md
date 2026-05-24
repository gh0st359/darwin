# CLI Reference

## Top-level commands

```bash
darwin run             # one-shot causal-learning run in the adaptive room
darwin live            # single-terminal mind + chat
darwin brain           # 24/7 daemon
darwin connect         # chat client attached to a running brain
darwin export-training # export DLM (plan -> rendering) pairs
```

### `darwin run`

Headless run of Darwin in the adaptive room simulation.

```bash
darwin run --steps 40 --seed 7 --exploration 0.25
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--steps` | 40 | number of step cycles |
| `--seed` | 7 | RNG seed |
| `--exploration` | 0.25 | exploration probability |

Prints the strongest causal beliefs and salient concepts at the end.

### `darwin live`

Single-process always-on mind + REPL.

```bash
darwin live --memory ~/state.sqlite3 --interval 2.0
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--memory` | `darwin_memory.sqlite3` | SQLite path |
| `--interval` | 3.0 | base background interval |
| `--exploration` | 0.20 | exploration probability |
| `--no-background` | off | run without background loops (request-response) |
| `--no-stream` | off | suppress live event printing |
| `--no-text-stream` | off | suppress incremental response printing |
| `--text-delay` | 0.012 | seconds per word for response streaming |
| `--dlm` | `stub` | `stub` (composer) or `gemma` (gemma-3-270m) |
| `--dlm-backend` | `ollama` | `ollama` / `llama-cpp` / `transformers` |
| `--dlm-model` | `gemma3:270m` | model identifier |

### `darwin brain`

24/7 daemon. Same flags as `darwin live` plus:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | bind interface |
| `--port` | 9870 | TCP port |
| `--quiet` | off | do not print events locally (chat clients still receive them when subscribed) |

### `darwin connect`

Clean chat client.

```bash
darwin connect --host 127.0.0.1 --port 9870
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | brain host |
| `--port` | 9870 | brain port |
| `--watch-events` | off | subscribe to the brain's background event stream and mirror it into this window |
| `--text-delay` | 0.0 | per-word delay for printed responses (0 = instant) |

### `darwin export-training`

Export accepted `(plan_payload -> rendering)` pairs from the training
log for DLM fine-tuning.

```bash
darwin export-training \
  --source training_logs/dlm_training_pairs.jsonl \
  --destination training_logs/dlm_training_export.jsonl \
  --min-quality 0.7
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source` | `training_logs/dlm_training_pairs.jsonl` | input JSONL |
| `--destination` | `training_logs/dlm_training_export.jsonl` | output JSONL |
| `--min-quality` | 0.7 | minimum quality score |
| `--renderer` | None | filter by renderer (`composer`, `gemma-3-270m`, …) |

## Chat-window commands

These work in both `darwin live` and `darwin connect`.

### Inspection

| Command | What it shows |
| --- | --- |
| `/status` | self-report (observations, known actions, learning priority, competence) + world model summary + semantic-memory summary + storage counts |
| `/beliefs` | strongest causal beliefs with confidence and sample counts |
| `/concepts` | concept hierarchy (state, effect, affordance, strategy, meta, cluster) |
| `/semantics` | recent parsed semantic frames (live only) |
| `/causal-graph` | distilled action → variable graph |
| `/experiments` | active experiment proposals from the current state |
| `/uncertainty` | latest per-action uncertainty scan |
| `/loops` | background loop status (intervals, last event per loop) |
| `/thoughts` | last internal thought trace |
| `/reason` | compact reasoning summary (live only) |
| `/retrieved` | memories used for the last response |
| `/critic` | self-critique of the last response |
| `/trace` | recent runtime events |
| `/dlm` | current DLM and last render validation notes |
| `/training` | DLM training-data corpus summary |
| `/metrics` | structured-logger metrics snapshot |

### Action

| Command | What it does |
| --- | --- |
| `/think` | run one cognition cycle now |
| `/dream` | consolidate memory now (also runs `ConceptIndex.consolidate()`) |
| `/simulate` | run one mental simulation now (multi-step causal chain) |
| `/selfmod` | propose and test self-modifications now |
| `/run N` | run N cognition cycles (live only) |
| `/plan` | show the current multi-step plan (live only) |
| `/stream` | inspect or change thought/text streaming (live only) |

### Session

| Command | What it does |
| --- | --- |
| `/help` | command list |
| `/exit` or `/quit` | disconnect (brain keeps running in `connect`; exits in `live`) |
| `/shutdown-brain` | stop the brain daemon (connect only) |
