# Instrumentation and Logs

Darwin is built to be observable. Every plan, every background
cognition wake-up, every metric is appended to a JSONL file so the
mind's behavior over time is fully replayable and auditable.

## `StructuredLogger`

```python
StructuredLogger(
    plan_log       = "training_logs/plans.jsonl",
    background_log = "training_logs/background.jsonl",
    metrics_log    = "training_logs/metrics.jsonl",
)
```

Thread-safe append-only writes. Each file is a stream of JSON objects,
one per line.

### `plans.jsonl`

One record per chat turn:

```json
{
  "plan_id": "9ab1...",
  "user_text": "What do you believe about open_curtains?",
  "semantic_summary": "source=user act=question topic=causality ...",
  "plan": { ...ResponsePlan.to_record()... },
  "rendering": "The beliefs I can defend are ...",
  "critique": {"passed": true, "issues": [], "revisions": []},
  "trace": { ...ThoughtTrace.to_record()... },
  "renderer": "composer",
  "background": false,
  "timestamp": 1758...
}
```

This file is the input corpus for DLM fine-tuning. Every record is a
self-contained `(structured plan, faithful rendering)` example. See
[Training Data Strategy](Training-Data-Strategy.md).

### `background.jsonl`

One record per background-loop wake-up:

```json
{
  "loop": "simulation",
  "kind": "simulation",
  "content": "Mental simulation: step 1: open_curtains ...",
  "payload": {"chain": {...}},
  "duration_ms": 1.2,
  "timestamp": 1758...
}
```

This is the high-resolution record of what Darwin's 24/7 mind was
doing while you weren't talking to it.

### `metrics.jsonl`

One record per named metric event:

```json
{"name": "plans_logged", "value": 137.0, "payload": {}, "timestamp": ...}
```

Plus an in-memory snapshot via `/metrics`:

```text
metrics:
- plans_logged: 137.0
- background_events: 814.0
counters:
- loop:experiment:       312
- loop:simulation:       210
- loop:dream:             80
- loop:self_modification: 50
- loop:uncertainty:      162
```

## `ThoughtTrace`

`ThoughtTrace` (`darwin/thought.py`) is the live, inspectable trace
of one cognitive cycle. Each step has:

```python
ThoughtStep(label, content, confidence, evidence, payload, started_at)
```

The trace itself carries a UUID `trace_id`, the original user text,
and a duration. Every chat turn produces one trace; it is dumped via
`/thoughts` and persisted via `PersistentStore.record_thought(...)`.

Typical labels: `parse`, `retrieve`, `plan`, `dlm`, `dlm_fallback`,
`critic`.

## `RuntimeEvent`

```python
RuntimeEvent(kind, content, payload, loop, timestamp)
```

The unit of streaming output. The runtime keeps the last 500 events
in memory (`runtime.events`) and broadcasts them to subscribed clients
via the daemon. Kinds: `experiment`, `simulation`, `dream`,
`self_modification`, `uncertainty`, `reflection`, `thought`,
`runtime`, `error`, `chat`.

## Backpressure

The daemon's per-subscriber outbound queue is bounded (size 512). If
a client is slower than the brain's event rate, the daemon drops the
**oldest** queued message to make room rather than blocking the
brain. Background cognition is therefore never stalled by a slow
chat client.

## Disabling logs

`StructuredLogger(..., enabled=False)` disables on-disk writes
without changing the API; in-memory metrics still accumulate. Useful
for tests and for ephemeral exploration.

## Reading the logs back

```python
from darwin.instrumentation import StructuredLogger

logger = StructuredLogger()
recent_plans = logger.read_plan_entries(limit=20)
for entry in recent_plans:
    print(entry["plan_id"], entry["renderer"], entry["critique"]["passed"])
```

Or just use jq:

```bash
jq -c 'select(.critique.passed == false) | {plan_id, rendering}' \
    training_logs/plans.jsonl | head
```

## Why all this is important

If you cannot inspect a mind, you cannot trust it. Darwin's design
choice is to make everything inspectable by default, leave durable
traces of every decision, and refuse to ship anything whose behavior
you cannot replay from disk.
