# Training Data Strategy

This page describes how Darwin collects training data for the DLM —
and, just as importantly, what it refuses to do.

## What we do NOT do

We do **not** generate Darwin's thinking, concepts, beliefs, or
causal rules from a larger model and then train a smaller model to
imitate them. That is contamination of the very thing Darwin is
designed to grow without contamination.

We do **not** run multi-hour synthetic-data passes where a giant
model "teaches Darwin how to think." Darwin's thinking comes from its
own causal model, its own concept index, its own self-modification
loop. Always.

## What we do

The DLM is a renderer. Its job is to take a structured `ResponsePlan`
that Darwin's symbolic mind has *already produced* and turn it into
fluent English. We have a deterministic renderer (the composer) that
is faithful by construction. To get a small neural renderer
(gemma-3-270m) to be *as fluent* without being *unfaithful*, we
fine-tune it on `(plan_payload, rendering)` pairs where:

- the `plan_payload` is what Darwin's mind actually produced
- the `rendering` is faithful (validated by `FaithfulnessValidator`)
  and fluent (initially from the composer; eventually edited or
  polished)

## The collector

`TrainingDataCollector` (`darwin/training_data.py`) silently logs a
`TrainingPair` for every chat turn (background renderings aren't
collected by default):

```python
TrainingPair(
    plan_id,            # UUID, joins back to plan log
    user_text,          # the original user message
    plan_payload,       # plan.to_dlm_payload(): the canonical input
    rendering,          # the text Darwin actually said
    renderer,           # "composer" | "gemma-3-270m" | ...
    critique_passed,    # did the critic accept it?
    quality,            # 0.8 if critique passed, else 0.3 (tunable)
    accepted,           # quality threshold for training-set inclusion
    timestamp,
)
```

The default path is `training_logs/dlm_training_pairs.jsonl`. One
JSON line per pair. Append-only. No mutation. Inspect at any time:

```bash
darwin connect
you> /training
training pairs collected=137 accepted=121 path=training_logs/dlm_training_pairs.jsonl
- composer: 137
```

## Export

```bash
darwin export-training \
  --source training_logs/dlm_training_pairs.jsonl \
  --destination training_logs/dlm_training_export.jsonl \
  --min-quality 0.7 \
  --renderer composer
```

The export is the canonical input for the LoRA fine-tune of
gemma-3-270m. Each line is a self-contained example with the full
structured plan and the target rendering.

## The intended pipeline

1. **Collect.** Run `darwin brain` + `darwin connect` for as long as
   you can. The longer Darwin lives, the more diverse the plans get
   and the better the eventual fine-tune.
2. **(Optional) Curate.** Hand-edit a small slice of the export to
   produce a fluency target you actually love. Replace the
   `rendering` field in those records; keep the `plan_payload`
   identical. This is the cheapest, highest-leverage step.
3. **(Optional, exactly once) Polish.** Run a single, filtered pass
   through a larger model: for each `(plan_payload, composer_rendering)`
   pair, ask the larger model to rewrite the composer rendering more
   naturally **without changing meaning**. Pipe every output through
   `FaithfulnessValidator`. Throw away rejections. Use the kept ones
   as additional training examples.
   - This is the only place a larger model is allowed.
   - The pass is one-shot. After it runs, the larger model is never
     used again in the pipeline.
   - The validator is the same one used at runtime — so by
     construction, polish-pass examples are no more permissive than
     what the runtime would accept.
4. **Fine-tune.** Train a LoRA adapter for gemma-3-270m on the
   resulting JSONL. Target: faithful renderer of structured plans.
5. **Swap in.** `darwin brain --dlm gemma --dlm-backend ollama`
   using the fine-tuned model. `FaithfulnessValidator` still guards
   every output at runtime, so even regressions in the fine-tune
   are safe.
6. **Periodically retrain.** As Darwin's mind grows richer, its
   plans grow richer; retraining the renderer keeps it in step.

## Audit trail

Every (plan, rendering) pair is durable and replayable. You can:

- regenerate any rendering with a different renderer just by
  re-feeding `plan_payload`
- replay an entire conversation from `training_logs/plans.jsonl`
- diff renderer outputs side by side for the same plan

This is what makes the strategy honest: at every step you can look
at the input and the output and confirm that nothing came from
outside Darwin.
