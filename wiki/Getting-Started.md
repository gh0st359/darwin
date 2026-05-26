# Getting Started

## Install

Requires Python 3.10 or newer. No GPU. No external services for the
core mind.

```bash
git clone https://github.com/gh0st359/darwin
cd darwin
git checkout v4
pip install -e .
```

Verify the install:

```bash
python -m unittest discover -s tests
```

## The 60-second tour

```bash
darwin run --steps 40 --seed 7
```

This runs Darwin in the simulated adaptive room for 40 steps and
prints the strongest causal beliefs and salient concepts at the end.
It is the simplest possible demonstration: action → consequence →
learning.

## The two-terminal life

This is how you actually use Darwin:

**Terminal 1: the 24/7 brain.**
```bash
darwin brain
```
You will see five background cognition loops fire continuously
(experiment, simulation, dream, self_modification, uncertainty) plus
the live thought stream when anyone chats. Leave this terminal open.

**Terminal 2: clean chat.**
```bash
darwin connect
```
You see `you>` for your input and `darwin>` for Darwin's response.
Nothing else. Background thinking stays in Terminal 1.

You can open as many `darwin connect` terminals as you want — they all
share one mind and one persistent memory.

See [The Two-Terminal Workflow](Two-Terminal-Workflow.md) for details.

## Try Darwin v4 with a tiny corpus

```bash
cat > /tmp/darwin-force.txt <<'EOF'
== Force ==
Force is an interaction that changes motion.
Force causes acceleration.
Aliases: push, pull
EOF

darwin ingest-corpus \
  --source wikidump \
  --path /tmp/darwin-force.txt \
  --memory /tmp/darwin-v4.sqlite3
```

Start the v4 generative kernel:

```bash
darwin brain \
  --kernel v4 \
  --workers auto \
  --accelerator auto \
  --memory /tmp/darwin-v4.sqlite3
```

Then connect:

```bash
darwin connect
```

Useful v4 commands:

```text
/knowledge force
/hypotheses
/worlds
/mind
/research status
```

## Single-terminal life (the old way)

If you want one window that does everything:

```bash
darwin live
```

Background loops, thought stream, and chat REPL all collapse into one
terminal. Useful for quick experiments; noisier for real conversation.

## Persistence

By default the brain writes:
- `darwin_memory.sqlite3` — durable mind state
- `darwin_runtime_state.json` — runtime posture
- `training_logs/*.jsonl` — plan, background, metrics, training-pair logs

Kill the brain and restart from the same directory and Darwin wakes up
with the same internal posture.

Move the database somewhere stable:
```bash
darwin brain --memory ~/darwin/state.sqlite3
```

## Try the optional DLM (gemma-3-270m)

```bash
ollama pull gemma3:270m
darwin brain --kernel v4 --dlm gemma --dlm-backend ollama --dlm-model gemma3:270m
```

Darwin will route its renderings through gemma-3-270m. If the output
fails `FaithfulnessValidator` (parser leak, missing causal claim,
unsurfaced uncertainty, hallucinated number, forbidden phrase), the
runtime silently falls back to the deterministic composer. You can
inspect what happened with `/dlm` from any chat client.

## What to do next

- Read [Philosophy and Non-Negotiables](Philosophy.md).
- Read [Architecture Overview](Architecture.md).
- Read [V4 Generative Universe Kernel](V4-Generative-Universe-Kernel.md).
- Browse the [CLI Reference](CLI-Reference.md).
- Look at `tests/test_v4_generative_universe.py` for executable v4 examples.
