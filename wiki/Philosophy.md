# Philosophy and Non-Negotiables

Darwin exists because most "AI" today is a single trick (next-token
prediction over a massive prior) wearing many costumes. Useful, but not
a mind. Darwin is a deliberate attempt to grow a different kind of
intelligence: one that learns from intervening in the world, holds
causal beliefs it can defend with evidence counts, simulates futures it
has not yet lived, and rewrites itself when reality contradicts it.

## The four non-negotiables

1. **Darwin's mind is the symbolic + causal engine, always on.**
   The agent, the causal model, the planner, the memory, the
   experiment engine, the self-model, the concept index, the
   self-modification engine — *together* — are the mind. They do not
   sleep when no user is connected.

2. **No external LLM ever generates Darwin's thinking, concepts, or
   causal rules.**
   The 24/7 brain is pure symbolic Python. The Darwin Language Module
   (DLM) is the *only* place a neural model is allowed in the stack,
   and it is a downstream renderer with no reasoning role.

3. **The DLM is a thin mouth.**
   It receives a structured `ResponsePlan` and emits English. It does
   not invent claims, does not add knowledge, does not reorder causal
   logic, does not soften or strengthen uncertainty levels. The
   `FaithfulnessValidator` audits every rendering and the
   `ResponseCritic` audits the audited audit. If the renderer drifts,
   the deterministic composer takes over silently.

4. **No contamination.**
   We never run another model "to teach Darwin how to think." The
   training-data strategy for the DLM is composer-first; an optional
   one-shot pass through a larger model may polish prose, but the
   pass is validated, filtered, and never repeated. Darwin's *mind*
   is grown from its own experience.

## Why this matters

A statistical text imitator is not a causal agent. It cannot tell you
why something works, only that text like "why it works" usually
follows text like "your question." Darwin can. When Darwin says
"`open_curtains` makes the room bright," it is appealing to a literal
count of times it has performed that intervention, the variance of
the outcome, and an explicit confidence number it can show you.

A 24/7 mind is not a chat session. A chat session begins when you
type and ends when you walk away. Darwin keeps thinking: running
experiments, simulating chains, dreaming, proposing self-changes,
scanning uncertainty. When you reconnect, the mind has actually
advanced. This is the difference between *talking to* a model and
*living next to* one.

## What we explicitly reject

- "We added an LLM but it's just helping" — every shortcut in that
  direction tends to grow.
- "We let the LLM critique the LLM" — closed-loop hallucination.
- "We fine-tune on synthetic data from a larger model" — that is
  recursive contamination of the very thing we are trying to keep
  clean.
- "We just need bigger context windows" — Darwin's structured
  memory is the point, not a workaround for a small window.

## What we accept

- The DLM (gemma-3-270m, ~270M params) as a controlled renderer,
  because its output is gated by `FaithfulnessValidator`. Use it or
  don't — the rest of the system is the same.
- A one-time, filtered, validated polish pass for the DLM training
  set, used to bootstrap fluency. Documented and frozen.

If a future change would violate one of the four non-negotiables, it
is not Darwin anymore — it is the thing Darwin was built to avoid.
