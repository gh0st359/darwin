# Roadmap

This is the working roadmap, in priority order. Items here are
candidates, not commitments. The principle in [Philosophy](Philosophy.md)
gates every item — anything that would violate the non-negotiables
is not on this list.

## Near-term (v2.x)

### Better world(s)
- A second, more open-ended adapter beyond `AdaptiveRoomWorld`. A
  small grid world or a text-only puzzle environment with hidden
  rules to stress the causal model harder.
- A way to swap worlds at runtime so a single Darwin can learn across
  multiple bodies.

### Stronger memory consolidation
- The dream loop currently forms cluster concepts from co-occurring
  affordances. Next: learned analogies (effect:A:X ≈ effect:B:Y when
  before/after deltas match), automatic pruning of clusters that
  never fire, and reward-weighted reinforcement of clusters that
  correlate with positive outcomes.

### Smarter self-modification
- Larger proposal space: per-action exploration rates, per-variable
  conditional-effect specificity bonuses, retrieval re-weighting.
- Per-proposal holdout slicing so the same evaluation can compare
  multiple variants simultaneously without contaminating the next
  cycle.
- Persistent self-modification history that informs future proposals
  (don't re-propose what was rejected twice).

### Faithfulness coverage
- Validator extensions: per-claim *direction* enforcement (e.g.
  rendering says `False -> True` when plan says the same); detection
  of stale references (a `referenced_experience` that contradicts the
  current causal_claims).
- A small offline validator-correctness test set: known-bad
  renderings the validator must reject, known-good renderings it
  must accept.

## DLM track

### Curated rendering set
- Hand-edit ~200 (plan, rendering) pairs across the v2 modes
  (`belief_answer`, `clarify`, `learn`, `experiment`,
  `self_report`, `memory_summary`, `unknown_terms`, `conversation`)
  to define the fluency target.

### One-shot polish pass
- Documented, single-run script that takes the composer renderings,
  asks a larger model for a faithful rewrite each, runs every
  candidate through `FaithfulnessValidator`, and saves the kept
  outputs as additional training data. Runs once, output is frozen,
  larger model is never used again.

### LoRA fine-tune
- Train gemma-3-270m LoRA on the combined corpus (composer + curated +
  polished). Publish the LoRA adapter. Document the eval suite.
- Set up CI eval: % of renderings that pass `FaithfulnessValidator`
  on a held-out plan set. Refuse to ship a regression.

## Operational track

### Auth on the brain socket
- Optional shared-secret or local-cert auth for the daemon. Default
  remains loopback-only.

### Multi-host brain
- Brain on one machine, chat clients on others. Currently doable via
  SSH tunnel; the goal is first-class TLS + auth.

### Dashboard
- Read-only web view of the JSONL logs: plan timeline, background
  loop heatmap, self-modification log, causal graph snapshot.

## Bigger questions

These are the questions Darwin should help us answer, not assumptions
we want to bake in:

- **What does "more general" look like?** Adding more knobs is easy.
  Showing that Darwin transfers from one world to another without
  catastrophic forgetting is the real test.
- **Does the simulation loop need a separate "imagined memory"?**
  Right now the highest-uncertainty step of an imagined chain feeds
  the prediction-failure counter. Should imagined experiences also
  populate a parallel-but-isolated episodic store?
- **Is there a place for a tiny embedding model inside retrieval?**
  Retrieval is currently lexical + structural. A small,
  on-device embedding (not a generation model) could improve
  cross-vocabulary recall without violating the
  no-reasoning-from-an-LLM rule.

## v3 themes (speculative)

- Multi-embodiment Darwin: one mind, several bodies.
- Differentiable-on-the-edges causal model: keep the symbolic core,
  add small learned residuals for hard-to-rule-encode dynamics, with
  the same hold-out validation gate the self-modification engine
  uses.
- Long-horizon goal stacks: persistent multi-day plans the
  background loops chip away at.

## How to influence the roadmap

Open an issue with a concrete proposal that:

1. names the file(s) and the function(s) it would touch,
2. names the test(s) it would add or change,
3. argues why it does not violate the [non-negotiables](Philosophy.md).

PRs that pass the existing 44 tests and bring their own are how
v2.x grows.
