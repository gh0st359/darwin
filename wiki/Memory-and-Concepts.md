# Memory and Concepts

Darwin keeps three intertwined memory systems: episodic (raw
transitions), conceptual (hierarchical learned structures), and
semantic (parsed language meaning). All three feed retrieval.

## EpisodicMemory (`darwin/memory.py`)

A bounded deque of `Transition`s with two indices:

- `_by_action: {action_name -> [transition_indices]}`
- `_by_variable: {variable_name -> [transition_indices]}`

Indices give fast access patterns used throughout the system:

```python
episodes.by_action("open_curtains", limit=20)          # last 20 times that action ran
episodes.by_variable("room_bright", limit=20)          # last 20 transitions touching room_bright
episodes.changed_variable("fuse_intact", polarity="decrease")
episodes.positive_reward(limit=10, threshold=0.0)
episodes.temporal_distance(index)                      # 0..1 recency score
```

`temporal_distance` is used by the retriever to combine recency with
grounding overlap and reward.

When the deque exceeds capacity, the oldest transition is dropped and
the indices are rebuilt lazily.

## ConceptIndex (`darwin/concepts.py`)

A grounded hierarchy that Darwin builds without any pretrained
ontology. Five kinds of concept emerge from `Transition`s:

| Kind | Level | Example name | Meaning |
| --- | --- | --- | --- |
| `state`       | 0 | `state:room_bright=True` | a value of a variable that Darwin has seen |
| `effect`      | 1 | `effect:open_curtains:room_bright` | this action moved this variable |
| `affordance`  | 2 | `affordance:open_curtains:can_set:room_bright=True` | this action can produce that value |
| `strategy`    | 3 | `strategy:seek:room_bright=True:via:open_curtains` | reward-positive way to achieve a value |
| `meta`        | 4 | `meta:reliable_action:open_curtains` | learned reliability tag |

Each concept tracks `support` (times observed), `reward_total`, a set
of `parents`, and up to 5 example payloads. `salience` =
`support × (1 + |reward_mean|) × (1 + 0.15 × level)`.

### Consolidation (the dream loop)

`ConceptIndex.consolidate()` is called by `_loop_dream`. It clusters
affordances that share parents into a new level-5 `cluster:via:<parent>`
concept (so groups of related affordances become a higher-order unit),
and decays support of stale low-salience concepts. The dream loop
reports how many clusters were formed and how many concepts were
decayed.

This is the part of Darwin that is meant to evoke the
"memory consolidation during sleep" metaphor: it does not happen
during interaction; it happens on its own cadence in the background.

## SemanticMemory (`darwin/semantics.py`)

Stores parsed `SemanticFrame`s rather than raw text. Each frame has:

- `source` (`user` or `darwin`), `speech_act` (`question`, `goal`,
  `claim`, `teaching`, …), `topic`, `intent`, `confidence`
- groundings: action / variable / concept symbols matched in the text
- propositions: subject-relation-object triples (`means`, `is`,
  `causes`, `prevents`, `implies`, …)
- goals, instructions, questions, corrections, values
- hypotheses (propositions of kind `hypothesis`)
- unknown terms (>=5 chars, not stop-worded, not grounded)

The memory aggregates:
- a counter of propositions
- the current set of active goals
- a counter of values (importance, preference, trust, autonomy, …)
- a counter of unknown terms (which become learning targets)

`SemanticMemory.summary()` is what `/status` prints.

## Retrieval (`darwin/retrieval.py`)

The `ContextRetriever.retrieve(darwin, frame, recent_events, limit)`
returns a `RetrievalPacket` ranked across **all** memory systems:

| Source | Scoring |
| --- | --- |
| semantic frames (`semantic`) | term overlap × grounding overlap × topic match × speech-act match × recency × structural richness |
| concept index (`concept`) | term overlap with concept name + clamped salience + kind-overlap with frame groundings |
| causal beliefs (`causal_belief`) | term overlap with action/variable/condition + grounding overlap + confidence |
| recent runtime events (`runtime_event`) | term overlap (only when the frame is about self) |
| **episodic transitions (`episode`)** | action term overlap + grounding overlap + recency + clamped reward |
| **completed experiments (`experiment`)** | action grounding (0.55 if surprising, 0.45 if confirmed) |

Top items flow into `ResponsePlan.referenced_experiences`. The DLM
sees them and is required to ground its rendering in them when
relevant.

`ContextRetriever.retrieve_for_topic(darwin, topic, groundings)` is a
lightweight retrieval path used by background loops that don't have a
`SemanticFrame` to drive retrieval.

## What this means for conversation

When you ask Darwin about something, it doesn't pattern-match a
surface query against a giant document store. It:

1. parses the question into a structured semantic frame,
2. retrieves across episodic transitions, concept hierarchy, causal
   beliefs, semantic memory, and completed experiments,
3. ranks results by scored relevance,
4. only uses items that score above thresholds,
5. binds those items into `referenced_experiences` on the plan,
6. validates that the rendering actually references the chosen
   experiences.

If Darwin has no relevant memory it will say so explicitly, not
hallucinate.
