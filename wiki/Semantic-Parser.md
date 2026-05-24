# Semantic Parser

`SemanticParser` (`darwin/semantics.py`) is the front door for
language. It turns raw text into a structured `SemanticFrame` that
the rest of Darwin can ground in symbols. It is a deterministic rule
+ pattern parser, not a neural model.

## A frame

```python
SemanticFrame(
    source: "user" | "darwin",
    original_text: str,
    normalized_text: str,
    tokens: list[str],
    speech_act: "question" | "goal" | "claim" | "teaching" | ...,
    topic: "language" | "self" | "causality" | ...,
    intent: ...,
    confidence: float,
    uncertainty: float,
    groundings: list[Grounding],       # action / variable / concept
    propositions: list[SemanticProposition],  # subject-rel-object
    goals: dict[str, Any],
    instructions: list[str],
    questions: list[str],
    corrections: list[str],
    values: dict[str, float],
    hypotheses: list[SemanticProposition],
    unknown_terms: list[str],
)
```

The parser:

1. Normalizes contractions and whitespace.
2. Tokenizes into word units.
3. Grounds tokens against:
   - known action aliases (`open curtains`, `flip switch`, …)
   - known variables and their aliases (`brightness`, `fuse`, …)
   - known concepts from the concept index
   - a small high-recall keyword set for vision/identity (`darwin`,
     `agi`, `consciousness`, …)
4. Extracts propositions from:
   - `if X then Y` → hypothesis (`X implies Y`)
   - `X causes Y`, `X leads to Y` → hypothesis
   - `X because Y` → explanation
   - `X prevents Y` → hypothesis
   - `X means Y`, `X is Y`, `X are Y` → definition / claim
5. Extracts goals from cue phrases (`i want`, `should`, `must`, …)
   combined with topic words (`bright`, `dark`, `fuse intact`, …)
   and grounded variables.
6. Extracts instructions, questions, corrections, values
   (`importance`, `preference`, `rejection`, `trust`, `autonomy`).
7. Computes a `speech_act` from cues + structural features.
8. Computes `topic` by token overlap with topic keyword sets.
9. Reports `unknown_terms` (≥5 chars, not stopwords, not grounded) —
   these become learning targets.
10. Scores `confidence` from the structural richness of the parse.

## SemanticMemory

`SemanticMemory` aggregates frames:

- bounded list of recent frames
- counter of propositions
- current set of active goals
- counter of values
- counter of unknown terms
- frames bucketed by topic and by speech_act

It is what `/status` summarizes and what the retriever scores against
when answering language queries.

## Why a deterministic parser

- It can be inspected. Every rule is a regex or a table.
- It cannot hallucinate. If it does not find a proposition, it does
  not invent one.
- It is fast. Parsing is microseconds, not milliseconds.
- It surfaces what it does **not** understand — `unknown_terms`
  become learning targets in `learning_priority` rather than silent
  failures.
- Its confidence is calibrated to structure, not to fluency. A
  fluent-sounding sentence with no extractable structure gets a low
  confidence; a terse sentence with a clear cause-effect relation
  gets a high one.

This is intentionally not the place to use an LLM. The DLM exists
downstream to render *Darwin's already-formed structured thought* as
prose; the parser exists upstream to ground *user language* into
Darwin's internal symbols. Putting an LLM in either place would
either contaminate the mind or undermine its calibration.

## What you see from a parse

`/semantics` (live mode) prints the most recent frames:

```text
- source=user act=question topic=causality confidence=0.61
    groundings=action:open_curtains, variable:room_bright
    propositions=open_curtains causes room_bright
- source=user act=teaching topic=planning confidence=0.55
    goals={'room_bright': True}
    values={'importance': 0.55}
```

Inside the cognitive cycle, the frame is then fed to retrieval,
discourse planning, and the response critic — each of which consults
`frame.speech_act`, `frame.confidence`, `frame.groundings`,
`frame.unknown_terms`, and the proposition list.
