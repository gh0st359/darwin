# Faithfulness Validation

`FaithfulnessValidator` (`darwin/dlm.py`) is the audit layer that
sits between any renderer and the user. It is what makes the DLM
safe to use. Every candidate rendering is tested against the
structured `ResponsePlan` it was supposed to render, and any
deviation is grounds for rejection.

## What it checks

| Check | Failure mode it catches |
| --- | --- |
| Parser notation markers (`act=`, `topic=`, `intent=`, `source=`, `confidence=`, `groundings=`, `propositions=`, `score=`, `semantic:`, triple backticks) | rendering leaks internal symbols instead of natural language |
| Forbidden phrases (`as an ai`, `language model`, `i don't have access`, `i was trained`, `openai`, `according to wikipedia`, `according to my training`) | rendering reveals it is a downstream LLM or appeals to external authority |
| Empty output | rendering failed silently |
| High-confidence causal claims missing | rendering ignores the model's strongest beliefs in `belief_answer` / `answer` modes |
| High-impact uncertainty levels not surfaced | rendering claims certainty Darwin does not have |
| Clarification questions missing the `?` | rendering drops the only purpose of clarify mode |
| Output length outside `target_length` | rendering ignored the requested length |
| Hallucinated implausible numbers (>1000) not present in the plan | rendering invented quantitative claims |

A claim is "high-confidence" if `confidence ≥ 0.7`. An uncertainty
level is "high-impact" if `level ≥ 0.5`. These thresholds are easy
to tune in code; they were chosen to be permissive enough that the
deterministic composer always passes while still catching obvious
hallucinations from a neural renderer.

## What it does not check

- Grammar / fluency — that is the renderer's job; the validator
  cares about faithfulness, not style.
- Tone — also the renderer's job, conditioned by `plan.tone`.
- Word-by-word match — the validator looks for the *presence* of
  required claim actions/variables, not literal substrings of the
  plan.

## Result

```python
valid, notes = validator.validate(plan, candidate_text)
```

`valid` is `False` if any check produced a note. `notes` is a list
of human-readable explanations. The runtime stores both on
`DLMRenderResult` and surfaces them via `/dlm`.

## How rejection plays out

```python
render = self.dlm.render(plan, frame, trace)
draft  = render.text

if not render.valid:
    draft = self.composer.compose(plan, frame, trace)
    trace.add("dlm_fallback", "DLM output rejected; falling back to deterministic composer.",
              confidence=0.5, evidence=render.validation_notes)
```

The fallback is *silent to the user* and *visible to the operator*.
The user sees a fluent answer (composer output); the operator sees
the rejection notes via `/dlm`, `training_logs/plans.jsonl`, and the
`thought` event payload.

## Defense in depth

Even after the DLM passes validation, the existing `ResponseCritic`
re-checks the result. The critic enforces:

- no overconfident hedges (`certainly`, `definitely`, `obviously`)
  when `plan.confidence < 0.7`
- no responses under 8 words outside of `clarify` mode
- presence of high-confidence causal claims in answer modes
- disclosure of any uncertainty level ≥ 0.55
- no parser notation in the surface text

If the critic fails the candidate, the runtime calls
`critic.revise(plan, critique, ...)` to mutate the plan toward
something the critic will accept, then re-renders and re-validates.
This is the loop:

```
plan ─► DLM.render ─► validator? ──► critic? ─► reply
        │                  │            │
        │      [composer]  │  [revise]  │
        ▼                  ▼            ▼
       text          fallback text     mutated plan ─► re-render
```

## Why this is the correct seam

The neural renderer's strength is fluency. Its weakness is invention.
The validator concedes the fluency, denies the invention, and gives
the symbolic mind the final say. The result is a system where you can
benefit from a fluent small model **without** trusting that fluent
small model to tell you the truth.
