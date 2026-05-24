# Discourse Planning and ResponsePlan

`DiscoursePlanner` (`darwin/discourse.py`) decides **what** Darwin
should communicate — before any word is chosen. Its output is a
`ResponsePlan`, the strictly-shaped data structure that drives every
downstream renderer (composer, DLM, critic).

## The plan

```python
@dataclass
class ResponsePlan:
    mode: str                     # "clarify" | "answer" | "belief_answer"
                                  # | "learn" | "experiment" | "self_report"
                                  # | "memory_summary" | "unknown_terms"
                                  # | "conversation"
    intent: str                   # one-line description of the goal
    thesis: str                   # the single sentence the plan stands on
    answer_points: list[str]      # ordered list of body points
    evidence: list[str]           # supporting summaries
    uncertainties: list[str]      # free-text uncertainty notes
    clarification_questions: list[str]
    next_actions: list[str]
    retrieved_used: list[RetrievedMemory]
    confidence: float
    should_answer_directly: bool

    # v2 structured fields:
    causal_claims: list[CausalClaim]                # action, variable, effect, condition, conf, n
    referenced_experiences: list[ReferencedExperience]  # kind, title, summary, score
    uncertainty_levels: list[UncertaintyLevel]      # target, level, reason
    self_reflection: list[str]                      # current learning posture
    plan_id: str                                    # UUID, propagates through logs
    tone: "confident" | "neutral" | "tentative"
    target_length: "short" | "medium" | "long"
```

`ResponsePlan.to_dlm_payload()` exposes a strictly-shaped view that the
DLM is allowed to render — no Darwin internals leak through; every
claim, uncertainty, and reference is explicit so the renderer can be
validated against it.

## Mode selection

`DiscoursePlanner.plan(...)` inspects `frame.speech_act`,
`frame.confidence`, focus terms, and the retrieval packet:

| User signal | Selected mode |
| --- | --- |
| weak parse + nothing retrieved | `clarify` |
| `speech_act == "teaching" / "goal" / "hypothesis" / "correction"` | `learn` |
| `speech_act == "question"` + focus on thinking/mind | `self_report` |
| `speech_act == "question"` + focus on beliefs/know | `belief_answer` |
| `speech_act == "question"` + focus on experiments/uncertainty | `experiment` |
| `speech_act == "question"` + focus on goals/values | `memory_summary` |
| `speech_act == "question"` + focus on unknown terms | `unknown_terms` |
| `speech_act == "question"` + strong retrieval | `answer` |
| otherwise | `conversation` |

## Enrichment (`_enrich_plan`)

After mode-specific construction, every plan is enriched with:

- the top 5 `causal_claims` from `darwin.causal_model.beliefs(...)`
- `referenced_experiences` from `plan.retrieved_used`
- `uncertainty_levels` for interpretation, answer, and any low-confidence
  causal claim
- `self_reflection` lines (learning priority, observation count,
  strongest belief)
- `tone` from `plan.confidence`
- `target_length` from the number of answer points

This is why downstream code can rely on the structured fields being
populated regardless of which mode was picked.

## Why the structured fields matter

Without them, the DLM is just generating prose loosely conditioned on
the plan's free-text strings. With them:

- `FaithfulnessValidator` can reject any rendering that drops a
  high-confidence causal claim or fails to surface a high-level
  uncertainty.
- `ResponseCritic` can enforce the same constraints structurally
  rather than via heuristics over the surface text.
- `TrainingDataCollector` can store `(plan_payload, rendering)` pairs
  with full structure on the input side, which is what makes the DLM
  fine-tunable in a way that preserves faithfulness.
- An external integration (robotics control, downstream policy) can
  consume the plan directly without ever calling a language renderer.

## The flow

```
SemanticFrame                                                ResponsePlan
     │                                                            │
     ▼                                                            ▼
ContextRetriever ─► RetrievalPacket ─► DiscoursePlanner ─► _enrich_plan ─► DLM
                                                                 ▲
                                                                 │
                                                       CausalModel.beliefs
                                                       SelfReport
```

## Inspecting a plan

Every chat turn persists the plan via
`PersistentStore.record_thought("response_plan", plan.thesis,
plan.to_record())`. You can dump the most recent via `/thoughts`,
`/retrieved`, and `/critic`. The full structured payload is also in
`training_logs/plans.jsonl` (one JSON line per turn), keyed by
`plan_id` — see [Instrumentation and Logs](Instrumentation.md).
