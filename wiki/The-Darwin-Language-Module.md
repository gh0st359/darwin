# The Darwin Language Module (DLM)

The DLM is the *only* place in Darwin where a neural language model is
allowed to operate, and even there it is strictly constrained to
rendering. It receives a structured `ResponsePlan` from Darwin's
symbolic mind and emits English prose. It cannot reason. It cannot
add facts. It cannot reorder causal logic. Its output is audited.

## Protocol

```python
class DarwinLanguageModule(Protocol):
    name: str
    def render(
        self,
        plan: ResponsePlan,
        frame: SemanticFrame,
        trace: ThoughtTrace,
    ) -> DLMRenderResult:
        ...

@dataclass
class DLMRenderResult:
    text: str                 # the prose Darwin will say
    renderer: str             # "composer" | "gemma-3-270m" | ...
    valid: bool               # FaithfulnessValidator verdict
    validation_notes: list[str]
    raw_output: str           # the renderer's unmodified output
    duration_ms: float
```

Two implementations ship with v2:

| Name | Class | Backend |
| --- | --- | --- |
| `stub` | `StubDLM` | the deterministic `NaturalLanguageComposer` |
| `gemma-3-270m` | `GemmaDLM` | Ollama HTTP, llama-cpp-python, or HF `transformers` |

The runtime always treats the DLM as a Protocol-shaped first-class
module, even when the underlying backend is the deterministic composer.

## StubDLM

Default. Wraps the composer; always passes validation; renders
instantly. Useful as a baseline and as the silent fallback when the
neural DLM is rejected.

## GemmaDLM

```bash
ollama pull gemma3:270m
darwin brain --dlm gemma --dlm-backend ollama --dlm-model gemma3:270m
```

The renderer is selected by `DARWIN_DLM_BACKEND` (or `--dlm-backend`)
and `DARWIN_DLM_MODEL` (or `--dlm-model`). Backends in priority order:

1. **Ollama** (`backend="ollama"`): POSTs to
   `$OLLAMA_HOST/api/chat` (default `http://127.0.0.1:11434`).
2. **llama-cpp** (`backend="llama-cpp"`): requires
   `llama-cpp-python` installed and `DARWIN_DLM_GGUF` pointing at a
   GGUF file.
3. **transformers** (`backend="transformers"`): requires the HF
   `transformers` library installed; uses
   `DARWIN_DLM_HF_MODEL` (default `google/gemma-3-270m-it`).

## The system prompt

Every call to `GemmaDLM` is prefixed with a fixed system prompt that
encodes the non-negotiables:

> You are the Darwin Language Module (DLM). You are a thin renderer,
> not a thinker. You receive a JSON object describing what Darwin's
> symbolic mind has decided to say, and you must render it as natural,
> fluent English. You MUST NOT add facts, claims, examples, knowledge,
> opinions, or reasoning that are not present in the JSON. You MUST
> preserve every causal_claim verbatim in meaning, every
> uncertainty_level explicitly, and you MUST never contradict the
> thesis. Do not introduce metaphors, analogies, or comparisons to
> external concepts. Do not say 'I think' unless the JSON has tone
> 'tentative'. Keep the response to the requested target_length. Do
> not output JSON, code, or parser notation. Output ONLY the rendered
> prose.

The user message is:

```
Render the following Darwin plan as natural English. Preserve all
causal_claims and uncertainty_levels. Do not invent any external
facts. Output prose only.

PLAN:
{ json.dumps(plan.to_dlm_payload(), indent=2) }
```

`temperature=0.4`, `num_predict=max_tokens=512` (configurable).

## The validation pass

Every renderer's output is fed to `FaithfulnessValidator.validate(plan,
text)`. See [Faithfulness Validation](Faithfulness-Validation.md).
On `valid=False` the runtime silently swaps in the composer's output
and tags the trace with a `dlm_fallback` step.

After the renderer + validator agree on a candidate, the existing
`ResponseCritic` re-checks the result against the plan from Darwin's
side (uncertainty disclosure, presence of high-confidence claims, no
parser leaks, no overconfidence, no thin replies).

## What the DLM never does

- It never decides whether Darwin should answer or clarify (the
  discourse planner did that).
- It never picks which causal claim to mention (the plan did that).
- It never adjusts confidence numbers (the model did that).
- It never invents references (`referenced_experiences` is the
  closed list).
- It never narrates Darwin's reasoning in its own voice.

## What you can swap

`DarwinRuntime.use_dlm(another_dlm)` swaps the renderer at runtime.
Any class that implements the `DarwinLanguageModule` Protocol is a
valid drop-in. This is how, for example, you could:

- A/B-test a fine-tuned gemma against the stub
- Plug in a different small renderer entirely
- Disable language output completely (return empty text and let a
  robotics policy consume `plan.to_dlm_payload()` directly)

## Observability

`/dlm` in the chat REPL shows the last render's renderer name,
validity, validation notes, and duration. `training_logs/plans.jsonl`
records the renderer used for every chat turn alongside the full
plan and the resulting text.
