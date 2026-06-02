"""Darwin's Speech cortex — non-LLM compositional natural language.

The Speech cortex turns a structured ``ResponsePlan`` into fluent prose
without any pretrained language model. A five-stage pipeline
(ContentSelection → DiscoursePlan → SentencePlan → LexicalChoice →
SurfaceRealization) composes the surface; a LeakGate then enforces the
hard invariant that chat output never leaks structured internals (no
JSON, no curly braces, no operator-bracketed inference tags, no event-
stream markers).

The pipeline ships behind the existing DarwinLanguageModule Protocol so
it is a drop-in for ``StubDLM`` / ``GemmaDLM``. The default DarwinRuntime
uses it whenever ``DARWIN_USE_SPEECH != "0"``.
"""

from darwin.speech.ccg import (
    ADJ,
    CCGCategory,
    CCGSign,
    N,
    NP,
    PREP,
    S,
    backward_apply,
    combine,
    forward_apply,
    forward_compose,
    parse_category,
)
from darwin.speech.dlm_adapter import SpeechDLM
from darwin.speech.leak_gate import LeakGate, LeakGateResult
from darwin.speech.lexicon import CCGLexicon, LexicalEntry, default_lexicon_path
from darwin.speech.pipeline import SpeechPipeline, SpeechRenderResult


__all__ = [
    "ADJ",
    "CCGCategory",
    "CCGLexicon",
    "CCGSign",
    "LeakGate",
    "LeakGateResult",
    "LexicalEntry",
    "N",
    "NP",
    "PREP",
    "S",
    "SpeechDLM",
    "SpeechPipeline",
    "SpeechRenderResult",
    "backward_apply",
    "combine",
    "default_lexicon_path",
    "forward_apply",
    "forward_compose",
    "parse_category",
]
