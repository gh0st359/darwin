from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from darwin.composer import NaturalLanguageComposer
from darwin.discourse import ResponsePlan
from darwin.semantics import SemanticFrame
from darwin.thought import ThoughtTrace


DLM_SYSTEM_PROMPT = (
    "You are Darwin's voice. You are given a short brief describing "
    "what Darwin has decided to say. Rephrase it as natural, "
    "conversational English in Darwin's first person. "
    "Rules:\n"
    "- Write 1-3 short sentences. Plain prose.\n"
    "- Do not invent facts, examples, or claims that are not in the brief.\n"
    "- Do not output JSON, code, lists, bullets, headings, or quoted keys.\n"
    "- Do not say 'as an AI', 'language model', 'I was trained', or similar.\n"
    "- Do not hedge unless the brief says you are uncertain.\n"
    "- If the brief is a short greeting or small-talk, answer briefly.\n"
    "- Output ONLY the sentences. Nothing else."
)


@dataclass
class DLMRenderResult:
    text: str
    renderer: str
    valid: bool
    validation_notes: list[str] = field(default_factory=list)
    raw_output: str = ""
    duration_ms: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "renderer": self.renderer,
            "valid": self.valid,
            "validation_notes": self.validation_notes,
            "raw_output": self.raw_output,
            "duration_ms": self.duration_ms,
        }


class DarwinLanguageModule(Protocol):
    name: str

    def render(
        self,
        plan: ResponsePlan,
        frame: SemanticFrame,
        trace: ThoughtTrace,
    ) -> DLMRenderResult:
        ...


class FaithfulnessValidator:
    """Validates a candidate rendering against the plan's structured content.

    The DLM is a renderer, not a reasoner. We reject any output that
    invents facts, drops required uncertainty levels, leaks parser
    notation, or contradicts the planned causal claims.
    """

    FORBIDDEN_PHRASES = (
        "as an ai",
        "language model",
        "i don't have access",
        "i was trained",
        "openai",
        "according to wikipedia",
        "according to my training",
    )
    NOTATION_MARKERS = (
        "act=",
        "topic=",
        "intent=",
        "source=",
        "confidence=",
        "groundings=",
        "propositions=",
        "score=",
        "semantic:",
        "```",
    )

    JSON_LEAK_KEYS = (
        '"thesis":',
        '"answer_points":',
        '"causal_claims":',
        '"uncertainty_levels":',
        '"referenced_experiences":',
        '"mode":',
        '"intent":',
        '"clarification_questions":',
        '"next_actions":',
        '"target_length":',
        '"plan_id":',
    )

    def validate(self, plan: ResponsePlan, text: str) -> tuple[bool, list[str]]:
        notes: list[str] = []
        stripped = text.strip()
        lowered = text.lower()

        for marker in self.NOTATION_MARKERS:
            if marker in text:
                notes.append(f"output leaked notation '{marker}'")

        for phrase in self.FORBIDDEN_PHRASES:
            if phrase in lowered:
                notes.append(f"output contained forbidden phrase '{phrase}'")

        if not stripped:
            notes.append("output was empty")

        # JSON / structured-plan leak detection.
        if stripped.startswith("{") or stripped.startswith("["):
            notes.append("output is a JSON object instead of prose")
        leak_hits = sum(1 for key in self.JSON_LEAK_KEYS if key in text)
        if leak_hits >= 2:
            notes.append("output regurgitated the structured plan as quoted fields")
        if "\n  -" in text or "\n- " in text or text.count("\n*") >= 2:
            notes.append("output is a bullet list instead of prose")
        if text.count("\n\n") >= 2 and stripped.startswith(("Situation:", "Goal", "Main point", "Things")):
            notes.append("output echoes the input brief rather than rendering it")

        for claim in plan.causal_claims[:3]:
            if claim.confidence >= 0.7 and claim.action not in lowered and claim.variable not in lowered:
                # Only enforce mention for high-confidence claims
                if plan.mode in {"belief_answer", "answer"}:
                    notes.append(
                        f"required causal claim {claim.action}->{claim.variable} missing from text"
                    )

        for level in plan.uncertainty_levels:
            if level.level >= 0.5 and not self._mentions_uncertainty(lowered):
                notes.append(
                    f"required uncertainty about {level.target} (level {level.level:.2f}) not surfaced"
                )
                break

        if plan.clarification_questions and "?" not in text:
            notes.append("clarification question missing from output")

        # Reject obviously hallucinated lists by checking for very long answers
        # when the plan asked for short ones.
        if plan.target_length == "short" and len(text.split()) > 60:
            notes.append("output is longer than requested target_length=short")
        if plan.target_length == "long" and len(text.split()) < 12:
            notes.append("output is shorter than requested target_length=long")

        # Sanity check: rendering should not introduce numbers not in plan.
        plan_numbers = self._numbers_from_plan(plan)
        text_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        invented = text_numbers - plan_numbers
        for value in invented:
            try:
                if float(value) > 1000:  # Soft check: only flag implausible numbers
                    notes.append(f"output introduced unsupported number '{value}'")
            except ValueError:
                continue

        return (not notes, notes)

    def _mentions_uncertainty(self, lowered: str) -> bool:
        markers = (
            "uncertain",
            "not sure",
            "low confidence",
            "tentatively",
            "tentative",
            "i may be wrong",
            "limited",
            "not yet",
            "still need",
            "weak",
            "thin",
        )
        return any(marker in lowered for marker in markers)

    def _numbers_from_plan(self, plan: ResponsePlan) -> set[str]:
        text_pool = " ".join(
            [
                plan.thesis,
                *plan.answer_points,
                *plan.evidence,
                *plan.uncertainties,
                *plan.clarification_questions,
                *[item.summary for item in plan.referenced_experiences],
                *[
                    f"{claim.confidence:.2f} {claim.samples}"
                    for claim in plan.causal_claims
                ],
                *[f"{level.level:.2f}" for level in plan.uncertainty_levels],
                f"{plan.confidence:.2f}",
            ]
        )
        return set(re.findall(r"\b\d+(?:\.\d+)?\b", text_pool))


class StubDLM:
    """The default 'DLM': a thin wrapper around the deterministic composer.

    This is what runs when no external gemma-3-270m is wired up. It always
    passes validation because it just defers to the deterministic composer
    and tags the result as renderer='composer'. Keeping it Protocol-shaped
    means the rest of Darwin treats the renderer as a true module.
    """

    name = "stub"

    def __init__(self) -> None:
        self.composer = NaturalLanguageComposer()
        self.validator = FaithfulnessValidator()

    def render(
        self,
        plan: ResponsePlan,
        frame: SemanticFrame,
        trace: ThoughtTrace,
    ) -> DLMRenderResult:
        started = time.perf_counter()
        text = self.composer.compose(plan, frame, trace)
        valid, notes = self.validator.validate(plan, text)
        return DLMRenderResult(
            text=text,
            renderer="composer",
            valid=valid,
            validation_notes=notes,
            raw_output=text,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


class GemmaDLM:
    """Render Darwin's structured plan via gemma-3-270m running locally.

    Backends, in priority order:
      1. Ollama HTTP API at ``OLLAMA_HOST`` (default http://127.0.0.1:11434)
      2. ``llama-cpp-python`` if installed and a GGUF path is configured
      3. Hugging Face ``transformers`` pipeline if available

    The DLM is *only* allowed to render. The system prompt forbids
    invention; the FaithfulnessValidator rejects any output that strays.
    On rejection or any backend failure we return ``valid=False``; the
    runtime then falls back to the deterministic composer.
    """

    name = "gemma-3-270m"

    def __init__(
        self,
        backend: str | None = None,
        model: str = "gemma3:270m",
        timeout: float = 30.0,
        max_tokens: int = 512,
    ) -> None:
        self.backend = backend or os.environ.get("DARWIN_DLM_BACKEND", "ollama")
        self.model = os.environ.get("DARWIN_DLM_MODEL", model)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.validator = FaithfulnessValidator()
        self.composer = NaturalLanguageComposer()
        self._lock = threading.RLock()
        self._llama = None
        self._transformer = None

    def render(
        self,
        plan: ResponsePlan,
        frame: SemanticFrame,
        trace: ThoughtTrace,
    ) -> DLMRenderResult:
        payload = plan.to_dlm_payload()
        started = time.perf_counter()
        raw_text: str = ""
        notes: list[str] = []
        try:
            raw_text = self._call_backend(payload)
        except Exception as exc:  # pragma: no cover - depends on env
            notes.append(f"backend error: {exc!r}")
            text = self.composer.compose(plan, frame, trace)
            return DLMRenderResult(
                text=text,
                renderer=self.name,
                valid=False,
                validation_notes=notes,
                raw_output="",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        cleaned = self._strip(raw_text)
        valid, validation_notes = self.validator.validate(plan, cleaned)
        notes.extend(validation_notes)

        if not valid:
            cleaned = self.composer.compose(plan, frame, trace)

        return DLMRenderResult(
            text=cleaned,
            renderer=self.name,
            valid=valid,
            validation_notes=notes,
            raw_output=raw_text,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _call_backend(self, payload: dict[str, Any]) -> str:
        if self.backend == "ollama":
            return self._call_ollama(payload)
        if self.backend == "llama-cpp":
            return self._call_llama_cpp(payload)
        if self.backend == "transformers":
            return self._call_transformers(payload)
        raise RuntimeError(f"Unknown DLM backend: {self.backend}")

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        # We do NOT hand gemma-3-270m the raw JSON — a 270M model just
        # echoes it. Instead we build a short human-readable brief
        # listing only the fields that matter for rendering.
        lines: list[str] = []
        mode = payload.get("mode", "")
        intent = payload.get("intent", "")
        if mode:
            lines.append(f"Situation: {mode}.")
        if intent:
            lines.append(f"Goal of the reply: {intent}.")

        thesis = payload.get("thesis", "").strip()
        if thesis:
            lines.append(f"Main point: {thesis}")

        points = [p for p in payload.get("answer_points", []) if p and p.strip()]
        if points:
            lines.append("Things to convey, in order:")
            for point in points[:4]:
                lines.append(f"  - {point.strip()}")

        causal = payload.get("causal_claims", []) or []
        strong_causal = [c for c in causal if float(c.get("confidence", 0.0)) >= 0.55]
        if strong_causal:
            lines.append("Facts you must respect (do not contradict):")
            for claim in strong_causal[:3]:
                cond = claim.get("condition", "always")
                cond_text = "" if cond == "always" else f" when {cond}"
                lines.append(
                    f"  - {claim['action'].replace('_',' ')} makes "
                    f"{claim['variable'].replace('_',' ')} "
                    f"{self._human_effect(claim['effect'])}"
                    f"{cond_text} (seen {claim['samples']} times)"
                )

        levels = payload.get("uncertainty_levels", []) or []
        strong_uncertainty = [u for u in levels if float(u.get("level", 0.0)) >= 0.55]
        if strong_uncertainty:
            lines.append("Uncertainty you must surface:")
            for level in strong_uncertainty[:2]:
                target = str(level.get("target", "")).replace("_", " ")
                reason = str(level.get("reason", ""))
                tail = f" because {reason}" if reason else ""
                lines.append(f"  - not very sure about {target}{tail}")

        clarifications = payload.get("clarification_questions", []) or []
        if clarifications:
            lines.append(f"End with this question: {clarifications[0]}")

        length = payload.get("target_length", "medium")
        if length == "short":
            length_note = "Keep it to one short sentence."
        elif length == "long":
            length_note = "Use up to three sentences."
        else:
            length_note = "Use one or two sentences."
        lines.append(length_note)

        tone = payload.get("tone", "neutral")
        if tone == "tentative":
            lines.append("Tone: tentative.")
        elif tone == "confident":
            lines.append("Tone: confident but not boastful.")
        else:
            lines.append("Tone: neutral and direct.")

        lines.append(
            "Now write the reply as Darwin, in plain English. "
            "Do not output JSON, lists, or anything other than the sentences."
        )
        return "\n".join(lines)

    def _human_effect(self, effect: str) -> str:
        cleaned = effect.strip()
        if cleaned == "False -> True":
            return "true"
        if cleaned == "True -> False":
            return "false"
        if cleaned.startswith("+="):
            try:
                delta = float(cleaned[2:].strip())
            except ValueError:
                return cleaned
            if delta < 0:
                return f"drop by {abs(delta):g}"
            if delta > 0:
                return f"rise by {delta:g}"
        return cleaned

    def _call_ollama(self, payload: dict[str, Any]) -> str:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        endpoint = host.rstrip("/") + "/api/chat"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": DLM_SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(payload)},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": self.max_tokens,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama unreachable: {exc!r}") from exc
        message = data.get("message", {})
        if isinstance(message, dict) and "content" in message:
            return str(message["content"])
        return str(data.get("response", ""))

    def _call_llama_cpp(self, payload: dict[str, Any]) -> str:
        with self._lock:
            if self._llama is None:
                from llama_cpp import Llama  # type: ignore

                model_path = os.environ.get("DARWIN_DLM_GGUF")
                if not model_path:
                    raise RuntimeError("DARWIN_DLM_GGUF env var is required for llama-cpp backend")
                self._llama = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=os.cpu_count() or 4,
                )
            response = self._llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": DLM_SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(payload)},
                ],
                temperature=0.4,
                max_tokens=self.max_tokens,
            )
        choice = response["choices"][0]
        return choice["message"]["content"]

    def _call_transformers(self, payload: dict[str, Any]) -> str:
        with self._lock:
            if self._transformer is None:
                from transformers import pipeline  # type: ignore

                self._transformer = pipeline(
                    "text-generation",
                    model=os.environ.get("DARWIN_DLM_HF_MODEL", "google/gemma-3-270m-it"),
                )
            prompt = f"<system>\n{DLM_SYSTEM_PROMPT}\n</system>\n<user>\n{self._build_prompt(payload)}\n</user>\n<assistant>\n"
            outputs = self._transformer(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=0.4,
                do_sample=True,
            )
        text = outputs[0]["generated_text"]
        if "<assistant>" in text:
            text = text.split("<assistant>", 1)[1]
        return text

    def _strip(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        # Collapse runaway whitespace.
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def gemma_dlm_available() -> bool:
    """Best-effort detection: do we have an Ollama or local model wired up?"""

    if shutil.which("ollama") is not None:
        return True
    try:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        request = urllib.request.Request(host.rstrip("/") + "/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False
