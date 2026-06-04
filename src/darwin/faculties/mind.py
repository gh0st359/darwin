"""Mind — the single cognition surface above the cognitive faculties.

Replaces the legacy ``AgentRegistry`` as the brain's dispatch surface.

Two responsibilities:

1. :meth:`consider` — read a message, decide what cognitive shape it
   demands by comparing its embedding to learned per-faculty exemplar
   centroids. No hardcoded subject regexes: the routing is over the
   *learned representation*, so as the corpus grows, the routing
   sharpens. Returns an :class:`Intent`. Internal-only — never rendered.

2. :meth:`solve` — recruit the relevant faculties **internally**, run
   them, and compose a single coherent natural-language reply in
   Darwin's voice. The faculty class name, the intent kind, and any
   internal step labels are stripped before the reply is returned.

This surface treats the agent classes (Coder, Calculator, Scientist,
Planner, Researcher, Conversationalist) as *internal tools*, not as
categorised personae. If you need a categorised dispatcher (autonomy /
task executor still does), call them by capability through
:meth:`recruit`, not by routing user-visible language through them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from darwin.agents.code_agent import CodeAgent, CodeProblem
from darwin.agents.dialogue_agent import DialogueAgent, DialogueProblem
from darwin.agents.math_agent import MathAgent, MathProblem
from darwin.agents.planning_agent import PlanningAgent, PlanningProblem
from darwin.agents.research_agent import ResearchAgent, ResearchProblem
from darwin.agents.science_agent import ScienceAgent, ScienceProblem
from darwin.mind.intent import Intent, IntentKind, MindReply


# Faculty exemplar tokens — concept terms that should drift toward each
# faculty's centroid in the learned embedding space. Routing reads
# similarity to these centroids, not hardcoded regexes.
_FACULTY_EXEMPLARS: dict[IntentKind, tuple[str, ...]] = {
    IntentKind.COMPUTE: ("calculate", "sum", "product", "equation", "number", "math"),
    IntentKind.DERIVE: ("code", "function", "program", "implement", "algorithm"),
    IntentKind.PLAN: ("plan", "steps", "schedule", "first", "then", "sequence"),
    IntentKind.RESEARCH: ("research", "find", "lookup", "study", "investigate"),
    IntentKind.RECALL: ("recall", "remember", "know", "fact", "knowledge"),
    IntentKind.SYNTHESIZE: ("explain", "why", "how", "relate", "connect"),
}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return s / (na * nb)


@dataclass
class Mind:
    """Single brain-level composition surface above the faculties."""

    runtime: Any = None
    coder: Any = None
    calculator: Any = None
    scientist: Any = None
    planner: Any = None
    researcher: Any = None
    conversationalist: Any = None
    # Confidence threshold below which Mind declines and lets the chat path
    # fall through. Calibrated against the learned exemplar centroids.
    intent_threshold: float = 0.18
    _centroid_cache: dict[IntentKind, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coder is None:
            self.coder = CodeAgent(self.runtime)
        if self.calculator is None:
            self.calculator = MathAgent(self.runtime)
        if self.scientist is None:
            self.scientist = ScienceAgent(self.runtime)
        if self.planner is None:
            self.planner = PlanningAgent(self.runtime)
        if self.researcher is None:
            self.researcher = ResearchAgent(self.runtime)
        if self.conversationalist is None:
            self.conversationalist = DialogueAgent(self.runtime)

    # -- back-compat surface for autonomy / executor ----------------------- #

    def recruit(self, kind: str) -> Any:
        """Return the faculty for ``kind`` (used by autonomy.task_executor).

        ``kind`` is internal: "code", "math", "science", "planning",
        "research", "dialogue". Never surfaces in chat.
        """

        match kind:
            case "code":
                return self.coder
            case "math":
                return self.calculator
            case "science":
                return self.scientist
            case "planning":
                return self.planner
            case "research":
                return self.researcher
            case "dialogue":
                return self.conversationalist
        return None

    # Legacy AgentRegistry surface kept for one phase so existing code paths
    # (autonomy.executor, scale.agent_subsystem_specs, bench tests still
    # importing AgentRegistry) keep working unchanged.
    @property
    def code(self) -> Any:
        return self.coder

    @property
    def math(self) -> Any:
        return self.calculator

    @property
    def science(self) -> Any:
        return self.scientist

    @property
    def planning(self) -> Any:
        return self.planner

    @property
    def research(self) -> Any:
        return self.researcher

    @property
    def dialogue(self) -> Any:
        return self.conversationalist

    def all(self) -> list[Any]:
        return [
            self.coder, self.calculator, self.scientist,
            self.planner, self.researcher, self.conversationalist,
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "faculties": [f.name for f in self.all()],
            "count": 6,
        }

    # -- intent classification --------------------------------------------- #

    def consider(self, message: str) -> Intent:
        """Decide the cognitive shape of ``message`` via learned centroids."""

        if not message or not message.strip():
            return Intent(kind=IntentKind.DECLINE, confidence=0.0)
        embedding_space = getattr(self.runtime, "embedding_space", None)
        message_emb = self._message_embedding(message, embedding_space)
        centroids = self._centroids(embedding_space)
        if not centroids or not message_emb:
            return Intent(kind=IntentKind.DIALOGUE, confidence=0.0,
                          embedding=message_emb)
        scored = sorted(
            ((kind, _cosine(message_emb, vec)) for kind, vec in centroids.items()),
            key=lambda x: x[1], reverse=True,
        )
        best_kind, best_score = scored[0]
        if best_score < self.intent_threshold:
            return Intent(
                kind=IntentKind.DIALOGUE, confidence=float(best_score),
                embedding=message_emb,
            )
        # Recruit additional faculties when their score is close to best.
        threshold = max(self.intent_threshold, best_score * 0.85)
        faculties = [
            self._faculty_name_for(kind)
            for kind, score in scored
            if score >= threshold
        ]
        return Intent(
            kind=best_kind,
            confidence=float(best_score),
            faculties=faculties,
            embedding=message_emb,
        )

    def _message_embedding(self, message: str, embedding_space: Any) -> list[float]:
        if embedding_space is None:
            return []
        # Tokenize the message using the same word splitter as the neural
        # tokenizer so embeddings are looked up consistently with how the
        # space was trained.
        try:
            from darwin.neural.tokenizer import split_words

            words = split_words(message)
        except Exception:
            words = message.lower().split()
        if not words:
            return []
        try:
            vecs = [embedding_space.embed(w) for w in words]
        except Exception:
            return []
        return _mean(vecs)

    def _centroids(self, embedding_space: Any) -> dict[IntentKind, list[float]]:
        if embedding_space is None:
            return {}
        # Cached centroids are invalidated whenever the vocab grows past the
        # cache's snapshot threshold — the trainer continuously adds tokens.
        cache_size = getattr(self, "_centroid_vocab_at", -1)
        try:
            current_size = embedding_space.vocab_size()
        except Exception:
            current_size = 0
        if not self._centroid_cache or abs(current_size - cache_size) > 16:
            self._centroid_cache = {}
            for kind, exemplars in _FACULTY_EXEMPLARS.items():
                vecs: list[list[float]] = []
                for token in exemplars:
                    try:
                        vecs.append(embedding_space.embed(token))
                    except Exception:
                        continue
                if vecs:
                    self._centroid_cache[kind] = _mean(vecs)
            self._centroid_vocab_at = current_size
        return self._centroid_cache

    @staticmethod
    def _faculty_name_for(kind: IntentKind) -> str:
        return {
            IntentKind.COMPUTE: "calculator",
            IntentKind.DERIVE: "coder",
            IntentKind.PLAN: "planner",
            IntentKind.RESEARCH: "researcher",
            IntentKind.RECALL: "researcher",
            IntentKind.SYNTHESIZE: "scientist",
            IntentKind.DIALOGUE: "conversationalist",
        }.get(kind, "conversationalist")

    # -- solve ------------------------------------------------------------- #

    def solve(self, message: str, intent: Intent | None = None) -> MindReply:
        """Recruit faculties internally; compose a single Darwin-voice reply."""

        if intent is None:
            intent = self.consider(message)
        if not intent.is_actionable():
            return MindReply(
                text="", intent_kind=intent.kind.value,
                confidence=float(intent.confidence), declined=True,
            )
        # Dispatch by kind. Each branch produces prose that doesn't name
        # the faculty or the intent kind. Provenance lives on MindReply but
        # is not in the text.
        if intent.kind is IntentKind.COMPUTE:
            return self._solve_compute(message, intent)
        if intent.kind is IntentKind.DERIVE:
            return self._solve_derive(message, intent)
        if intent.kind is IntentKind.PLAN:
            return self._solve_plan(message, intent)
        if intent.kind is IntentKind.RESEARCH or intent.kind is IntentKind.RECALL:
            return self._solve_research(message, intent)
        if intent.kind is IntentKind.SYNTHESIZE:
            return self._solve_synthesize(message, intent)
        return MindReply(
            text="", intent_kind=intent.kind.value, declined=True,
        )

    # -- faculty-specific composers ---------------------------------------- #

    def _solve_compute(self, message: str, intent: Intent) -> MindReply:
        try:
            sol = self.calculator.solve(MathProblem(prompt=message))
        except Exception:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        if not sol.succeeded or not sol.answer:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        text = self._compose_compute_prose(sol.answer)
        return MindReply(
            text=text, intent_kind=intent.kind.value,
            faculties_used=["calculator"], confidence=float(sol.confidence),
            steps=list(sol.steps),
        )

    @staticmethod
    def _compose_compute_prose(answer: str) -> str:
        clean = answer.strip()
        if not clean:
            return ""
        return f"The result works out to {clean}."

    def _solve_derive(self, message: str, intent: Intent) -> MindReply:
        try:
            sol = self.coder.solve(CodeProblem(prompt=message))
        except Exception:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        if not sol.answer:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        text = self._compose_derive_prose(sol.answer)
        return MindReply(
            text=text, intent_kind=intent.kind.value,
            faculties_used=["coder"], confidence=float(sol.confidence),
            steps=list(sol.steps),
        )

    @staticmethod
    def _compose_derive_prose(answer: str) -> str:
        clean = answer.strip()
        if not clean:
            return ""
        # Surface code as a fenced block but introduce it in Darwin's voice.
        if "\n" in clean and ("def " in clean or "return" in clean):
            return f"Here is one way to write it:\n\n```\n{clean}\n```"
        return f"Here is what I would say to that: {clean}"

    def _solve_plan(self, message: str, intent: Intent) -> MindReply:
        try:
            sol = self.planner.solve(PlanningProblem(prompt=message))
        except Exception:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        if not sol.answer:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        text = self._compose_plan_prose(sol.answer)
        return MindReply(
            text=text, intent_kind=intent.kind.value,
            faculties_used=["planner"], confidence=float(sol.confidence),
            steps=list(sol.steps),
        )

    @staticmethod
    def _compose_plan_prose(answer: str) -> str:
        clean = answer.strip()
        if not clean:
            return ""
        return clean

    def _solve_research(self, message: str, intent: Intent) -> MindReply:
        try:
            sol = self.researcher.solve(ResearchProblem(prompt=message))
        except Exception:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        if not sol.answer:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        text = sol.answer.strip()
        return MindReply(
            text=text, intent_kind=intent.kind.value,
            faculties_used=["researcher"], confidence=float(sol.confidence),
            steps=list(sol.steps),
        )

    def _solve_synthesize(self, message: str, intent: Intent) -> MindReply:
        # Multi-faculty: run the scientist (which is the natural multi-step
        # composer in the existing agents) and let it integrate.
        try:
            sol = self.scientist.solve(ScienceProblem(prompt=message))
        except Exception:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        if not sol.answer:
            return MindReply(text="", intent_kind=intent.kind.value, declined=True)
        return MindReply(
            text=sol.answer.strip(), intent_kind=intent.kind.value,
            faculties_used=["scientist"], confidence=float(sol.confidence),
            steps=list(sol.steps),
        )

    # -- bus telemetry ----------------------------------------------------- #

    def publish_step(self, intent: Intent, reply: MindReply) -> None:
        """Publish MIND_STEP for the brain terminal — never visible to chat."""

        bus = getattr(self.runtime, "bus", None)
        if bus is None:
            return
        try:
            from darwin.mysterio.bus import BusTopic

            bus.publish(
                BusTopic.MIND_STEP,
                {
                    "intent": intent.to_record(),
                    "reply": reply.to_record(),
                },
                source="mind",
            )
        except Exception:
            return


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for k in range(min(dim, len(vec))):
            acc[k] += vec[k]
    n = float(len(vectors))
    return [v / n for v in acc]


__all__ = ["Mind"]
