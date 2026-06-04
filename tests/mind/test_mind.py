"""Mind — composition surface above the faculties."""

from __future__ import annotations

from darwin.faculties import Mind
from darwin.mind.intent import IntentKind
from darwin.neural.learned_space import LearnedCausalSpace


class _FakeRuntime:
    """Minimal stand-in carrying just the runtime hooks Mind reads."""

    def __init__(self):
        self.embedding_space = LearnedCausalSpace(dim=24, seed=11)
        self.bus = None
        self.universe = None
        self.cortical_mesh = None
        self.tool_world = None
        self.tool_sandbox_root = None
        self.tool_registry = None
        self.speech_pipeline = None
        self.autonomous_runner = None
        self.reasoning_dispatcher = None


def _train_for_compute(runtime: _FakeRuntime) -> None:
    # Train so "calculate", "sum", etc. co-occur with arithmetic-flavoured
    # terms — the COMPUTE centroid sharpens.
    for _ in range(60):
        runtime.embedding_space.train_tokens(
            ["calculate", "sum", "equation", "number", "math", "what", "is"]
        )


def test_mind_constructs_six_faculties():
    mind = Mind(runtime=_FakeRuntime())
    assert mind.coder is not None
    assert mind.calculator is not None
    assert mind.scientist is not None
    assert mind.planner is not None
    assert mind.researcher is not None
    assert mind.conversationalist is not None


def test_mind_back_compat_properties_expose_legacy_names():
    mind = Mind(runtime=_FakeRuntime())
    # AgentRegistry's surface — autonomy/executor and bench tests rely on it.
    assert mind.code is mind.coder
    assert mind.math is mind.calculator
    assert mind.science is mind.scientist
    assert mind.planning is mind.planner
    assert mind.research is mind.researcher
    assert mind.dialogue is mind.conversationalist
    assert len(mind.all()) == 6
    assert mind.summary()["count"] == 6


def test_recruit_by_capability_returns_faculty():
    mind = Mind(runtime=_FakeRuntime())
    assert mind.recruit("code") is mind.coder
    assert mind.recruit("math") is mind.calculator
    assert mind.recruit("dialogue") is mind.conversationalist
    assert mind.recruit("unknown_capability") is None


def test_consider_declines_on_empty_input():
    mind = Mind(runtime=_FakeRuntime())
    intent = mind.consider("")
    assert intent.kind is IntentKind.DECLINE


def test_consider_falls_back_to_dialogue_when_centroids_below_threshold():
    runtime = _FakeRuntime()
    mind = Mind(runtime=runtime, intent_threshold=0.99)
    intent = mind.consider("Talk to me about whatever.")
    # Untrained space: centroids exist but similarity ≪ 0.99.
    assert intent.kind in (IntentKind.DIALOGUE, IntentKind.DECLINE)


def test_consider_picks_compute_for_arithmetic_phrasing():
    runtime = _FakeRuntime()
    _train_for_compute(runtime)
    mind = Mind(runtime=runtime, intent_threshold=0.05)
    intent = mind.consider("What is the sum of 7 and 3?")
    assert intent.kind in (
        IntentKind.COMPUTE, IntentKind.RECALL, IntentKind.SYNTHESIZE,
    )
    assert intent.confidence > 0


def test_solve_compute_returns_prose_without_faculty_name():
    runtime = _FakeRuntime()
    _train_for_compute(runtime)
    mind = Mind(runtime=runtime, intent_threshold=0.0)
    # Force a compute intent by bypassing the classifier.
    from darwin.mind.intent import Intent

    intent = Intent(kind=IntentKind.COMPUTE, confidence=1.0)
    reply = mind.solve("What is 2 + 3?", intent=intent)
    assert reply.text  # got something
    text_lower = reply.text.lower()
    assert "5" in reply.text
    # No faculty / intent leakage.
    for token in ("calculator", "coder", "mathagent", "intent", "compute"):
        assert token not in text_lower


def test_solve_declined_intent_returns_empty_text():
    mind = Mind(runtime=_FakeRuntime())
    from darwin.mind.intent import Intent

    intent = Intent(kind=IntentKind.DIALOGUE, confidence=0.0)
    reply = mind.solve("hi", intent=intent)
    assert reply.text == ""
    assert reply.declined is True
