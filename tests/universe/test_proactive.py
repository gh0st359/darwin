"""Tests for the proactive dialogue layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.universe.proactive import VolunteeredRemark, choose_volunteer


@dataclass
class _FakeHypothesis:
    source: str
    target: str
    kind: str = "is_a"
    confidence: float = 0.5
    pathway: str = "transitive"
    rationale: str = "because"

    def as_question(self) -> str:
        return f"Is {self.source} a {self.target}?"


@dataclass
class _FakeContradiction:
    a: str
    b: str
    reason: str = "opposes"


@dataclass
class _FakeCuriosity:
    question: str
    concepts: list = field(default_factory=list)
    score: float = 0.5


def test_hypothesis_winners_must_involve_grounded_concepts() -> None:
    """A high-confidence hypothesis NOT about anything the user just
    grounded must NOT be volunteered — that would feel random."""

    remark = choose_volunteer(
        grounded_concepts=["alpha"],
        hypotheses=[_FakeHypothesis(source="zeta", target="omega", confidence=0.9)],
    )
    assert remark is None


def test_hypothesis_winner_when_grounded_and_high_confidence() -> None:
    remark = choose_volunteer(
        grounded_concepts=["alpha"],
        hypotheses=[
            _FakeHypothesis(source="alpha", target="beta", confidence=0.85),
        ],
    )
    assert remark is not None
    assert remark.source_kind == "hypothesis"
    assert "alpha" in remark.text.lower() or "beta" in remark.text.lower()


def test_contradiction_volunteered_when_about_grounded_concepts() -> None:
    remark = choose_volunteer(
        grounded_concepts=["fire"],
        contradictions=[_FakeContradiction(a="fire", b="water")],
    )
    assert remark is not None
    assert remark.source_kind == "contradiction"


def test_curiosity_volunteered_when_about_grounded_concepts() -> None:
    remark = choose_volunteer(
        grounded_concepts=["mind"],
        curiosities=[
            _FakeCuriosity(
                question="What does mind relate to?",
                concepts=["mind"],
                score=0.7,
            ),
        ],
    )
    assert remark is not None
    assert remark.source_kind == "curiosity"


def test_no_volunteer_during_greeting() -> None:
    remark = choose_volunteer(
        grounded_concepts=["alpha"],
        hypotheses=[_FakeHypothesis(source="alpha", target="beta", confidence=0.9)],
        last_question_kind="greeting",
    )
    assert remark is None


def test_priority_hypothesis_beats_contradiction_beats_curiosity() -> None:
    # All three available; hypothesis must win.
    remark = choose_volunteer(
        grounded_concepts=["x"],
        hypotheses=[_FakeHypothesis(source="x", target="y", confidence=0.9)],
        contradictions=[_FakeContradiction(a="x", b="z")],
        curiosities=[
            _FakeCuriosity(question="?", concepts=["x"], score=0.9),
        ],
    )
    assert remark is not None
    assert remark.source_kind == "hypothesis"


def test_low_confidence_hypothesis_not_volunteered() -> None:
    remark = choose_volunteer(
        grounded_concepts=["x"],
        hypotheses=[_FakeHypothesis(source="x", target="y", confidence=0.4)],
    )
    assert remark is None


def test_volunteered_remark_serializes() -> None:
    r = VolunteeredRemark(
        text="hi", source_kind="hypothesis", confidence=0.7,
        grounded_concepts=["a"],
    )
    record = r.to_record()
    assert record["text"] == "hi"
    assert record["source_kind"] == "hypothesis"
