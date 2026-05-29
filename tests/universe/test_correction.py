"""Tests for CorrectionDetector + apply_correction."""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.universe.correction import (
    Correction,
    apply_correction,
    detect_correction,
)


def test_detect_pure_negation() -> None:
    c = detect_correction("No, that's wrong.")
    assert c is not None
    assert c.kind == "negate_prior"


def test_detect_negation_with_replacement() -> None:
    c = detect_correction("No, actually a dog is a vertebrate.")
    assert c is not None
    assert c.kind == "replace"
    assert "dog" in c.replacement.lower()


def test_detect_standalone_replacement() -> None:
    c = detect_correction("Actually a whale is a mammal.")
    assert c is not None
    assert c.kind == "replace"
    assert "whale" in c.replacement.lower()


def test_detect_retraction() -> None:
    c = detect_correction("I was wrong about that earlier.")
    assert c is not None
    assert c.kind == "retract"


def test_detect_no_correction_in_neutral_text() -> None:
    c = detect_correction("Tell me more about gravity.")
    assert c is None


def test_detect_negation_emoji_friendly() -> None:
    c = detect_correction("nope")
    assert c is not None
    assert c.kind == "negate_prior"


@dataclass
class _FakeInference:
    operator: str = "is_a_chain"
    source: str = "a"
    target: str = "b"


class _FakeHypEngine:
    def __init__(self):
        self.refuted = []

    def refute(self, src, kind, tgt):
        self.refuted.append((src, kind, tgt))


@dataclass
class _FakeFused:
    source: str
    kind: str
    target: str


class _FakeFusion:
    def __init__(self):
        self.fused_calls = []
        self._recent = []

    def fuse(self, text):
        self.fused_calls.append(text)

    def recent(self, limit=4):
        return self._recent


def test_apply_negate_prior_refutes_last_inferences() -> None:
    correction = Correction(kind="negate_prior", text="no")
    hyp = _FakeHypEngine()
    refuted = apply_correction(
        correction,
        last_grounded_concepts=["a", "b"],
        last_inferences=[_FakeInference()],
        fusion=_FakeFusion(),
        hypothesis_engine=hyp,
        universe=None,
    )
    assert refuted == [("a", "is_a", "b")]
    assert hyp.refuted == [("a", "is_a", "b")]


def test_apply_replace_refutes_and_fuses_replacement() -> None:
    correction = Correction(
        kind="replace",
        text="no, actually a whale is a mammal",
        replacement="a whale is a mammal",
    )
    fusion = _FakeFusion()
    apply_correction(
        correction,
        last_grounded_concepts=["a", "b"],
        last_inferences=[_FakeInference()],
        fusion=fusion,
        hypothesis_engine=_FakeHypEngine(),
        universe=None,
    )
    assert fusion.fused_calls == ["a whale is a mammal"]


def test_apply_retract_refutes_recent_fused_edges() -> None:
    correction = Correction(kind="retract", text="I was wrong")
    fusion = _FakeFusion()
    fusion._recent = [
        _FakeFused(source="x", kind="is_a", target="y"),
        _FakeFused(source="m", kind="causes", target="n"),
    ]
    hyp = _FakeHypEngine()
    refuted = apply_correction(
        correction,
        last_grounded_concepts=[],
        last_inferences=[],
        fusion=fusion,
        hypothesis_engine=hyp,
        universe=None,
    )
    assert ("x", "is_a", "y") in refuted
    assert ("m", "causes", "n") in refuted
    assert ("x", "is_a", "y") in hyp.refuted


def test_correction_serializes() -> None:
    c = Correction(kind="negate_prior", text="no")
    record = c.to_record()
    assert record["kind"] == "negate_prior"
