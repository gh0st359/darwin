"""Tests for ActiveLearner — Darwin asks the operator to fill gaps."""

from __future__ import annotations

from darwin.universe.active_learning import ActiveLearner, LearningProbe
from darwin.universe.concept_universe import ConceptUniverse


def _world(*edges: tuple[str, str, str]) -> ConceptUniverse:
    u = ConceptUniverse()
    for src, kind, tgt in edges:
        u.add_relation(src, tgt, kind, ensure_concepts=True)
    return u


def test_missing_link_probe_for_kind_check() -> None:
    """If 'is dog an animal' is asked, and Darwin knows dog→mammal but not
    mammal→animal, the probe should ask whether mammal is_a animal."""

    u = _world(("dog", "is_a", "mammal"))
    u.add_concept("animal")
    learner = ActiveLearner(u)
    probes = learner.probe(
        question_kind="kind_check",
        grounded_concepts=["dog", "animal"],
        inferences=[],
    )
    # Expect a probe pointing at the mammal→animal gap.
    assert any(p.source == "mammal" and p.target == "animal" for p in probes)


def test_no_probes_when_high_confidence_inference_exists() -> None:
    u = _world(("dog", "is_a", "mammal"))
    learner = ActiveLearner(u)

    class _HiConf:
        confidence = 0.9

    probes = learner.probe(
        question_kind="kind_check",
        grounded_concepts=["dog", "mammal"],
        inferences=[_HiConf()],
    )
    assert probes == []


def test_definition_probe_when_concept_has_no_definition() -> None:
    u = ConceptUniverse()
    u.add_concept("undefined_thing")
    learner = ActiveLearner(u)
    probes = learner.probe(
        question_kind="definition",
        grounded_concepts=["undefined_thing"],
        inferences=[],
    )
    assert any(
        "undefined_thing" in p.question.lower()
        and p.expected_kind == "definition"
        for p in probes
    )


def test_cross_domain_probe_when_concepts_span_unrelated_domains() -> None:
    u = ConceptUniverse()
    u.add_concept("music", domain="arts")
    u.add_concept("ratio", domain="mathematics")
    learner = ActiveLearner(u)
    probes = learner.probe(
        question_kind="relation",
        grounded_concepts=["music", "ratio"],
        inferences=[],
    )
    assert any(
        "music" in p.question.lower() and "ratio" in p.question.lower()
        for p in probes
    )


def test_no_duplicate_probes_for_repeated_question() -> None:
    u = _world(("dog", "is_a", "mammal"))
    u.add_concept("animal")
    learner = ActiveLearner(u)
    first = learner.probe(
        question_kind="kind_check",
        grounded_concepts=["dog", "animal"],
        inferences=[],
    )
    second = learner.probe(
        question_kind="kind_check",
        grounded_concepts=["dog", "animal"],
        inferences=[],
    )
    assert first
    # Same probes should not be re-asked.
    assert second == []


def test_no_probes_when_no_grounded_concepts() -> None:
    u = ConceptUniverse()
    learner = ActiveLearner(u)
    probes = learner.probe(
        question_kind="definition",
        grounded_concepts=[],
        inferences=[],
    )
    assert probes == []


def test_learning_probe_serializes() -> None:
    p = LearningProbe(
        question="?", source="x", target="y", expected_kind="is_a",
        rationale="r", score=0.6,
    )
    record = p.to_record()
    assert record["question"] == "?"
    assert record["expected_kind"] == "is_a"
