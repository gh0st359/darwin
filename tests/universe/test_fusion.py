"""Tests for ConceptFusion — chat statements become typed graph edges."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.fusion import ConceptFusion, FusedRelation


def test_is_a_statement_adds_is_a_edge() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("a sparrow is a bird")
    assert result.added
    assert any(r.kind == "is_a" for r in result.added)
    assert u.has("sparrow") and u.has("bird")
    rels = u.neighbors("sparrow", kinds=["is_a"])
    assert any(r.target == "bird" for r in rels)


def test_causes_statement_adds_causes_edge() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("rain causes flooding")
    assert any(r.kind == "causes" for r in result.added)
    rels = u.neighbors("rain", kinds=["causes"])
    assert any(r.target == "flooding" for r in rels)


def test_requires_statement_adds_requires_edge() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("combustion requires oxygen")
    assert any(r.kind == "requires" for r in result.added)


def test_opposes_statement_adds_opposes_edge() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("entropy opposes order")
    assert any(r.kind == "opposes" for r in result.added)


def test_multiple_statements_in_one_utterance() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("rain causes flooding. a sparrow is a bird.")
    kinds = {r.kind for r in result.added}
    assert "causes" in kinds
    assert "is_a" in kinds


def test_self_loop_rejected() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("rain causes rain")
    assert not any(r.source == r.target for r in result.added)


def test_pronouns_rejected_as_concepts() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("it is a thing")
    # 'it' should be rejected as a concept name.
    assert all(r.source != "it" for r in result.added)


def test_duplicate_edge_is_idempotent() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    fuser.fuse("a dog is a mammal")
    second = fuser.fuse("a dog is a mammal")
    # Second statement should NOT add a duplicate edge.
    assert not second.added
    rels = u.neighbors("dog", kinds=["is_a"])
    targets = [r.target for r in rels]
    assert targets.count("mammal") == 1


def test_introductory_phrases_stripped() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    result = fuser.fuse("tell me that water is a liquid")
    assert any(r.source == "water" and r.target == "liquid" for r in result.added)


def test_summary_aggregates_kinds() -> None:
    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    fuser.fuse("a sparrow is a bird")
    fuser.fuse("rain causes flooding")
    summary = fuser.summary()
    assert summary["total_fused"] == 2
    assert "is_a" in summary["by_kind"]
    assert "causes" in summary["by_kind"]


def test_fused_relation_serializes() -> None:
    fused = FusedRelation(source="a", target="b", kind="is_a", surface="a is a b")
    record = fused.to_record()
    assert record["source"] == "a"
    assert record["kind"] == "is_a"


def test_inference_engine_can_use_fused_edges() -> None:
    """Critically: an inference query over a fused edge must succeed."""

    from darwin.universe.inference import InferenceEngine

    u = ConceptUniverse()
    fuser = ConceptFusion(u)
    fuser.fuse("a dolphin is a mammal")
    fuser.fuse("a mammal is an animal")
    engine = InferenceEngine(u)
    inf = engine.is_a_chain("dolphin", "animal")
    assert inf is not None
    assert len(inf.chain) == 2  # dolphin -> mammal -> animal
