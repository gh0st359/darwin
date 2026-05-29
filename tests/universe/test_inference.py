"""Tests for the symbolic InferenceEngine."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.inference import (
    Contradiction,
    Inference,
    InferenceEngine,
)
from darwin.universe.primitive_seed import seed_primitives


def _world(*edges: tuple[str, str, str]) -> ConceptUniverse:
    u = ConceptUniverse()
    for source, kind, target in edges:
        u.add_relation(source, target, kind, ensure_concepts=True)
    return u


# -- is_a chains -------------------------------------------------------------


def test_is_a_chain_one_hop() -> None:
    u = _world(("dog", "is_a", "mammal"))
    engine = InferenceEngine(u)
    inf = engine.is_a_chain("dog", "mammal")
    assert isinstance(inf, Inference)
    assert inf.operator == "is_a_chain"
    assert len(inf.chain) == 1


def test_is_a_chain_transitively_two_hops() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
    )
    engine = InferenceEngine(u)
    inf = engine.is_a_chain("dog", "animal")
    assert inf is not None
    assert inf.target == "animal"
    assert len(inf.chain) == 2


def test_is_a_chain_no_chain_returns_none() -> None:
    u = _world(("dog", "is_a", "mammal"))
    engine = InferenceEngine(u)
    assert engine.is_a_chain("dog", "fish") is None


def test_super_kinds_returns_all_reachable() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
        ("animal", "is_a", "thing"),
    )
    engine = InferenceEngine(u)
    supers = engine.super_kinds("dog")
    assert set(supers) == {"mammal", "animal", "thing"}


def test_sub_kinds_returns_all_descendents() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("cat", "is_a", "mammal"),
        ("poodle", "is_a", "dog"),
    )
    engine = InferenceEngine(u)
    subs = engine.sub_kinds("mammal")
    assert {"dog", "cat", "poodle"} <= set(subs)


# -- inheritance -------------------------------------------------------------


def test_inherited_properties_propagate_from_super_kinds() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "part_of", "vertebrates"),
        ("mammal", "requires", "spine"),
    )
    engine = InferenceEngine(u)
    inheritances = engine.inherited_properties("dog")
    targets = {inf.target for inf in inheritances}
    assert "spine" in targets
    assert "vertebrates" in targets


# -- causal chains -----------------------------------------------------------


def test_causal_chain_one_hop() -> None:
    u = _world(("heat", "causes", "expansion"))
    engine = InferenceEngine(u)
    inf = engine.causal_chain("heat", "expansion")
    assert inf is not None
    assert inf.operator == "causal_chain"


def test_causal_chain_transitively() -> None:
    u = _world(
        ("rain", "causes", "wetness"),
        ("wetness", "causes", "slipperiness"),
        ("slipperiness", "causes", "falls"),
    )
    engine = InferenceEngine(u)
    inf = engine.causal_chain("rain", "falls")
    assert inf is not None
    assert len(inf.chain) == 3


def test_downstream_and_upstream_reachable() -> None:
    u = _world(
        ("a", "causes", "b"),
        ("b", "causes", "c"),
        ("c", "causes", "d"),
    )
    engine = InferenceEngine(u)
    assert "d" in engine.downstream_effects("a")
    assert "a" in engine.upstream_causes("d")


# -- contradictions ----------------------------------------------------------


def test_contradicts_detects_direct_opposition() -> None:
    u = _world(("hot", "opposes", "cold"))
    engine = InferenceEngine(u)
    c = engine.contradicts("hot", "cold")
    assert isinstance(c, Contradiction)


def test_contradicts_detects_super_kind_opposition() -> None:
    u = _world(
        ("specific_truth", "is_a", "true"),
        ("specific_lie", "is_a", "false"),
        ("true", "opposes", "false"),
    )
    engine = InferenceEngine(u)
    c = engine.contradicts("specific_truth", "specific_lie")
    assert c is not None
    assert "oppose" in c.reason


def test_contradicts_returns_none_when_no_opposition() -> None:
    u = _world(("dog", "is_a", "mammal"))
    engine = InferenceEngine(u)
    assert engine.contradicts("dog", "mammal") is None


# -- explanation -------------------------------------------------------------


def test_explain_yields_multiple_proof_chains_when_available() -> None:
    u = _world(
        ("a", "is_a", "b"),
        ("a", "causes", "c"),
        ("c", "causes", "b"),
    )
    engine = InferenceEngine(u)
    explanations = engine.explain("a", "b")
    assert explanations
    operators = {inf.operator for inf in explanations}
    assert "is_a_chain" in operators


# -- proactive derivation ---------------------------------------------------


def test_derive_new_relations_proposes_transitive_closure() -> None:
    u = _world(
        ("dog", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
    )
    engine = InferenceEngine(u)
    proposals = engine.derive_new_relations()
    # The transitive closure (dog, is_a, animal) is missing — it must be proposed.
    assert ("dog", "is_a", "animal") in proposals


def test_inference_serializes() -> None:
    u = _world(("a", "is_a", "b"))
    engine = InferenceEngine(u)
    inf = engine.is_a_chain("a", "b")
    assert inf is not None
    record = inf.to_record()
    assert record["operator"] == "is_a_chain"
    assert record["chain"]
