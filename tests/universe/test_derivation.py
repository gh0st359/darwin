"""Tests for the ConceptDeriver — Darwin growing its own universe."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.derivation import ConceptDeriver, DerivedConcept
from darwin.universe.primitive_seed import seed_primitives


def _baseline() -> tuple[ConceptUniverse, ConceptDeriver]:
    u = ConceptUniverse()
    seed_primitives(u)
    deriver = ConceptDeriver(u, cooccurrence_threshold=3)
    return u, deriver


def test_derive_with_no_input_returns_no_concepts() -> None:
    u, d = _baseline()
    accepted = d.derive()
    assert accepted == []


def test_cooccurrence_derives_a_link_concept_between_known_concepts() -> None:
    u, d = _baseline()
    # 'cause' and 'effect' are in the primitive seed.
    for _ in range(4):
        d.observe_text("the cause and the effect")
    accepted = d.derive()
    pathways = {c.pathway for c in accepted}
    assert "cooccurrence" in pathways
    # Either a link node was added, or a related_to edge between them.
    cause_neighbors = {rel.target for rel in u.neighbors("cause")}
    effect_neighbors = {rel.target for rel in u.neighbors("effect")}
    assert "effect" in cause_neighbors or "cause" in effect_neighbors or any(
        c.name.startswith("link_") for c in accepted
    )


def test_cooccurrence_adds_new_concept_when_one_partner_is_ungrounded() -> None:
    u, d = _baseline()
    # 'self' is in the seed; 'whorzplatz' is not.
    for _ in range(5):
        d.observe_text("self and whorzplatz together again")
    accepted = d.derive()
    assert any(c.name == "whorzplatz" for c in accepted) or u.has("whorzplatz")


def test_derive_from_causal_regularities_creates_named_concept() -> None:
    u, d = _baseline()

    class _FakeBelief:
        def __init__(self, action: str, variable: str, effect: str, confidence: float) -> None:
            self.action = action
            self.variable = variable
            self.effect = effect
            self.confidence = confidence
            self.samples = 10

    class _FakeCausalModel:
        def beliefs(self, limit: int = 64):
            return [
                _FakeBelief("press_button", "ringing", "+1", 0.9),
                _FakeBelief("open_window", "noise", "+1", 0.85),
                _FakeBelief("noise_only_low_conf", "noise", "+1", 0.3),
            ]

    class _FakeDarwin:
        causal_model = _FakeCausalModel()

    accepted = d.derive(darwin=_FakeDarwin())
    pathways = [c.pathway for c in accepted]
    assert pathways.count("regularity") == 2  # 0.3 confidence one is below threshold
    names = {c.name for c in accepted}
    assert any(name.startswith("reg_") for name in names)


def test_deriver_does_not_re_derive_same_signature() -> None:
    u, d = _baseline()

    class _FakeBelief:
        action = "press"
        variable = "alarm"
        effect = "+1"
        confidence = 0.9
        samples = 10

    class _FakeCausalModel:
        def beliefs(self, limit: int = 64):
            return [_FakeBelief()]

    class _FakeDarwin:
        causal_model = _FakeCausalModel()

    first = d.derive(darwin=_FakeDarwin())
    second = d.derive(darwin=_FakeDarwin())
    assert first
    assert second == []


def test_composition_pathway_proposes_parent_kind_for_similar_concepts() -> None:
    u = ConceptUniverse()
    # Two concepts that share most of their outgoing structure should
    # trigger the composition pathway to propose a parent kind.
    u.add_concept("hammer", domain="tool")
    u.add_concept("wrench", domain="tool")
    u.add_concept("metal", domain="material")
    u.add_concept("grip", domain="part")
    u.add_concept("user", domain="person")
    u.add_relation("hammer", "metal", "part_of")
    u.add_relation("hammer", "grip", "part_of")
    u.add_relation("hammer", "user", "requires")
    u.add_relation("wrench", "metal", "part_of")
    u.add_relation("wrench", "grip", "part_of")
    u.add_relation("wrench", "user", "requires")
    deriver = ConceptDeriver(u, cooccurrence_threshold=999)  # cooc off
    accepted = deriver.derive()
    composed = [c for c in accepted if c.pathway == "composition"]
    assert composed
    # The proposed parent should declare both as is_a it.
    relations = composed[0].relations
    children = {src for src, kind, _ in relations if kind == "is_a"}
    assert {"hammer", "wrench"} <= children


def test_deriver_summary_reports_pathway_counts() -> None:
    u, d = _baseline()
    for _ in range(5):
        d.observe_text("self alongside whorzplatz")
    d.derive()
    summary = d.summary()
    assert "tracked_word_pairs" in summary
    assert summary["proposals_accepted"] >= 1
    assert "cooccurrence" in summary["pathways"]


def test_derived_concept_serializes() -> None:
    concept = DerivedConcept(
        name="x", definition="t", derived_from=("a", "b"), pathway="regularity",
    )
    record = concept.to_record()
    assert record["name"] == "x"
    assert record["pathway"] == "regularity"
