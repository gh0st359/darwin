"""Tests for the per-user OperatorModel."""

from __future__ import annotations

from darwin.operator_model import OperatorModel, OperatorModelRegistry


def test_observe_updates_style_profile_toward_short() -> None:
    model = OperatorModel(user_id="terse")
    for turn in ["yes", "ok", "hmm", "right"]:
        model.observe(turn)
    assert model.style_profile.samples == 4
    assert model.preferred_length() == "short"


def test_observe_updates_style_profile_toward_long() -> None:
    model = OperatorModel(user_id="verbose")
    long_turn = (
        "I want to know everything you can tell me about the partition "
        "between the grounded substrate and the interior substrate, in "
        "as much detail as possible please, including how the divergence "
        "probe interprets the gap and what it would mean for a notable "
        "score to surface in the brain terminal during normal operation."
    )
    for _ in range(3):
        model.observe(long_turn)
    assert model.preferred_length() == "long"


def test_observe_extracts_interests() -> None:
    model = OperatorModel(user_id="curious")
    for _ in range(3):
        model.observe("tell me about partition and divergence and visibility")
    interests = model.top_interests()
    assert "partition" in interests
    assert "divergence" in interests
    assert "visibility" in interests


def test_agreement_and_disagreement_detection() -> None:
    model = OperatorModel(user_id="opinionated")
    model.observe("yes, that's exactly right")
    model.observe("no, that's wrong")
    assert any("right" in a for a in model.agreements)
    assert any("wrong" in d for d in model.disagreements)


def test_registry_caches_models_per_user_id() -> None:
    registry = OperatorModelRegistry()
    a = registry.get("alice")
    b = registry.get("bob")
    a_again = registry.get("alice")
    assert a is a_again
    assert a is not b
    assert set(registry.known_users()) == {"alice", "bob"}


def test_registry_default_anonymous_model() -> None:
    registry = OperatorModelRegistry()
    anon_a = registry.get()
    anon_b = registry.get(None)
    assert anon_a is anon_b
    assert anon_a.user_id == "anonymous"


def test_to_record_serializes_full_model() -> None:
    model = OperatorModel(user_id="serializable")
    model.observe("yes, that's right")
    model.observe("tell me about uncertainty")
    record = model.to_record()
    assert record["user_id"] == "serializable"
    assert record["samples"] == 2
    assert isinstance(record["top_interests"], list)
    assert isinstance(record["agreements"], list)
