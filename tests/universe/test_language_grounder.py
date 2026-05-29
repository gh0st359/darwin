"""Tests for the LanguageGrounder — words → concepts."""

from __future__ import annotations

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.language_universe import (
    LanguageGrounder,
    content_words,
    tokenize,
)
from darwin.universe.primitive_seed import seed_primitives


def test_tokenize_strips_punctuation_and_lowercases() -> None:
    assert tokenize("Tell me about CAUSE, please!") == [
        "tell", "me", "about", "cause", "please"
    ]


def test_content_words_filters_stopwords_and_short_tokens() -> None:
    words = content_words("I am thinking about the cause of change.")
    assert "thinking" in words
    assert "cause" in words
    assert "change" in words
    assert "the" not in words and "i" not in words and "am" not in words


def test_grounder_resolves_exact_primitives() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    grounder = LanguageGrounder(u, auto_register=False)
    result = grounder.ground("What is the cause of change?")
    names = result.concept_names
    assert "cause" in names
    assert "change" in names


def test_grounder_auto_registers_unknown_words_by_default() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    grounder = LanguageGrounder(u)
    result = grounder.ground("tell me about whorzplatz")
    assert "whorzplatz" in result.concept_names
    assert u.has("whorzplatz")


def test_grounder_can_decline_to_auto_register() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    grounder = LanguageGrounder(u, auto_register=False)
    result = grounder.ground("tell me about quarbleflinch")
    assert "quarbleflinch" not in result.concept_names
    assert "quarbleflinch" in result.unrecognized
    assert not u.has("quarbleflinch")


def test_grounder_resolves_aliases() -> None:
    u = ConceptUniverse()
    u.add_concept("free_will", domain="philosophy", aliases=("freewill", "agency_choice"))
    grounder = LanguageGrounder(u, auto_register=False)
    result = grounder.ground("can you discuss freewill")
    assert "free_will" in result.concept_names


def test_grounder_handles_repeated_words_once() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    grounder = LanguageGrounder(u)
    result = grounder.ground("cause cause cause and effect")
    names = result.concept_names
    assert names.count("cause") == 1
    assert "effect" in names


def test_grounding_serializes_to_record() -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    grounder = LanguageGrounder(u)
    result = grounder.ground("self and model")
    record = result.to_record()
    assert "text" in record and "grounded" in record
    assert all("concept" in g for g in record["grounded"])
