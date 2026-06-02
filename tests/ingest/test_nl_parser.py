"""Tests for the hand-rolled NL parser."""

from __future__ import annotations

from darwin.ingest.nl_parser import (
    Fact,
    NLParser,
    extract_facts,
    named_entities,
    pos_tag,
    sentences,
    tokenize,
)


def test_tokenize_basic_sentence() -> None:
    toks = tokenize("The quick brown fox jumps over the lazy dog.")
    surfaces = [t.surface for t in toks]
    assert "quick" in surfaces
    assert "fox" in surfaces
    assert "." in surfaces


def test_tokenize_capitalisation_tracked() -> None:
    toks = tokenize("Darwin meets Newton.")
    caps = [t.surface for t in toks if t.is_capitalised]
    assert "Darwin" in caps
    assert "Newton" in caps


def test_pos_tag_lexicon_known_words() -> None:
    toks = pos_tag(tokenize("The cat is a mammal."))
    by_surface = {t.surface.lower(): t.pos for t in toks}
    assert by_surface["the"] == "DT"
    assert by_surface["is"] == "VBZ"
    assert by_surface["a"] == "DT"


def test_pos_tag_suffix_fallback() -> None:
    toks = pos_tag(tokenize("Quickly running carefully."))
    by_surface = {t.surface.lower(): t.pos for t in toks}
    assert by_surface["quickly"] == "RB"


def test_sentences_segments_at_period() -> None:
    parts = sentences("A is B. C is D. E is F.")
    assert len(parts) == 3


def test_sentences_handles_no_terminal_punctuation() -> None:
    parts = sentences("A single sentence with no period")
    assert parts == ["A single sentence with no period"]


def test_named_entities_picks_up_capitalised_proper_nouns() -> None:
    toks = pos_tag(tokenize("Charles Darwin published in London."))
    ents = named_entities(toks)
    assert "Charles Darwin" in ents


def test_extract_is_a_relation() -> None:
    facts = extract_facts("A neuron is a cell.")
    assert facts
    f = facts[0]
    assert f.subject == "neuron"
    assert f.predicate == "is_a"
    assert f.object == "cell"


def test_extract_causes_relation() -> None:
    facts = extract_facts("Rain causes flooding.")
    assert facts[0].predicate == "causes"
    assert facts[0].subject == "rain"
    assert facts[0].object == "flooding"


def test_extract_part_of_relation() -> None:
    facts = extract_facts("A cell is part of an organism.")
    assert facts[0].subject == "cell"
    assert facts[0].predicate == "part_of"
    assert facts[0].object == "organism"


def test_extract_is_composed_of_swaps_direction() -> None:
    facts = extract_facts("A brain is composed of neurons.")
    # "X is composed of Y" → (Y, part_of, X), i.e. neurons are part_of brain.
    assert facts[0].subject == "neurons"
    assert facts[0].predicate == "part_of"
    assert facts[0].object == "brain"


def test_extract_requires_via_depends_on() -> None:
    facts = extract_facts("Combustion depends on oxygen.")
    assert facts[0].predicate == "requires"
    assert facts[0].object == "oxygen"


def test_extract_no_facts_when_no_verb() -> None:
    assert extract_facts("Big fluffy clouds") == []


def test_nl_parser_aggregates_multiple_sentences() -> None:
    parser = NLParser()
    facts = parser.parse(
        "A neuron is a cell. Rain causes flooding. Entropy opposes order."
    )
    predicates = {f.predicate for f in facts}
    assert "is_a" in predicates
    assert "causes" in predicates
    assert "opposes" in predicates


def test_nl_parser_tracks_counters() -> None:
    parser = NLParser()
    parser.parse("A dog is a mammal. A cat is a mammal.")
    assert parser.sentence_count == 2
    assert parser.fact_count == 2
