"""Linguistics domain seed — language structure, morphology, syntax, semantics."""

from __future__ import annotations


def relations() -> list[tuple[str, str, str, float]]:
    return [
        # Levels of linguistic analysis.
        ("phonetics", "is_a", "linguistic_level", 1.0),
        ("phonology", "is_a", "linguistic_level", 1.0),
        ("morphology", "is_a", "linguistic_level", 1.0),
        ("syntax", "is_a", "linguistic_level", 1.0),
        ("semantics", "is_a", "linguistic_level", 1.0),
        ("pragmatics", "is_a", "linguistic_level", 1.0),
        # Parts of speech.
        ("noun", "is_a", "part_of_speech", 1.0),
        ("verb", "is_a", "part_of_speech", 1.0),
        ("adjective", "is_a", "part_of_speech", 1.0),
        ("adverb", "is_a", "part_of_speech", 1.0),
        ("pronoun", "is_a", "part_of_speech", 1.0),
        ("preposition", "is_a", "part_of_speech", 1.0),
        ("conjunction", "is_a", "part_of_speech", 1.0),
        ("determiner", "is_a", "part_of_speech", 1.0),
        ("interjection", "is_a", "part_of_speech", 1.0),
        # Morphology.
        ("morpheme", "is_a", "linguistic_unit", 1.0),
        ("root", "is_a", "morpheme", 1.0),
        ("prefix", "is_a", "morpheme", 1.0),
        ("suffix", "is_a", "morpheme", 1.0),
        ("infix", "is_a", "morpheme", 1.0),
        ("morpheme", "part_of", "word", 1.0),
        ("word", "part_of", "phrase", 1.0),
        ("phrase", "part_of", "clause", 1.0),
        ("clause", "part_of", "sentence", 1.0),
        # Syntax.
        ("noun_phrase", "is_a", "phrase", 1.0),
        ("verb_phrase", "is_a", "phrase", 1.0),
        ("prepositional_phrase", "is_a", "phrase", 1.0),
        ("subject", "is_a", "grammatical_role", 1.0),
        ("object", "is_a", "grammatical_role", 1.0),
        ("predicate", "is_a", "grammatical_role", 1.0),
        # Phonology.
        ("phoneme", "is_a", "linguistic_unit", 1.0),
        ("vowel", "is_a", "phoneme", 1.0),
        ("consonant", "is_a", "phoneme", 1.0),
        ("syllable", "is_a", "linguistic_unit", 1.0),
        ("phoneme", "part_of", "syllable", 1.0),
        ("syllable", "part_of", "word", 1.0),
        # Languages.
        ("english", "is_a", "language", 1.0),
        ("spanish", "is_a", "language", 1.0),
        ("french", "is_a", "language", 1.0),
        ("german", "is_a", "language", 1.0),
        ("mandarin", "is_a", "language", 1.0),
        ("japanese", "is_a", "language", 1.0),
        ("arabic", "is_a", "language", 1.0),
        ("hindi", "is_a", "language", 1.0),
        ("english", "is_a", "germanic_language", 1.0),
        ("german", "is_a", "germanic_language", 1.0),
        ("spanish", "is_a", "romance_language", 1.0),
        ("french", "is_a", "romance_language", 1.0),
        ("germanic_language", "is_a", "indo_european_language", 1.0),
        ("romance_language", "is_a", "indo_european_language", 1.0),
        # Semantics.
        ("synonymy", "is_a", "semantic_relation", 1.0),
        ("antonymy", "is_a", "semantic_relation", 1.0),
        ("hyponymy", "is_a", "semantic_relation", 1.0),
        ("hypernymy", "is_a", "semantic_relation", 1.0),
        ("meronymy", "is_a", "semantic_relation", 1.0),
        ("polysemy", "is_a", "semantic_relation", 1.0),
        # Grammar.
        ("tense", "is_a", "grammatical_category", 1.0),
        ("aspect", "is_a", "grammatical_category", 1.0),
        ("mood", "is_a", "grammatical_category", 1.0),
        ("voice", "is_a", "grammatical_category", 1.0),
        ("number", "is_a", "grammatical_category", 1.0),
        ("gender", "is_a", "grammatical_category", 1.0),
        ("case", "is_a", "grammatical_category", 1.0),
    ]


__all__ = ["relations"]
