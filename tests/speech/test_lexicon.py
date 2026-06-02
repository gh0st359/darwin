"""Tests for the growing CCG lexicon."""

from __future__ import annotations

from pathlib import Path

from darwin.speech.lexicon import CCGLexicon, default_lexicon_path


def test_core_function_words_seeded() -> None:
    lex = CCGLexicon()
    # Determiners, copulas, conjunctions seeded at construction.
    assert lex.lookup("the")
    assert lex.lookup("is")
    assert lex.lookup("and")


def test_register_creates_new_entry() -> None:
    lex = CCGLexicon()
    lex.register(surface="photon", category="N", concept="photon", pos="NN")
    entries = lex.lookup("photon")
    assert entries
    assert entries[0].concept == "photon"


def test_register_increments_frequency_on_duplicate() -> None:
    lex = CCGLexicon()
    lex.register(surface="alpha", category="N", concept="alpha")
    lex.register(surface="alpha", category="N", concept="alpha")
    entries = lex.lookup("alpha")
    assert entries[0].frequency >= 2


def test_register_distinct_categories_create_separate_entries() -> None:
    lex = CCGLexicon()
    lex.register(surface="run", category="N", concept="run")
    lex.register(surface="run", category="S\\NP", concept="running")
    entries = lex.lookup("run")
    cats = {e.category for e in entries}
    assert "N" in cats
    assert "S\\NP" in cats


def test_surface_for_concept_returns_preferred_form() -> None:
    lex = CCGLexicon()
    lex.register(surface="photon", concept="photon")
    assert lex.surface_for_concept("photon") == "photon"


def test_surface_for_concept_falls_back_to_humanised_name() -> None:
    lex = CCGLexicon()
    assert lex.surface_for_concept("free_will") == "free will"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    lex = CCGLexicon()
    lex.register(surface="quark", category="N", concept="quark", pos="NN")
    path = tmp_path / "lex.json"
    lex.save(path)
    lex2 = CCGLexicon()
    added = lex2.load(path)
    assert added >= 1
    assert lex2.lookup("quark")


def test_load_is_idempotent(tmp_path: Path) -> None:
    lex = CCGLexicon()
    lex.register(surface="zed", category="N", concept="zed")
    path = tmp_path / "lex.json"
    lex.save(path)
    lex2 = CCGLexicon()
    lex2.load(path)
    second = lex2.load(path)
    # The second load shouldn't double-register.
    assert second == 0


def test_default_lexicon_path_uses_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    assert default_lexicon_path() == tmp_path / "darwin_lexicon.json"


def test_empty_surface_rejected() -> None:
    lex = CCGLexicon()
    import pytest
    with pytest.raises(ValueError):
        lex.register(surface="", category="N")
