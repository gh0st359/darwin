"""Tests for universe persistence."""

from __future__ import annotations

from pathlib import Path

from darwin.universe.concept_universe import ConceptUniverse
from darwin.universe.persistence import (
    default_universe_path,
    load_universe,
    save_universe,
)
from darwin.universe.primitive_seed import seed_primitives


def test_round_trip_preserves_concepts_and_relations(tmp_path: Path) -> None:
    u = ConceptUniverse()
    u.add_concept("dog", domain="bio", definition="a domesticated mammal")
    u.add_concept("mammal", domain="bio")
    u.add_relation("dog", "mammal", "is_a")
    path = tmp_path / "universe.json"
    assert save_universe(u, path)

    loaded = ConceptUniverse()
    seed_primitives(loaded)
    n = load_universe(loaded, path)
    assert n >= 1
    assert loaded.has("dog")
    assert loaded.has("mammal")
    dog = loaded.get("dog")
    assert dog is not None
    assert dog.definition == "a domesticated mammal"
    assert any(
        r.target == "mammal" and r.kind == "is_a"
        for r in loaded.neighbors("dog")
    )


def test_load_is_idempotent(tmp_path: Path) -> None:
    u = ConceptUniverse()
    u.add_concept("a")
    u.add_concept("b")
    u.add_relation("a", "b", "is_a")
    path = tmp_path / "universe.json"
    save_universe(u, path)
    loaded = ConceptUniverse()
    first = load_universe(loaded, path)
    second = load_universe(loaded, path)
    # The first load added relations; the second should be a no-op.
    assert first >= 1
    assert second == 0


def test_load_nonexistent_returns_zero(tmp_path: Path) -> None:
    u = ConceptUniverse()
    n = load_universe(u, tmp_path / "does-not-exist.json")
    assert n == 0


def test_load_malformed_file_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "garbage.json"
    path.write_text("{not valid json")
    u = ConceptUniverse()
    n = load_universe(u, path)
    assert n == 0


def test_load_enriches_existing_concept_without_replacing(tmp_path: Path) -> None:
    u = ConceptUniverse()
    u.add_concept("dog", definition="primary")
    path = tmp_path / "universe.json"
    save_universe(u, path)
    loaded = ConceptUniverse()
    loaded.add_concept("dog", definition="alternate")  # already there
    load_universe(loaded, path)
    # First-write wins; load is enriching, not replacing.
    dog = loaded.get("dog")
    assert dog is not None
    assert dog.definition == "alternate"


def test_default_universe_path_matches_memory_path(tmp_path: Path) -> None:
    memory = tmp_path / "darwin_memory.sqlite3"
    expected = tmp_path / "darwin_memory_universe.json"
    assert default_universe_path(memory) == expected


def test_save_and_reload_with_primitives_keeps_primitives_intact(
    tmp_path: Path,
) -> None:
    u = ConceptUniverse()
    seed_primitives(u)
    u.add_concept("dog")
    u.add_relation("dog", "thing", "is_a")
    path = tmp_path / "universe.json"
    save_universe(u, path)

    fresh = ConceptUniverse()
    seed_primitives(fresh)
    load_universe(fresh, path)
    assert fresh.has("dog")
    assert fresh.has("thing")
    assert any(
        r.target == "thing" and r.kind == "is_a"
        for r in fresh.neighbors("dog")
    )
