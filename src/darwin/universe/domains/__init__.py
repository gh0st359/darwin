"""Curated domain seeds — loaded into the ConceptUniverse on request.

Each domain module exports a ``relations()`` callable that returns a list
of ``(source, kind, target, weight)`` tuples. The KnowledgeSeeder reads
them and folds them into a fresh universe so Darwin starts a session
with non-trivial knowledge to reason over.

The data is not pretrained weights — it is symbolic facts the operator
deliberately seeds. Darwin can extend, contradict, or supersede any of
them through ingestion.
"""

from __future__ import annotations

from typing import Callable

from darwin.universe.domains import (
    biology,
    chemistry,
    computing,
    geography,
    linguistics,
    math_domain,
    physics,
)


_REGISTRY: dict[str, Callable[[], list[tuple[str, str, str, float]]]] = {
    "biology": biology.relations,
    "chemistry": chemistry.relations,
    "computing": computing.relations,
    "geography": geography.relations,
    "linguistics": linguistics.relations,
    "math": math_domain.relations,
    "physics": physics.relations,
}


def domains_available() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_domain(name: str) -> list[tuple[str, str, str, float]]:
    fn = _REGISTRY.get(name)
    if fn is None:
        return []
    return list(fn())


def load_all() -> list[tuple[str, str, str, float]]:
    out: list[tuple[str, str, str, float]] = []
    for name in domains_available():
        out.extend(load_domain(name))
    return out


__all__ = ["domains_available", "load_all", "load_domain"]
