"""Persistence — Darwin's universe survives restarts.

The ConceptUniverse lives in memory. Without persistence, every brain
restart wipes Darwin's accumulated knowledge from previous chat sessions
— a frontier system has to keep what it learned.

This module serializes a ConceptUniverse to a JSON file and loads it
back on startup. Loading is *additive*: an existing universe (with the
primitive seed already loaded) accepts the persisted concepts and
relations on top of the seed. Concepts already present are enriched
(definition, aliases, examples, salience merged) rather than replaced.

The save format is plain JSON: a top-level object with ``concepts``
(name → record) and ``relations`` (list of triples + kind). Forward-
compatible: unknown fields are ignored on load. Backward-compatible: a
missing field falls back to the default.

The runtime calls ``save_universe(runtime.universe, path)`` after every
chat turn that grew the universe and on shutdown, and calls
``load_universe(runtime.universe, path)`` once during construction.
Both functions are safe to call repeatedly.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from darwin.universe.concept_universe import ConceptUniverse


def save_universe(universe: ConceptUniverse, path: str | Path) -> bool:
    """Atomically write the universe to ``path``. Returns True on success."""

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    payload: dict[str, Any] = {
        "version": 1,
        "saved_at": time.time(),
        "concepts": {
            c.name: c.to_record() for c in universe.all_concepts()
        },
        "domains": [d.to_record() for d in universe.domains()],
        "relations": [r.to_record() for r in universe.relations()],
        "summary": universe.summary(),
    }
    try:
        # Atomic write: temp file then rename.
        fd, tmp_path = tempfile.mkstemp(
            prefix="universe_", suffix=".json", dir=str(target.parent),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(tmp_path, target)
        return True
    except OSError:
        return False


def load_universe(universe: ConceptUniverse, path: str | Path) -> int:
    """Load concepts/relations from ``path`` into ``universe``.

    Returns the number of relations added. Concepts already present are
    enriched (definition, aliases, examples merged) but never replaced.
    Returns 0 if the file does not exist or is malformed (no error;
    universe is left untouched).
    """

    source = Path(path)
    if not source.exists():
        return 0
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    # Concepts first.
    concepts = payload.get("concepts", {}) or {}
    if isinstance(concepts, dict):
        for name, record in concepts.items():
            try:
                universe.add_concept(
                    name,
                    domain=record.get("domain", "general"),
                    definition=record.get("definition", ""),
                    depth=int(record.get("depth", 0) or 0),
                    aliases=tuple(record.get("aliases") or ()),
                    examples=tuple(record.get("examples") or ()),
                    derived_from=tuple(record.get("derived_from") or ()),
                    salience=float(record.get("salience", 1.0) or 1.0),
                )
            except Exception:
                continue
    # Then relations.
    relations_added = 0
    relations = payload.get("relations", []) or []
    if isinstance(relations, list):
        for record in relations:
            if not isinstance(record, dict):
                continue
            try:
                source_name = record.get("source", "")
                target_name = record.get("target", "")
                kind = record.get("kind", "related_to")
                if not (source_name and target_name):
                    continue
                # Skip if the typed edge already exists (idempotent).
                already = any(
                    r.target == target_name and r.kind == kind
                    for r in universe.neighbors(source_name) if universe.has(source_name)
                )
                if already:
                    continue
                universe.add_relation(
                    source_name, target_name, kind,
                    weight=float(record.get("weight", 1.0) or 1.0),
                    notes=record.get("notes", ""),
                    ensure_concepts=True,
                )
                relations_added += 1
            except Exception:
                continue
    return relations_added


def default_universe_path(memory_path: str | Path | None = None) -> Path:
    """Conventional persistence path next to the sqlite memory file.

    When no ``memory_path`` is supplied, the universe file lands at the
    default location resolved through ``darwin.paths`` (which respects the
    ``DARWIN_DATA_DIR`` environment variable used by the test harness).
    When ``memory_path`` is supplied, the universe file sits next to it
    with a ``_universe.json`` suffix so multiple memory files coexist
    without colliding.
    """

    if memory_path is None:
        from darwin.paths import universe_path

        return universe_path()
    p = Path(memory_path)
    if p.parent == Path():
        return Path(p.stem + "_universe.json")
    return p.parent / (p.stem + "_universe.json")
