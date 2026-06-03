"""Dataset loaders for frontier benchmarks.

Each loader looks under ``data_dir() / "benchmarks" / <name>`` for the
benchmark's fixture data. If absent, the loader returns a ``LoadResult``
with ``status="not_provisioned"`` and an empty item list. Frontier tasks
turn ``not_provisioned`` into a score of 0 with an explanatory evidence
record. **No network fetch, no surprise downloads.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darwin.paths import data_dir


@dataclass
class LoadResult:
    """The outcome of attempting to load a benchmark dataset."""

    name: str
    status: str  # "loaded" | "not_provisioned" | "load_failed"
    items: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"


def _benchmarks_root() -> Path:
    return data_dir() / "benchmarks"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                items.append(item)
    return items


def load_benchmark(name: str, *, fixture_file: str = "tasks.jsonl") -> LoadResult:
    """Load a benchmark by name. Returns a LoadResult describing the outcome.

    Supports two on-disk shapes:
      ``benchmarks/<name>/tasks.jsonl``  — newline-delimited JSON.
      ``benchmarks/<name>/tasks.json``   — JSON array of objects.
    """

    root = _benchmarks_root() / name
    jsonl_path = root / fixture_file
    json_path = root / "tasks.json"
    if not root.exists():
        return LoadResult(name=name, status="not_provisioned")
    try:
        if jsonl_path.exists():
            items = _load_jsonl(jsonl_path)
        elif json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                items = json.load(handle)
            if not isinstance(items, list):
                return LoadResult(
                    name=name, status="load_failed",
                    error=f"{json_path.name} is not a JSON array",
                )
        else:
            return LoadResult(name=name, status="not_provisioned")
    except OSError as e:
        return LoadResult(name=name, status="load_failed", error=str(e))
    return LoadResult(name=name, status="loaded", items=items)


__all__ = ["LoadResult", "load_benchmark"]
