"""Filesystem modality: source-tree changes as observable state.

Watches a directory tree (default: ``src/``). Each poll diffs the prior set of
file SHAs against the current set; new/changed/removed files become
transitions whose ``action`` is the change kind and whose ``before``/``after``
encode the file's role. Polling is intentionally simple — no inotify, no
watchdog dep — so it works in any environment.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darwin.types import Transition


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


@dataclass
class CodeModalityAdapter:
    root: str | Path = "src"
    track: str = "grounded"
    pattern_suffixes: tuple[str, ...] = (".py",)
    _index: dict[str, str] = field(default_factory=dict)
    _t: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.exists():
            self.active = False

    def scan(self) -> list[Transition]:
        if not self.active:
            return []
        current: dict[str, str] = {}
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if self.pattern_suffixes and path.suffix not in self.pattern_suffixes:
                continue
            current[str(path)] = _file_sha256(path)
        new_files = set(current) - set(self._index)
        removed_files = set(self._index) - set(current)
        changed = {p for p in (set(current) & set(self._index)) if current[p] != self._index[p]}
        transitions: list[Transition] = []
        for p in new_files:
            transitions.append(self._make("added", p, after_sha=current[p]))
        for p in removed_files:
            transitions.append(self._make("removed", p, before_sha=self._index[p]))
        for p in changed:
            transitions.append(self._make("changed", p, before_sha=self._index[p], after_sha=current[p]))
        self._index = current
        return transitions

    def _make(
        self,
        kind: str,
        path: str,
        *,
        before_sha: str = "",
        after_sha: str = "",
    ) -> Transition:
        self._t += 1
        return Transition(
            before={"path": path, "sha": before_sha},
            action=f"code:{kind}",
            after={"path": path, "sha": after_sha},
            reward=0.0,
            t=self._t,
            metadata={"track": self.track, "modality": "code"},
        )

    def status(self) -> dict[str, Any]:
        return {
            "modality": "code",
            "root": str(self.root),
            "active": self.active,
            "tracked_files": len(self._index),
            "track": self.track,
        }
