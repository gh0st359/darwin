"""First-class snapshot + diff over Darwin's mind.

Snapshots capture the substrate state that self-modification cares about:
causal beliefs (summary), self-model state, world-model variables/hidden
factors, planner overrides, ledger high-water marks, and the identity of
the currently-active accept gate. The diff is structured so an operator
can read what changed between two points in Darwin's evolution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _ulid() -> str:
    """Lexicographically-sortable id with a millisecond timestamp prefix."""
    millis = int(time.time() * 1000)
    return f"{millis:013d}-{uuid.uuid4().hex[:12]}"


@dataclass
class MindSnapshot:
    snapshot_id: str
    created_at: float
    causal: dict[str, Any]
    self_model: dict[str, Any]
    world_model: dict[str, Any]
    planner: dict[str, Any]
    exploration_rate: float
    ledger_marks: dict[str, int]
    gate_identity: str
    self_mod_history_len: int
    generated_modules: dict[str, str] = field(default_factory=dict)
    embedding_checkpoint_hash: str = ""

    @classmethod
    def capture(
        cls,
        darwin: Any,
        *,
        gate_identity: str = "default",
        self_mod_history_len: int = 0,
        ledger_marks: dict[str, int] | None = None,
        generated_modules: dict[str, str] | None = None,
        embedding_checkpoint_hash: str = "",
    ) -> "MindSnapshot":
        causal_model = darwin.causal_model
        self_model = darwin.self_model
        world_model = darwin.world_model
        planner_overrides = getattr(darwin, "_planner_overrides", {}) or {}

        beliefs_summary = [
            {
                "action": belief.action,
                "variable": belief.variable,
                "effect": belief.effect,
                "confidence": float(belief.confidence),
                "samples": int(belief.samples),
            }
            for belief in causal_model.beliefs(limit=64)
        ]

        return cls(
            snapshot_id=_ulid(),
            created_at=time.time(),
            causal={
                "min_samples": int(causal_model.min_samples),
                "total_observations": int(causal_model.total_observations()),
                "beliefs": beliefs_summary,
                "known_actions": list(causal_model.known_actions()),
            },
            self_model={
                "competence_by_action": {
                    name: {
                        "samples": comp.samples,
                        "reward_mean": float(comp.reward_mean),
                        "surprise_mean": float(comp.surprise_mean),
                    }
                    for name, comp in getattr(self_model, "competence_by_action", {}).items()
                },
                "prediction_failures": dict(
                    getattr(self_model, "prediction_failures", {})
                ),
                "known_variables": dict(getattr(self_model, "known_variables", {})),
            },
            world_model={
                "variables": dict(getattr(world_model, "variables", {})),
                "hidden_factors": dict(getattr(world_model, "hidden_factors", {})),
            },
            planner=dict(planner_overrides),
            exploration_rate=float(getattr(darwin, "exploration_rate", 0.0)),
            ledger_marks=dict(ledger_marks or {}),
            gate_identity=gate_identity,
            self_mod_history_len=int(self_mod_history_len),
            generated_modules=dict(generated_modules or {}),
            embedding_checkpoint_hash=embedding_checkpoint_hash,
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> "MindSnapshot":
        return cls(**copy.deepcopy(payload))

    def content_hash(self) -> str:
        payload = json.dumps(self.to_record(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SnapshotDiff:
    a: str
    b: str
    summary: str
    changed: dict[str, dict[str, Any]]
    added: dict[str, Any]
    removed: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _flatten(prefix: str, value: Any, into: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if not value:
            into[prefix or "."] = {}
            return
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, into)
    elif isinstance(value, list):
        into[prefix] = value
    else:
        into[prefix] = value


def diff(a: MindSnapshot, b: MindSnapshot) -> SnapshotDiff:
    flat_a: dict[str, Any] = {}
    flat_b: dict[str, Any] = {}
    _flatten("", a.to_record(), flat_a)
    _flatten("", b.to_record(), flat_b)

    keys_a = set(flat_a)
    keys_b = set(flat_b)
    added_keys = keys_b - keys_a
    removed_keys = keys_a - keys_b
    shared = keys_a & keys_b
    changed = {
        k: {"before": flat_a[k], "after": flat_b[k]}
        for k in shared
        if flat_a[k] != flat_b[k]
        and k not in {"snapshot_id", "created_at"}
    }

    pieces: list[str] = []
    if changed:
        pieces.append(f"{len(changed)} changed")
    if added_keys:
        pieces.append(f"{len(added_keys)} added")
    if removed_keys:
        pieces.append(f"{len(removed_keys)} removed")
    summary = (
        f"snapshot diff {a.snapshot_id} → {b.snapshot_id}: " + ", ".join(pieces)
        if pieces
        else f"snapshot diff {a.snapshot_id} → {b.snapshot_id}: no substantive change"
    )

    return SnapshotDiff(
        a=a.snapshot_id,
        b=b.snapshot_id,
        summary=summary,
        changed=changed,
        added={k: flat_b[k] for k in added_keys},
        removed={k: flat_a[k] for k in removed_keys},
    )


class SnapshotStore:
    """Append-only, ULID-named directory of snapshots.

    Each snapshot is written as a single JSON file. The store keeps an
    in-memory index keyed by snapshot_id for fast lookup. Operators can
    inspect snapshots out-of-band by reading the directory directly.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        if directory is None:
            from darwin.paths import snapshots_dir

            directory = snapshots_dir()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, MindSnapshot] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
                snapshot = MindSnapshot.from_record(payload)
                self._index[snapshot.snapshot_id] = snapshot
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

    def record(self, snapshot: MindSnapshot) -> str:
        path = self.directory / f"{snapshot.snapshot_id}.json"
        path.write_text(json.dumps(snapshot.to_record(), sort_keys=True, default=str))
        self._index[snapshot.snapshot_id] = snapshot
        return snapshot.snapshot_id

    def get(self, snapshot_id: str) -> MindSnapshot | None:
        return self._index.get(snapshot_id)

    def recent(self, limit: int = 20) -> list[MindSnapshot]:
        ordered = sorted(self._index.values(), key=lambda s: s.snapshot_id, reverse=True)
        return ordered[:limit]

    def latest(self) -> MindSnapshot | None:
        recent = self.recent(limit=1)
        return recent[0] if recent else None

    def __len__(self) -> int:
        return len(self._index)
