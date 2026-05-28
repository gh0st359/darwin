"""Tag-and-inspect register for substrate-touching self-modifications.

`QuarantineQueue` records every `KERNEL`/`GATE`/`LEDGER`/`MODULE`/`SUBSYSTEM`
mutation that has been applied so the operator can inspect or roll it back
later. The queue does NOT block activation — mutations apply immediately;
the entry is the inspection handle and the snapshot pointer.

Rollback restores the pre-apply snapshot via the supplied revert callable
plus a reverse-replay of the `TouchRecorder.records` if needed.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from darwin.mysterio.safety import INSPECTION_KINDS, MutationKind


class QuarantineStatus(str, Enum):
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"


@dataclass
class QuarantineEntry:
    entry_id: str
    proposal_id: str
    kind: MutationKind
    description: str
    snapshot_id: str
    submitted_at: float
    status: QuarantineStatus = QuarantineStatus.APPLIED
    rolled_back_at: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value if isinstance(self.kind, MutationKind) else str(self.kind)
        data["status"] = self.status.value if isinstance(self.status, QuarantineStatus) else str(self.status)
        return data


RollbackHandler = Callable[[QuarantineEntry], None]


class QuarantineQueue:
    """In-memory + persisted (via callback) register of substrate mutations."""

    def __init__(
        self,
        persist: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._entries: dict[str, QuarantineEntry] = {}
        self._lock = threading.RLock()
        self._handlers: dict[MutationKind, RollbackHandler] = {}
        self._persist = persist

    def register_handler(self, kind: MutationKind, handler: RollbackHandler) -> None:
        self._handlers[kind] = handler

    def submit(
        self,
        proposal_id: str,
        kind: MutationKind,
        description: str,
        snapshot_id: str,
        notes: str = "",
        extra: dict[str, Any] | None = None,
    ) -> QuarantineEntry:
        """Record a substrate mutation. Returns the entry handle.

        Only kinds in `INSPECTION_KINDS` are recorded. Calls with parameter
        or rule kinds are no-ops so callers can submit unconditionally.
        """
        if kind not in INSPECTION_KINDS:
            return QuarantineEntry(
                entry_id="",
                proposal_id=proposal_id,
                kind=kind,
                description=description,
                snapshot_id=snapshot_id,
                submitted_at=time.time(),
                notes="not-inspected (parameter/rule kind)",
            )

        entry = QuarantineEntry(
            entry_id=uuid.uuid4().hex,
            proposal_id=proposal_id,
            kind=kind,
            description=description,
            snapshot_id=snapshot_id,
            submitted_at=time.time(),
            notes=notes,
            extra=dict(extra or {}),
        )
        with self._lock:
            self._entries[entry.entry_id] = entry
        if self._persist is not None:
            self._persist(entry.to_record())
        return entry

    def rollback(self, entry_id: str) -> QuarantineEntry | None:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return None
            handler = self._handlers.get(entry.kind)
            if handler is not None:
                handler(entry)
            entry.status = QuarantineStatus.ROLLED_BACK
            entry.rolled_back_at = time.time()
        if self._persist is not None:
            self._persist(entry.to_record())
        return entry

    def get(self, entry_id: str) -> QuarantineEntry | None:
        with self._lock:
            return self._entries.get(entry_id)

    def recent(self, limit: int = 20) -> list[QuarantineEntry]:
        with self._lock:
            ordered = sorted(
                self._entries.values(), key=lambda e: e.submitted_at, reverse=True
            )
            return ordered[:limit]

    def pending(self) -> list[QuarantineEntry]:
        with self._lock:
            return [
                e for e in self._entries.values() if e.status == QuarantineStatus.APPLIED
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
