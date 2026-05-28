"""Typed proposal grammar for mysterio self-modification.

`MutationKind` enumerates the surfaces self-modification can touch.
`SafetyTier` carries informational metadata per kind — recorded with every
proposal for operator inspection. Tiers do not block activation; the only
runtime invariant is `ContainmentError`, which makes rollback tractable
at scale by forcing each apply to declare what it will touch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class MutationKind(str, Enum):
    PARAMETER = "parameter"
    RULE = "rule"
    KERNEL = "kernel"
    GATE = "gate"
    LEDGER = "ledger"
    MODULE = "module"
    SUBSYSTEM = "subsystem"


@dataclass(frozen=True)
class SafetyTier:
    """Informational metadata about a mutation kind.

    `default_validations` is the suggested number of holdout evaluations for
    a proposal of this kind; the live gate is free to ignore it. `notes` is
    free-text for operator review. No fields here block activation.
    """

    name: str
    kind: MutationKind
    default_validations: int
    notes: str
    inspection_topic: str = ""


SAFETY_BOUNDS: dict[MutationKind, SafetyTier] = {
    MutationKind.PARAMETER: SafetyTier(
        name="parameter",
        kind=MutationKind.PARAMETER,
        default_validations=1,
        notes="scalar tweak; immediate; one holdout test",
        inspection_topic="self_modification",
    ),
    MutationKind.RULE: SafetyTier(
        name="rule",
        kind=MutationKind.RULE,
        default_validations=1,
        notes="new derived rule or precondition; immediate",
        inspection_topic="self_modification",
    ),
    MutationKind.KERNEL: SafetyTier(
        name="kernel",
        kind=MutationKind.KERNEL,
        default_validations=1,
        notes="new kernel job or scheduler change; tagged for operator review",
        inspection_topic="quarantine",
    ),
    MutationKind.GATE: SafetyTier(
        name="gate",
        kind=MutationKind.GATE,
        default_validations=1,
        notes="replaces the accept-gate function; shadow-tested for inspection",
        inspection_topic="quarantine",
    ),
    MutationKind.LEDGER: SafetyTier(
        name="ledger",
        kind=MutationKind.LEDGER,
        default_validations=1,
        notes="schema or persistence path change; tagged for operator review",
        inspection_topic="quarantine",
    ),
    MutationKind.MODULE: SafetyTier(
        name="module",
        kind=MutationKind.MODULE,
        default_validations=1,
        notes="generates a new Python module on disk; rollback removes the file",
        inspection_topic="quarantine",
    ),
    MutationKind.SUBSYSTEM: SafetyTier(
        name="subsystem",
        kind=MutationKind.SUBSYSTEM,
        default_validations=1,
        notes="registers a new long-running cognitive subsystem",
        inspection_topic="quarantine",
    ),
}


INSPECTION_KINDS: set[MutationKind] = {
    kind
    for kind, tier in SAFETY_BOUNDS.items()
    if tier.inspection_topic == "quarantine"
}


class ContainmentError(Exception):
    """Raised when `apply()` writes to a target outside its declared `touches`.

    This is the only hard runtime invariant in mysterio. It exists so that
    snapshots + rollback remain tractable as the self-modification surface
    grows: rollback only works if we know what each apply touched.
    """


@dataclass
class TouchRecord:
    target: str
    attribute: str
    old: Any
    new: Any


class TouchRecorder:
    """Context manager that records writes to declared targets.

    Each `target` is an arbitrary object exposed via `register(name, obj)`.
    During the `with` block, attribute writes to those objects are intercepted:
    declared writes (their attribute path appears in `touches`) are recorded
    for rollback; undeclared writes raise `ContainmentError`.

    The recorder is permissive about reads and about writes to attributes
    of objects not registered (so apply functions can freely mutate locals
    or call methods that don't touch the substrate).

    Limitations: this is a structural guard, not a sandbox. An apply that
    bypasses attribute access (e.g., monkey-patches sys.modules) will not be
    caught. The recorder is enough for the typed-grammar use case where
    proposals declare their write surface honestly.
    """

    def __init__(self, touches: set[str]) -> None:
        self.touches: set[str] = set(touches)
        self._targets: dict[str, Any] = {}
        self._records: list[TouchRecord] = []
        self._originals: dict[int, dict[str, Any]] = {}
        self._active: bool = False

    def register(self, name: str, obj: Any) -> None:
        self._targets[name] = obj

    @property
    def records(self) -> list[TouchRecord]:
        return list(self._records)

    def __enter__(self) -> "TouchRecorder":
        self._active = True
        for name, obj in self._targets.items():
            self._install_interceptor(name, obj)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._active = False
        for obj in list(self._targets.values()):
            self._uninstall_interceptor(obj)

    def _install_interceptor(self, name: str, obj: Any) -> None:
        recorder = self
        original_setattr = type(obj).__setattr__

        def intercepted(target: Any, key: str, value: Any) -> None:
            if recorder._active and id(target) in recorder._originals:
                path = f"{name}.{key}"
                if path not in recorder.touches:
                    raise ContainmentError(
                        f"undeclared write to {path!r} (declared touches: "
                        f"{sorted(recorder.touches)})"
                    )
                old = getattr(target, key, _MISSING)
                recorder._records.append(
                    TouchRecord(target=name, attribute=key, old=old, new=value)
                )
            original_setattr(target, key, value)

        # Per-instance interception by swapping __class__ to a subclass.
        cls = type(obj)
        interceptor_cls = type(
            f"_TouchRecorded_{cls.__name__}",
            (cls,),
            {"__setattr__": intercepted},
        )
        self._originals[id(obj)] = {"class": cls, "interceptor": interceptor_cls}
        try:
            obj.__class__ = interceptor_cls
        except TypeError:
            # Built-in or __slots__ instances cannot have __class__ reassigned;
            # silently skip — those objects cannot be intercepted, but the
            # declared-touches grammar still gives the rollback layer a manifest.
            self._originals.pop(id(obj), None)

    def _uninstall_interceptor(self, obj: Any) -> None:
        info = self._originals.pop(id(obj), None)
        if info is None:
            return
        try:
            obj.__class__ = info["class"]
        except TypeError:
            pass


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


@contextmanager
def recorder_for(touches: set[str], **targets: Any) -> Iterator[TouchRecorder]:
    """Convenience: build a recorder and register targets in one call."""
    rec = TouchRecorder(touches)
    for name, obj in targets.items():
        rec.register(name, obj)
    with rec:
        yield rec
