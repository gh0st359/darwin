"""Tests that TouchRecorder containment actually fires on undeclared writes."""

from __future__ import annotations

import pytest

from darwin.mysterio.safety import ContainmentError, TouchRecorder


class _Stub:
    """Simple object with two attributes the recorder can intercept."""

    def __init__(self) -> None:
        self.declared = 0
        self.undeclared = 0


def test_declared_write_is_recorded() -> None:
    stub = _Stub()
    recorder = TouchRecorder({"stub.declared"})
    recorder.register("stub", stub)
    with recorder:
        stub.declared = 42
    assert stub.declared == 42
    assert any(r.attribute == "declared" for r in recorder.records)


def test_undeclared_write_raises() -> None:
    stub = _Stub()
    recorder = TouchRecorder({"stub.declared"})
    recorder.register("stub", stub)
    with pytest.raises(ContainmentError):
        with recorder:
            stub.undeclared = 99  # not declared in touches


def test_writes_outside_block_are_unguarded() -> None:
    stub = _Stub()
    recorder = TouchRecorder({"stub.declared"})
    recorder.register("stub", stub)
    with recorder:
        stub.declared = 1
    # Outside the with block, writes are unrestricted.
    stub.undeclared = 7
    assert stub.undeclared == 7


def test_multiple_targets_partitioned() -> None:
    a = _Stub()
    b = _Stub()
    recorder = TouchRecorder({"a.declared", "b.declared"})
    recorder.register("a", a)
    recorder.register("b", b)
    with recorder:
        a.declared = 1
        b.declared = 2
    assert a.declared == 1
    assert b.declared == 2


def test_unregistered_object_is_free() -> None:
    a = _Stub()
    free = _Stub()
    recorder = TouchRecorder({"a.declared"})
    recorder.register("a", a)
    # `free` is not registered; writes to it are not intercepted.
    with recorder:
        a.declared = 5
        free.undeclared = 10
    assert free.undeclared == 10
