"""Cognition bus pub/sub, ring-buffer bounds, and throughput."""

from __future__ import annotations

import time

from darwin.mysterio.bus import BusEvent, BusTopic, CognitionBus


def test_publish_and_subscribe_in_process() -> None:
    bus = CognitionBus()
    received: list[BusEvent] = []
    bus.subscribe(BusTopic.TRANSITIONS, received.append)
    bus.publish(BusTopic.TRANSITIONS, {"x": 1}, source="test")
    bus.publish(BusTopic.TRANSITIONS, {"x": 2}, source="test")
    assert len(received) == 2
    assert received[0].payload["x"] == 1
    assert received[1].seq == 2


def test_topic_isolation() -> None:
    bus = CognitionBus()
    got: list[str] = []
    bus.subscribe(BusTopic.PROPOSALS, lambda e: got.append("prop"))
    bus.subscribe(BusTopic.NARRATIVE, lambda e: got.append("narr"))
    bus.publish(BusTopic.PROPOSALS, {})
    assert got == ["prop"]


def test_unsubscribe_stops_delivery() -> None:
    bus = CognitionBus()
    count = {"n": 0}

    def cb(_event: BusEvent) -> None:
        count["n"] += 1

    unsub = bus.subscribe(BusTopic.SIMULATIONS, cb)
    bus.publish(BusTopic.SIMULATIONS, {})
    unsub()
    bus.publish(BusTopic.SIMULATIONS, {})
    assert count["n"] == 1


def test_ring_buffer_bounded() -> None:
    bus = CognitionBus(ring_size=8)
    for i in range(100):
        bus.publish(BusTopic.GATE_HISTORY, {"i": i})
    recent = bus.recent(BusTopic.GATE_HISTORY, limit=50)
    assert len(recent) == 8  # capped by ring size
    assert recent[-1].payload["i"] == 99  # newest retained
    stats = bus.stats()
    assert stats["dropped_estimate"] >= 92


def test_misbehaving_subscriber_does_not_break_publish() -> None:
    bus = CognitionBus()
    good: list[int] = []

    def boom(_e: BusEvent) -> None:
        raise RuntimeError("subscriber failure")

    bus.subscribe(BusTopic.PROPOSALS, boom)
    bus.subscribe(BusTopic.PROPOSALS, lambda e: good.append(e.seq))
    event = bus.publish(BusTopic.PROPOSALS, {})
    assert event.seq == 1
    assert good == [1]


def test_throughput_smoke() -> None:
    """Sustains a high publish rate in-process (smoke, not a hard SLA)."""
    bus = CognitionBus(ring_size=65536)
    sink: list[int] = []
    bus.subscribe(BusTopic.TRANSITIONS, lambda e: sink.append(e.seq))
    n = 20000
    start = time.perf_counter()
    for i in range(n):
        bus.publish(BusTopic.TRANSITIONS, {"i": i})
    elapsed = time.perf_counter() - start
    assert len(sink) == n
    rate = n / elapsed
    # Comfortably above 50k/sec on any modern machine; assert a loose floor.
    assert rate > 20000, f"throughput too low: {rate:.0f}/s"
