"""Tests for the cognition bus.

The bus is the cross-subsystem pub/sub transport. v6 ships the in-process
fast path; cross-process fan-out is exercised through ``attach_inbound`` /
``attach_outbound`` here without spawning real subprocesses.
"""

from __future__ import annotations

import queue
import threading
import time

from darwin.mysterio.bus import BusEvent, BusTopic, CognitionBus


def test_publish_and_subscribe_in_process() -> None:
    bus = CognitionBus()
    received: list[BusEvent] = []
    bus.subscribe(BusTopic.TRANSITIONS, received.append)

    event = bus.publish(BusTopic.TRANSITIONS, {"action": "open_curtains"})
    assert event.topic == "transitions"
    assert received == [event]


def test_ring_buffer_bounded_and_drop_estimate_increments() -> None:
    bus = CognitionBus(ring_size=4)
    for i in range(10):
        bus.publish(BusTopic.SIMULATIONS, {"step": i})

    recent = bus.recent(BusTopic.SIMULATIONS, limit=10)
    assert len(recent) == 4
    assert [e.payload["step"] for e in recent] == [6, 7, 8, 9]

    stats = bus.stats()
    assert stats["published"] == 10
    assert stats["dropped_estimate"] >= 6


def test_misbehaving_subscriber_does_not_break_publisher() -> None:
    bus = CognitionBus()

    def bad_sub(_event: BusEvent) -> None:
        raise RuntimeError("subscriber blew up")

    received: list[BusEvent] = []
    bus.subscribe(BusTopic.PROPOSALS, bad_sub)
    bus.subscribe(BusTopic.PROPOSALS, received.append)

    bus.publish(BusTopic.PROPOSALS, {"kind": "parameter"})
    assert len(received) == 1


def test_unsubscribe_stops_callback() -> None:
    bus = CognitionBus()
    received: list[BusEvent] = []
    unsub = bus.subscribe(BusTopic.NARRATIVE, received.append)

    bus.publish(BusTopic.NARRATIVE, {"chunk": "first"})
    assert len(received) == 1

    unsub()
    bus.publish(BusTopic.NARRATIVE, {"chunk": "second"})
    assert len(received) == 1


def test_string_topics_interoperate_with_enum() -> None:
    bus = CognitionBus()
    received_enum: list[BusEvent] = []
    received_str: list[BusEvent] = []
    bus.subscribe(BusTopic.CODE_GEN, received_enum.append)
    bus.subscribe("code_gen", received_str.append)

    bus.publish("code_gen", {"sha": "abc"})
    bus.publish(BusTopic.CODE_GEN, {"sha": "def"})

    assert len(received_enum) == 2
    assert len(received_str) == 2


def test_cross_process_drain_via_queue() -> None:
    """attach_inbound/outbound move events between two CognitionBus instances
    via a shared mp.Queue substitute (a plain queue.Queue works for the
    in-process simulation)."""

    transport: "queue.Queue[dict]" = queue.Queue()
    sender = CognitionBus()
    sender.attach_outbound(transport)
    receiver = CognitionBus()
    received: list[BusEvent] = []
    receiver.subscribe(BusTopic.SUBSYSTEM_HEALTH, received.append)
    receiver.attach_inbound(transport)

    sender.publish(BusTopic.SUBSYSTEM_HEALTH, {"name": "kernel", "alive": True})

    deadline = time.time() + 1.0
    while time.time() < deadline and not received:
        time.sleep(0.01)

    receiver.stop()
    assert received, "receiver should have observed at least one event"
    assert received[0].payload["name"] == "kernel"


def test_thread_safety_under_concurrent_publish() -> None:
    bus = CognitionBus()
    received: list[BusEvent] = []
    lock = threading.Lock()

    def collect(event: BusEvent) -> None:
        with lock:
            received.append(event)

    bus.subscribe(BusTopic.TRANSITIONS, collect)

    def producer(start: int) -> None:
        for i in range(50):
            bus.publish(BusTopic.TRANSITIONS, {"i": start + i})

    threads = [threading.Thread(target=producer, args=(p * 1000,)) for p in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(received) == 200
    # Sequence numbers monotonic per bus
    assert bus.stats()["published"] == 200
