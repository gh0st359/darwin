"""Daemon + client for running Darwin as a 24/7 brain process.

`darwin brain` runs in one terminal and keeps Darwin's 5 background
cognition loops alive. `darwin connect` runs in another terminal and
gives you a chat REPL that subscribes to the brain's live thought
stream — so you watch the mind think even when you're not typing,
and any /command you'd give in single-process `live` mode works
across the socket.
"""

from __future__ import annotations

import json
import queue
import selectors
import socket
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from darwin.runtime import DarwinRuntime, RuntimeEvent


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9870


def _serialize(payload: Any) -> str:
    def _default(value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    return json.dumps(payload, default=_default)


@dataclass
class Subscriber:
    """A single connected client and its outbound queue."""

    handle: socket.socket
    out_queue: "queue.Queue[str]" = field(default_factory=lambda: queue.Queue(maxsize=512))
    alive: threading.Event = field(default_factory=threading.Event)
    address: str = ""

    def send(self, payload: dict[str, Any]) -> None:
        if not self.alive.is_set():
            return
        try:
            self.out_queue.put_nowait(_serialize(payload))
        except queue.Full:
            # Drop oldest to keep up with bursts
            try:
                self.out_queue.get_nowait()
                self.out_queue.put_nowait(_serialize(payload))
            except queue.Empty:
                pass


class DarwinDaemon:
    """24/7 brain server. One process owns Darwin; many clients can attach."""

    def __init__(
        self,
        runtime: DarwinRuntime,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self._server: socketserver.ThreadingTCPServer | None = None
        self._subscribers: list[Subscriber] = []
        self._subscribers_lock = threading.RLock()
        self._stop = threading.Event()
        self._original_sink = runtime.event_sink
        runtime.event_sink = self._on_event

    @property
    def running(self) -> bool:
        return self._server is not None and not self._stop.is_set()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.runtime.start()
        handler = self._make_handler()
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self._server = socketserver.ThreadingTCPServer((self.host, self.port), handler)
        thread = threading.Thread(
            target=self._server.serve_forever,
            name="darwin-server",
            daemon=True,
        )
        thread.start()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.alive.clear()
            try:
                subscriber.handle.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self.runtime.stop()

    def serve_forever(self) -> None:
        """Block in the foreground until interrupted."""

        self.start()
        try:
            while not self._stop.wait(1.0):
                continue
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # -- event broadcasting --------------------------------------------

    def _on_event(self, event: RuntimeEvent) -> None:
        if self._original_sink is not None:
            try:
                self._original_sink(event)
            except Exception:
                pass
        payload = {
            "type": "event",
            "kind": event.kind,
            "loop": event.loop,
            "content": event.content,
            "payload": event.payload,
            "timestamp": event.timestamp,
        }
        with self._subscribers_lock:
            for subscriber in list(self._subscribers):
                subscriber.send(payload)

    # -- per-connection handler ----------------------------------------

    def _make_handler(self):
        daemon = self

        class _Handler(socketserver.StreamRequestHandler):
            timeout = None

            def handle(self) -> None:
                subscriber = Subscriber(handle=self.request, address=str(self.client_address))
                subscriber.alive.set()
                with daemon._subscribers_lock:
                    daemon._subscribers.append(subscriber)
                writer = threading.Thread(
                    target=daemon._writer_loop,
                    args=(subscriber,),
                    name=f"darwin-writer-{self.client_address[1]}",
                    daemon=True,
                )
                writer.start()
                subscriber.send(
                    {
                        "type": "welcome",
                        "brain": "darwin",
                        "loops": list(daemon.runtime.loop_intervals),
                        "running": daemon.runtime.running,
                    }
                )
                try:
                    daemon._reader_loop(subscriber)
                finally:
                    subscriber.alive.clear()
                    try:
                        subscriber.out_queue.put_nowait("")  # wake writer
                    except queue.Full:
                        pass
                    with daemon._subscribers_lock:
                        if subscriber in daemon._subscribers:
                            daemon._subscribers.remove(subscriber)

        return _Handler

    def _writer_loop(self, subscriber: Subscriber) -> None:
        while subscriber.alive.is_set():
            try:
                message = subscriber.out_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not message:
                break
            try:
                subscriber.handle.sendall((message + "\n").encode("utf-8"))
            except OSError:
                subscriber.alive.clear()
                break

    def _reader_loop(self, subscriber: Subscriber) -> None:
        buffer = b""
        while subscriber.alive.is_set() and not self._stop.is_set():
            try:
                chunk = subscriber.handle.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                self._dispatch(subscriber, text)

    # -- command dispatch ----------------------------------------------

    def _dispatch(self, subscriber: Subscriber, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            subscriber.send({"type": "error", "message": f"bad json: {exc}"})
            return

        cmd = str(message.get("cmd", "")).lower()
        request_id = message.get("id")

        try:
            if cmd == "chat":
                text = str(message.get("message", ""))
                response = self.runtime.chat(text)
                plan = self.runtime.last_response_plan
                subscriber.send(
                    {
                        "type": "response",
                        "id": request_id,
                        "text": response,
                        "plan": plan.to_record() if plan is not None else None,
                    }
                )
            elif cmd == "ping":
                subscriber.send({"type": "pong", "id": request_id, "ts": time.time()})
            elif cmd == "command":
                lines = self._handle_named_command(str(message.get("command", "")))
                subscriber.send({"type": "command_result", "id": request_id, "lines": lines})
            elif cmd == "shutdown":
                subscriber.send({"type": "shutting_down", "id": request_id})
                threading.Thread(target=self.stop, daemon=True).start()
            else:
                subscriber.send({"type": "error", "id": request_id, "message": f"unknown cmd: {cmd}"})
        except Exception as exc:  # pragma: no cover - defensive
            subscriber.send({"type": "error", "id": request_id, "message": repr(exc)})

    def _handle_named_command(self, command: str) -> list[str]:
        """Re-implement a subset of the live-CLI commands for remote clients."""

        runtime = self.runtime
        command = command.strip()
        if not command.startswith("/"):
            return [f"unknown command: {command!r}"]
        parts = command.split()
        head = parts[0].lower()

        if head == "/status":
            lines = list(runtime.darwin.self_report().lines())
            lines.append(runtime.darwin.world_model.summary(runtime.darwin.causal_model))
            lines.append(runtime.darwin.semantic_memory.summary())
            if runtime.store is not None:
                lines.append(f"storage={runtime.store.counts()}")
            return lines
        if head == "/beliefs":
            beliefs = runtime.darwin.causal_model.beliefs(limit=15)
            if not beliefs:
                return ["No grounded causal beliefs yet."]
            return [
                (
                    f"- if {b.condition}: {b.action} -> {b.variable} {b.effect} "
                    f"conf={b.confidence:.2f} n={b.samples}"
                )
                for b in beliefs
            ]
        if head == "/concepts":
            return [
                f"- L{c.level} {c.kind}: {c.name} support={c.support} rmean={c.reward_mean:.2f}"
                for c in runtime.darwin.memory.concepts.hierarchy(limit=25)
            ]
        if head == "/experiments":
            proposals = runtime.darwin.experiment_engine.propose(
                runtime.adapter.observe(), runtime.adapter.possible_actions(), goal=runtime.goal, limit=8
            )
            return [f"- {p.question} [{p.rationale}]" for p in proposals] or ["no proposals"]
        if head == "/think":
            event = runtime.cognition_cycle()
            return [f"{event.kind}: {event.content}"]
        if head == "/dream":
            event = runtime.dream()
            return [event.content]
        if head == "/simulate":
            snapshot = runtime.run_simulation()
            if snapshot is None:
                return ["no simulation produced"]
            out = [
                f"chain confidence={snapshot.get('chain_confidence', 0):.3f}",
                f"chain uncertainty={snapshot.get('chain_uncertainty', 0):.3f}",
                f"total expected reward={snapshot.get('total_expected_reward', 0):.3f}",
            ]
            for node in snapshot.get("nodes", [])[:6]:
                out.append(
                    f"- step {node['step']}: {node['action']} conf={node['confidence']:.2f}"
                )
            return out
        if head == "/selfmod":
            outcomes = runtime.run_self_modification()
            if not outcomes:
                return ["no self-modification proposals this cycle"]
            return [
                (
                    f"- [{'accepted' if o.accepted else 'rejected'}] {o.proposal.kind} "
                    f"gain={o.improvement:.4f} rationale={o.proposal.rationale}"
                )
                for o in outcomes
            ]
        if head == "/uncertainty":
            scan = runtime.last_uncertainty_scan
            if scan is None:
                return ["no uncertainty scan yet"]
            return [
                f"- {item['action']:>20} unc={item['uncertainty']:.2f}"
                for item in scan.get("scan", [])[:10]
            ]
        if head == "/loops":
            out = ["background loops:"]
            for name, interval in runtime.loop_intervals.items():
                state = runtime._loop_state.get(name, {})
                last = state.get("last_event", "n/a")
                out.append(f"- {name:<18} interval={interval:.1f}s last={last}")
            return out
        if head == "/causal-graph":
            graph = runtime.darwin.planner.chain_engine.graph(min_confidence=0.0, limit=80)
            out = [f"actions={len(graph.actions)} variables={len(graph.variables)} edges={len(graph.edges)}"]
            for edge in graph.edges[:20]:
                out.append(
                    f"- {edge.source_action} -> {edge.variable} effect={edge.effect} "
                    f"conf={edge.confidence:.2f} n={edge.samples}"
                )
            return out
        if head == "/dlm":
            render = runtime.last_render
            out = [f"current DLM: {runtime.dlm.name}"]
            if render is None:
                out.append("no render in this session yet.")
            else:
                out.append(f"renderer={render.renderer} valid={render.valid} duration={render.duration_ms:.1f}ms")
                for note in render.validation_notes[:5]:
                    out.append(f"- note: {note}")
            return out
        if head == "/training":
            summary = runtime.training_collector.summary()
            out = [
                f"training pairs collected={summary['total']} accepted={summary['accepted']} "
                f"path={summary['path']}"
            ]
            for renderer, count in summary["by_renderer"].items():
                out.append(f"- {renderer}: {count}")
            return out
        if head == "/metrics":
            snapshot = runtime.logger.snapshot()
            out = ["metrics:"] + [f"- {k}: {v}" for k, v in snapshot["metrics"].items()]
            if snapshot["counters"]:
                out.append("counters:")
                out.extend(f"- {k}: {v}" for k, v in snapshot["counters"].items())
            return out
        if head == "/thoughts":
            trace = runtime.last_thought_trace
            if trace is None:
                return ["No thought trace yet."]
            out = [trace.semantic_summary]
            for step in trace.steps:
                out.append(f"- {step.label} [{step.confidence:.2f}]: {step.content}")
            return out
        if head == "/retrieved":
            packet = runtime.last_retrieval
            if packet is None:
                return ["No retrieval packet yet."]
            return [f"- {item.kind}:{item.title} score={item.score:.2f} | {item.content}" for item in packet.top(12)]
        if head == "/critic":
            critique = runtime.last_critique
            if critique is None:
                return ["No critique yet."]
            out = [f"passed={critique.passed}"]
            out.extend(f"- issue: {issue}" for issue in critique.issues)
            out.extend(f"- revision: {rev}" for rev in critique.revisions)
            return out
        if head == "/trace":
            return [f"- {e.kind}: {e.content}" for e in runtime.recent_events(limit=12)]
        return [f"unknown command: {head}"]


# -- client ------------------------------------------------------------


class DarwinClient:
    """A thin TCP/JSON-line client. Subscribes to live thought events."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._inflight: dict[str, "queue.Queue[dict[str, Any]]"] = {}
        self._inflight_lock = threading.RLock()
        self._on_event: Callable[[dict[str, Any]], None] | None = None
        self._lock = threading.RLock()
        self._id_seq = 0

    def connect(self, on_event: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        self._on_event = on_event
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self.host, self.port))
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="darwin-client-reader",
            daemon=True,
        )
        self._reader_thread.start()
        return {}

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def chat(self, message: str, timeout: float = 30.0) -> dict[str, Any]:
        return self._request({"cmd": "chat", "message": message}, timeout=timeout)

    def command(self, command: str, timeout: float = 30.0) -> list[str]:
        result = self._request({"cmd": "command", "command": command}, timeout=timeout)
        return list(result.get("lines", []))

    def ping(self, timeout: float = 5.0) -> dict[str, Any]:
        return self._request({"cmd": "ping"}, timeout=timeout)

    def shutdown_brain(self, timeout: float = 5.0) -> dict[str, Any]:
        return self._request({"cmd": "shutdown"}, timeout=timeout)

    # -- internal -------------------------------------------------------

    def _request(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("client not connected")
        with self._lock:
            self._id_seq += 1
            request_id = str(self._id_seq)
        payload = {**payload, "id": request_id}
        inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()
        with self._inflight_lock:
            self._inflight[request_id] = inbox
        try:
            self._socket.sendall((_serialize(payload) + "\n").encode("utf-8"))
            response = inbox.get(timeout=timeout)
            return response
        finally:
            with self._inflight_lock:
                self._inflight.pop(request_id, None)

    def _reader_loop(self) -> None:
        assert self._socket is not None
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = self._socket.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    continue
                self._handle(message)

    def _handle(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        request_id = message.get("id")
        if request_id is not None:
            with self._inflight_lock:
                inbox = self._inflight.get(str(request_id))
            if inbox is not None:
                inbox.put(message)
                return
        if message_type == "event" and self._on_event is not None:
            self._on_event(message)
        elif message_type == "welcome" and self._on_event is not None:
            self._on_event(message)
