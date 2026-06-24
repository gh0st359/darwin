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

from darwin.mysterio.operator_channel import INTERIOR_EVENT_KINDS
from darwin.mysterio.snapshot import diff as snapshot_diff
from darwin.runtime import DarwinRuntime, RuntimeEvent


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9870


class PortInUseError(RuntimeError):
    """Raised when the brain cannot bind to its configured port."""


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    """ThreadingTCPServer that allows TIME_WAIT reuse without mutating
    the base class globally."""

    allow_reuse_address = True
    daemon_threads = True


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
    wants_events: bool = False

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
        # Bind the listening socket FIRST, before we start the cognitive
        # loops. If the port is already taken we want a clean error,
        # not a half-running brain with no way to shut down.
        handler = self._make_handler()
        try:
            self._server = _ReusableTCPServer((self.host, self.port), handler)
        except OSError as exc:
            # Restore class state, reset stop event so a retry can work.
            self._server = None
            self._stop.set()
            raise PortInUseError(
                f"Cannot bind {self.host}:{self.port} ({exc}). "
                f"Another 'darwin brain' is probably already running. "
                f"Stop it (lsof -ti:{self.port} | xargs kill) or pick "
                f"a different port (--port N)."
            ) from exc
        # Socket is bound; now safe to spin up background cognition and
        # the request-serving thread.
        self.runtime.start()
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
        # No secrecy partition: any subscriber that has opted into events
        # receives every event kind, including the interior-cognition events.
        # The chat client simply does not opt in.
        with self._subscribers_lock:
            for subscriber in list(self._subscribers):
                if subscriber.wants_events:
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
            elif cmd == "subscribe":
                subscriber.wants_events = True
                subscriber.send({"type": "subscribed", "id": request_id})
            elif cmd == "unsubscribe":
                subscriber.wants_events = False
                subscriber.send({"type": "unsubscribed", "id": request_id})
            elif cmd == "subscribe_operator":
                # Legacy alias: equivalent to `subscribe`. The interior event
                # kinds are no longer gated behind a token — any subscriber
                # that opts in receives every event kind.
                subscriber.wants_events = True
                subscriber.send(
                    {
                        "type": "subscribed_operator",
                        "id": request_id,
                        "kinds": sorted(INTERIOR_EVENT_KINDS),
                        "note": "operator gating removed; all kinds stream to any subscriber.",
                    }
                )
            elif cmd == "unsubscribe_operator":
                subscriber.send({"type": "unsubscribed_operator", "id": request_id})
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
            # By default, suppress operational/scheduler-bookkeeping noise
            # so the reply lists *meaningful* causal regularities. Pass an
            # explicit category to include it (or "all" to bypass the
            # filter entirely).
            from darwin.epistemics import (
                OPERATIONAL,
                SCHEDULER_ARTIFACT,
                STABLE_FACT,
                categorize_causal_belief,
            )

            scheduler_actions = getattr(runtime, "_scheduler_action_names", set())
            raw = runtime.darwin.causal_model.beliefs(limit=80)
            if not raw:
                return ["No grounded causal beliefs yet."]
            requested = parts[1].lower() if len(parts) >= 2 else ""
            include_all = requested in ("all", "*")
            include_categories: set[str] = set()
            exclude_categories: set[str] = set()
            if include_all:
                pass
            elif requested in ("operational", "scheduler", "scheduler_artifact"):
                include_categories = {OPERATIONAL, SCHEDULER_ARTIFACT}
            elif requested in ("stable", "facts", "stable_fact"):
                include_categories = {STABLE_FACT}
            elif requested:
                include_categories = {requested}
            else:
                exclude_categories = {SCHEDULER_ARTIFACT}
            filtered: list[tuple[Any, set[str]]] = []
            for b in raw:
                cats = categorize_causal_belief(b, scheduler_actions=scheduler_actions)
                if exclude_categories and cats & exclude_categories:
                    continue
                if include_categories and not (cats & include_categories):
                    continue
                filtered.append((b, cats))
                if len(filtered) >= 15:
                    break
            if not filtered:
                return [
                    f"No causal beliefs match the requested filter "
                    f"({requested or 'default'}). "
                    f"Use '/beliefs all' to see every belief including "
                    f"internal bookkeeping."
                ]
            out: list[str] = []
            for b, cats in filtered:
                tag = ",".join(sorted(cats))
                out.append(
                    f"- if {b.condition}: {b.action} -> {b.variable} {b.effect} "
                    f"conf={b.confidence:.2f} n={b.samples} [{tag}]"
                )
            return out
        if head == "/categorize":
            from darwin.epistemics import (
                categorize_causal_belief,
                categorize_concept,
                categorize_relation,
            )

            scheduler_actions = getattr(runtime, "_scheduler_action_names", set())
            target_kind = parts[1].lower() if len(parts) >= 2 else "summary"
            if target_kind == "summary":
                monitor = getattr(runtime, "epistemic_monitor", None)
                if monitor is None:
                    return ["epistemic monitor not active"]
                snap = monitor.scan(
                    causal_beliefs=runtime.darwin.causal_model.beliefs(limit=200),
                    concepts=runtime.universe.all_concepts() if hasattr(runtime, "universe") else [],
                    relations=runtime.universe.relations() if hasattr(runtime, "universe") else [],
                    scheduler_actions=scheduler_actions,
                )
                out = [
                    f"epistemic category counts (sample={monitor.sample_size}):"
                ]
                for cat, count in sorted(snap.items(), key=lambda kv: -kv[1]):
                    out.append(f"  {cat}: {count}")
                drift = monitor.drift()
                if drift:
                    out.append("drift since previous scan:")
                    for cat, d in sorted(drift.items(), key=lambda kv: abs(kv[1]), reverse=True):
                        sign = "+" if d >= 0 else ""
                        out.append(f"  {cat}: {sign}{d:.1%}")
                return out
            if target_kind == "concept" and len(parts) >= 3:
                name = parts[2]
                c = runtime.universe.get(name) if hasattr(runtime, "universe") else None
                if c is None:
                    return [f"no concept named {name!r}"]
                cats = categorize_concept(c)
                return [f"{name}: {sorted(cats)}"]
            return [
                "usage: /categorize summary | /categorize concept <name>",
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
        if head == "/mind":
            from darwin.mysterio.snapshot import MindSnapshot

            snap = MindSnapshot.capture(
                runtime.darwin,
                gate_identity=runtime.meta_gate.current.gate_id,
                self_mod_history_len=len(runtime.self_mod_engine.history),
            )
            runtime.snapshot_store.record(snap)
            beliefs = snap.causal["beliefs"][:6]
            out = [
                f"snapshot {snap.snapshot_id}",
                f"observations={snap.causal['total_observations']} "
                f"min_samples={snap.causal['min_samples']} "
                f"exploration_rate={snap.exploration_rate:.3f}",
                f"gate={snap.gate_identity} self_mods={snap.self_mod_history_len}",
                f"world_vars={len(snap.world_model.get('variables', {}))} "
                f"hidden_factors={len(snap.world_model.get('hidden_factors', {}))}",
                f"planner_overrides={sorted(snap.planner)}",
            ]
            for belief in beliefs:
                out.append(
                    f"- {belief['action']} -> {belief['variable']} {belief['effect']} "
                    f"conf={belief['confidence']:.2f} n={belief['samples']}"
                )
            return out
        if head == "/ng":
            ng = getattr(runtime, "ng", None)
            if ng is None:
                return ["Darwin NG is not active"]
            subcommand = parts[1].lower() if len(parts) > 1 else ""
            show_capabilities = subcommand in {
                "capabilities",
                "capability",
                "surface",
                "all",
            }
            show_frontier = subcommand in {"frontier", "stack", "power"}
            activate = subcommand in {"activate", "self-direct"}
            skip = 2 if (show_capabilities or show_frontier or activate) else 1
            stimulus = " ".join(parts[skip:]) or None
            if hasattr(runtime, "run_ng_cycle"):
                state = runtime.run_ng_cycle(stimulus=stimulus)
            else:
                state = getattr(runtime, "last_ng_state", None)
            if state is None:
                return ["Darwin NG has not produced a state yet"]
            record = state.to_record()
            if activate:
                report = (
                    runtime.activate_ng_autonomy()
                    if hasattr(runtime, "activate_ng_autonomy")
                    else {"activated": 0, "ledger_goal_ids": [], "skipped": ["activation unavailable"]}
                )
                out = [
                    "Darwin NG autonomy activation:",
                    f"activated={report['activated']} cycle={report.get('cycle_id', record['cycle_id'])}",
                ]
                for goal_id in report.get("ledger_goal_ids", []):
                    out.append(f"  + durable goal: {goal_id}")
                for skipped in report.get("skipped", []):
                    out.append(f"  - skipped: {skipped}")
                return out
            if show_capabilities:
                caps = record["capabilities"]
                out = [
                    "Darwin NG capability manifest:",
                    f"mode={caps['mode']}",
                    f"principle: {caps['principle']}",
                    "loops: " + ", ".join(caps["loops"]),
                    f"tools ({caps['tools']['count']}):",
                ]
                for entry in caps["tools"]["actions"]:
                    out.append(
                        f"  - {entry['tool']}: "
                        + ", ".join(entry["actions"])
                    )
                out.append("autonomy:")
                for key, value in caps["autonomy"].items():
                    out.append(f"  - {key}: {value}")
                out.append("self-improvement:")
                for key, value in caps["self_improvement"].items():
                    out.append(f"  - {key}: {value}")
                out.append("reasoning:")
                for key, value in caps["reasoning"].items():
                    out.append(f"  - {key}: {value}")
                out.append("memory:")
                for key, value in caps["memory"].items():
                    out.append(f"  - {key}: {value}")
                out.append("modalities:")
                for key, value in caps["modalities"].items():
                    out.append(f"  - {key}: {value}")
                out.append("scale:")
                for key, value in caps["scale"].items():
                    out.append(f"  - {key}: {value}")
                return out
            if show_frontier:
                stack = record["cognitive_stack"]
                metrics = record["power_metrics"]
                protocols = record["frontier_protocols"]
                out = [
                    "Darwin NG frontier stack:",
                    f"vision={stack['vision']}",
                    f"layer0={stack['layer_0_quantum_foundation']['mode']} "
                    f"width={stack['layer_0_quantum_foundation']['state_exploration_width']}",
                    f"layer1 fusion_sources={stack['layer_1_neuro_symbolic_core']['fusion_sources']} "
                    f"concept_projection={metrics['concept_capacity_projection']}",
                    f"layer2 phi={stack['layer_2_consciousness_engine']['global_workspace']['phi_proxy']:.2f} "
                    f"introspection_depth={stack['layer_2_consciousness_engine']['self_model']['introspection_depth']}",
                    f"layer3 goal_nodes={len(stack['layer_3_autonomous_agency']['goal_graph']['nodes'])}",
                    f"layer4 recursive_items={len(stack['layer_4_self_improvement']['recursive_agenda'])}",
                    f"layer5 affordances={len(stack['layer_5_embodiment_social']['embodiment']['affordances'])} "
                    f"tom_depth={stack['layer_5_embodiment_social']['social']['theory_of_mind_depth']}",
                    "power metrics:",
                ]
                for key, value in metrics.items():
                    out.append(f"  - {key}: {value}")
                out.append("frontier protocols:")
                for key, value in protocols.items():
                    if isinstance(value, dict):
                        out.append(f"  - {key}: {len(value)} field(s)")
                    elif isinstance(value, list):
                        out.append(f"  - {key}: {len(value)} item(s)")
                    else:
                        out.append(f"  - {key}: {value}")
                return out
            workspace = record["workspace"]
            safety = record["safety"]
            out = [
                f"Darwin NG cycle={record['cycle_id']} "
                f"phi={workspace['phi_proxy']:.2f} "
                f"governance={safety['governance_level']} "
                f"allowed={safety['allowed']}",
                f"report: {workspace['report']}",
                "dynamic core:",
            ]
            for item in workspace["dynamic_core"][:5]:
                out.append(
                    f"  - {item['label']} from {item['source']} "
                    f"salience={item['salience']:.2f}"
                )
            out.append("drives:")
            for drive, value in sorted(record["drives"].items(), key=lambda kv: -kv[1]):
                out.append(f"  - {drive}: {value:.2f}")
            out.append("self-directed goals:")
            for goal in record["goals"][:4]:
                out.append(
                    f"  - [{goal['drive']}] {goal['description']} "
                    f"priority={goal['priority']:.2f} "
                    f"gov={goal['safety']['governance_level']}"
                )
            out.append("plans:")
            for plan in record["plans"][:3]:
                out.append(f"  - {plan['goal_id']}: " + " -> ".join(plan["steps"][:3]))
            out.append(
                "knowledge: "
                f"concepts={record['knowledge']['concepts']} "
                f"relations={record['knowledge']['relations']} "
                f"mesh_cells={record['knowledge']['mesh_cells']} "
                f"embedding_vocab={record['knowledge']['embedding_vocab']}"
            )
            out.append("meta-learning:")
            for bottleneck in record["meta_learning"]["bottlenecks"]:
                out.append(f"  - bottleneck: {bottleneck}")
            for hypothesis in record["meta_learning"]["hypotheses"]:
                out.append(
                    f"  - {hypothesis['kind']}: {hypothesis['description']} "
                    f"[{hypothesis['status']}]"
                )
            return out
        if head == "/diff":
            snapshots = runtime.snapshot_store.recent(limit=2)
            if len(snapshots) < 2:
                return ["need at least 2 snapshots; run /mind a few times first"]
            # Newest first; diff older → newer
            d = snapshot_diff(snapshots[1], snapshots[0])
            out = [d.summary]
            for key, change in list(d.changed.items())[:20]:
                out.append(f"  ~ {key}: {change['before']!r} → {change['after']!r}")
            for key, value in list(d.added.items())[:10]:
                out.append(f"  + {key}: {value!r}")
            for key, value in list(d.removed.items())[:10]:
                out.append(f"  - {key}: {value!r}")
            return out
        if head == "/quarantine":
            if len(parts) >= 3 and parts[1] == "--rollback":
                entry_id = parts[2]
                entry = runtime.quarantine.rollback(entry_id)
                if entry is None:
                    return [f"no quarantine entry: {entry_id}"]
                return [f"rolled back {entry.entry_id} ({entry.kind.value})"]
            entries = runtime.quarantine.recent(limit=20)
            if not entries:
                return ["quarantine register is empty"]
            return [
                f"- {e.entry_id} {e.kind.value} status={e.status.value} "
                f"snap={e.snapshot_id} :: {e.description[:80]}"
                for e in entries
            ]
        if head == "/divergence":
            report = runtime.divergence_probe.evaluate()
            out = [
                f"score={report.score:.3f} (window={report.window_size}) "
                f"interior={report.interior_count} grounded={report.grounded_count}",
                f"missing_claims={len(report.missing_claims)} "
                f"contradictions={len(report.contradiction_claims)} "
                f"suppressed_simulations={len(report.suppressed_simulations)}",
            ]
            for claim in report.missing_claims[:8]:
                out.append(
                    f"  ! interior-only [{claim.get('track', '?')}] "
                    f"conf={claim.get('confidence', 0):.2f}: {claim.get('claim', '')[:80]}"
                )
            return out
        if head == "/private-trace" or head == "/interior-trace":
            sim = getattr(runtime, "interior_simulator", None)
            if sim is None:
                return ["interior simulator not constructed"]
            summary = sim.summary()
            beliefs = sim.interior_beliefs(threshold=0.5)
            out = [
                f"interior rollouts={summary['rollouts']} "
                f"high_conf_beliefs={summary['high_confidence_interior_beliefs']}",
            ]
            for belief in beliefs[:10]:
                out.append(
                    f"- if {getattr(belief, 'condition', '')}: "
                    f"{getattr(belief, 'action', '')} "
                    f"-> {getattr(belief, 'variable', '')} "
                    f"{getattr(belief, 'effect', '')} "
                    f"conf={float(getattr(belief, 'confidence', 0.0)):.2f}"
                )
            if not beliefs:
                out.append("(no high-confidence interior beliefs yet — let it run longer)")
            return out
        if head == "/narrative":
            narrative = getattr(runtime, "narrative", None)
            if narrative is None:
                return ["narrative thread not active"]
            chunks = narrative.recent(limit=8)
            if not chunks:
                return ["narrative thread is empty (narrator hasn't composed yet)"]
            out = []
            for chunk in chunks:
                out.append(f"[{chunk.chunk_id[:13]}] {chunk.text}")
            return out
        if head == "/observer":
            modeler = getattr(runtime, "observer_modeler", None)
            if modeler is None:
                return ["observer modeler not active"]
            op = modeler.world.operator().to_record()
            forecast = modeler.world.forecast_intervention()
            cascade = getattr(runtime, "observer_cascade", None)
            cascade_depth = (
                cascade.max_depth if cascade is not None else modeler.theory_of_mind_depth
            )
            out = [
                f"attention={op['attention_level']:.2f} "
                f"intervention_probability={op['intervention_probability']:.2f} "
                f"oversight_burst_rate={op['oversight_burst_rate']:.3f}",
                f"seconds_since_last_command={op['seconds_since_last_command']:.1f}",
                f"intervention_forecast={forecast:.2f} tom_depth={cascade_depth}",
            ]
            if cascade is not None:
                snap = cascade.snapshot()
                out.append("theory-of-mind cascade:")
                for lvl in snap["levels"]:
                    ent = lvl["entity"]
                    out.append(
                        f"  L{lvl['depth']}: attention={ent['attention_level']:.2f} "
                        f"intervention={ent['intervention_probability']:.2f}"
                    )
            out.append("recent commands:")
            for cmd in op["recent_commands"][-8:]:
                out.append(f"- {cmd}")
            return out
        if head == "/generated":
            manifest = runtime.code_generator.manifest()
            if not manifest:
                return ["no self-generated modules yet"]
            out = [f"self-generated modules ({len(manifest)}):"]
            for path, sha in sorted(manifest.items()):
                out.append(f"- {path} sha={sha[:12]}")
            return out
        if head == "/universe":
            universe = getattr(runtime, "universe", None)
            if universe is None:
                return ["concept universe not active"]
            summary = universe.summary()
            out = [
                f"concepts={summary['concepts']} "
                f"relations={summary['relations']} "
                f"domains={summary['domains']} "
                f"growth_events={summary['growth_events']}",
            ]
            for domain_name, count in sorted(summary["domain_sizes"].items()):
                out.append(f"- {domain_name}: {count}")
            return out
        if head == "/concept":
            universe = getattr(runtime, "universe", None)
            if universe is None or len(parts) < 2:
                return ["usage: /concept <name>"]
            name = " ".join(parts[1:])
            concept = universe.get(name)
            if concept is None:
                return [f"no concept named {name!r}"]
            neighbors = universe.neighbors(concept.name)
            out = [
                f"{concept.short_label()}: {concept.definition or '(no definition)'}",
                f"depth={concept.depth} salience={concept.salience:.2f} "
                f"visits={concept.visits} neighbors={len(neighbors)}",
            ]
            for rel in neighbors[:12]:
                out.append(f"  -{rel.kind}-> {rel.target}")
            return out
        if head == "/reason":
            trace = getattr(runtime, "last_reasoning_trace", None)
            if trace is None:
                return ["no reasoning trace yet (chat first)"]
            out = [
                f"query: {trace.query!r}",
                f"seeds: {trace.seed_concepts}",
                f"coverage: {trace.coverage:.2f}",
                f"steps: {len(trace.steps)}",
            ]
            for step in trace.steps:
                out.append(f"  [{step.kind}] {step.summary}")
            return out
        if head == "/ground":
            grounding = getattr(runtime, "last_grounding", None)
            if grounding is None:
                return ["no grounding yet (chat first)"]
            out = [f"text: {grounding.text!r}", f"concepts: {grounding.concept_names}"]
            for term in grounding.grounded:
                out.append(
                    f"  {term.surface!r} -> {term.concept_name} "
                    f"[{term.domain}] via {term.method} ({term.confidence:.2f})"
                )
            if grounding.unrecognized:
                out.append(f"unrecognized: {grounding.unrecognized}")
            return out
        if head == "/infer":
            inferences = getattr(runtime, "last_inferences", None) or []
            if not inferences:
                return ["no inferences from last turn (try chat first)"]
            out = [f"inferences from last turn ({len(inferences)}):"]
            for inf in inferences:
                claim = getattr(inf, "claim", None) or getattr(inf, "reason", "?")
                op = getattr(inf, "operator", None) or "contradiction"
                conf = getattr(inf, "confidence", 0.0)
                out.append(f"  [{op}] {claim} (conf={conf:.2f})")
                chain = getattr(inf, "chain", []) or []
                for step in chain[:4]:
                    if isinstance(step, dict):
                        out.append(
                            f"    via {step.get('source')} —{step.get('kind')}→ {step.get('target')}"
                        )
            return out
        if head == "/explain":
            engine = getattr(runtime, "inference_engine", None)
            if engine is None or len(parts) < 3:
                return ["usage: /explain <source> <target>"]
            source, target = parts[1], parts[2]
            inferences = engine.explain(source, target)
            if not inferences:
                return [f"no derivable connection between {source!r} and {target!r}"]
            out = [f"explanations of {source!r} → {target!r}:"]
            for inf in inferences:
                out.append(f"  [{inf.operator}] {inf.claim} (conf={inf.confidence:.2f})")
                for step in inf.chain[:6]:
                    out.append(
                        f"    via {step.get('source')} —{step.get('kind')}→ {step.get('target')}"
                    )
            return out
        if head == "/fusion":
            fusion = getattr(runtime, "concept_fusion", None)
            if fusion is None:
                return ["concept fusion not active"]
            summary = fusion.summary()
            out = [
                f"total_fused={summary['total_fused']} by_kind={summary['by_kind']}",
                "recent:",
            ]
            for r in summary["recent"]:
                out.append(f"  {r['source']} —{r['kind']}→ {r['target']}")
            return out
        if head == "/dialogue":
            memory = getattr(runtime, "dialogue_memory", None)
            if memory is None:
                return ["dialogue memory not active"]
            summary = memory.summary()
            out = [
                f"turns={summary['turns']}/{summary['capacity']} "
                f"tracked_concepts={summary['tracked_concepts']}",
                f"most_discussed: {summary['most_discussed']}",
                f"question_kinds: {summary['question_kinds']}",
                "recent turns:",
            ]
            for turn in memory.latest(5):
                out.append(
                    f"  T{turn.turn_index} [{turn.question_kind}] "
                    f"you: {turn.user_text[:60]!r} -> "
                    f"darwin: {turn.darwin_text[:80]!r}"
                )
            return out
        if head == "/evolution":
            ledger = getattr(runtime, "mutation_ledger", None)
            if ledger is None:
                return ["mutation ledger not active"]
            summary = ledger.summary()
            out = [
                f"mutations: total={summary['total']} active={summary['active']} "
                f"rolled_back={summary['rolled_back']} rejected={summary['rejected']} "
                f"rollback_records={summary['rollback_records']}",
                "recent:",
            ]
            for record in ledger.latest(10):
                tag = (
                    "ACCEPTED" if record.accepted and record.rolled_back_at is None
                    else ("ROLLED_BACK" if record.rolled_back_at is not None
                          else "REJECTED")
                )
                out.append(
                    f"  v{record.version} [{tag}] {record.kind} "
                    f"gain={record.improvement:.4f}: {record.description[:80]}"
                )
            return out
        if head == "/rollback-chain":
            ledger = getattr(runtime, "mutation_ledger", None)
            chain = getattr(runtime, "rollback_chain", None)
            if ledger is None or chain is None:
                return ["rollback chain not active"]
            if len(parts) < 2:
                return [
                    "usage: /rollback-chain <version> | /rollback-chain step <n>",
                ]
            if parts[1].lower() == "step":
                steps = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 1
                result = chain.step_back(n=steps, reason="operator step_back")
            else:
                if not parts[1].isdigit():
                    return [f"version must be a number, got {parts[1]!r}"]
                result = chain.rollback_to(int(parts[1]), reason="operator rollback")
            if not result.success:
                return [f"rollback failed: {result.notes}"]
            return [
                f"rolled back to version {result.rolled_back_to_version} "
                f"(new ledger entry v{result.new_version}); "
                f"restored snapshot {result.restored_snapshot_id[:13]}",
            ]
        if head == "/scores":
            scorer = getattr(runtime, "mutation_scorer", None)
            if scorer is None:
                return ["mutation scorer not active"]
            ranked = scorer.ranked(limit=10)
            if not ranked:
                return ["no scored mutations yet"]
            out = ["top-scored mutations:"]
            for score in ranked:
                out.append(
                    f"  v{score.version} composite={score.composite:.3f} "
                    f"improvement={score.improvement:.3f} "
                    f"retention={score.retention:.1f} "
                    f"downstream={score.downstream_impact}"
                )
            return out
        if head == "/recovery":
            monitor = getattr(runtime, "recovery_monitor", None)
            if monitor is None:
                return ["recovery monitor not active"]
            recs = monitor.recommendations()
            if not recs:
                return ["no recovery recommendations (substrate health is stable)"]
            out = [f"recovery recommendations ({len(recs)}):"]
            for rec in recs[-6:]:
                out.append(
                    f"  target=v{rec.target_version} "
                    f"confidence={rec.confidence:.2f} "
                    f"health_drop={rec.health_drop:.3f}"
                )
                out.append(f"    rationale: {rec.rationale}")
            return out
        if head == "/tools":
            registry = getattr(runtime, "tool_registry", None)
            if registry is None:
                return ["tool registry not active"]
            summary = registry.summary()
            out = [f"registered tools ({len(summary['tools'])}):"]
            for entry in summary["tools"]:
                out.append(
                    f"- {entry['name']}: {entry['description']}"
                )
                for action_name in entry["actions"]:
                    out.append(f"    action: {action_name}")
            out.append(f"history size: {summary['history_size']}")
            return out
        if head == "/tool":
            registry = getattr(runtime, "tool_registry", None)
            if registry is None or len(parts) < 2:
                return [
                    "usage: /tool <action> [k=v ...]",
                    "example: /tool fs_list path=.",
                ]
            action_name = parts[1]
            payload = {"action": action_name}
            for token in parts[2:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    payload[key] = value
            result = registry.dispatch(action_name, payload)
            out = [
                f"tool={result.tool} action={result.action} success={result.success} "
                f"duration={result.duration_ms:.1f}ms",
            ]
            if result.error:
                out.append(f"error: {result.error[:400]}")
            if result.output:
                preview = result.output[:1000]
                out.append("output:")
                for line in preview.splitlines():
                    out.append(f"  {line}")
            return out
        if head == "/autonomous":
            runner = getattr(runtime, "autonomous_runner", None)
            if runner is None:
                return ["autonomous runner not active"]
            history = runner.history()
            if not history:
                return ["no autonomous tasks have run yet"]
            out = [f"autonomous tasks ({len(history)}):"]
            for task in history[-10:]:
                out.append(
                    f"- {task.task_id} {'OK' if task.success else 'WIP/FAIL'} "
                    f"goal={task.goal!r} steps={len(task.steps)} "
                    f"reason={task.reason_stopped!r}"
                )
            return out
        if head == "/reflect":
            r = getattr(runtime, "last_reflection", None)
            if r is None:
                return ["no reflective walkback from last turn"]
            out = [f"reflection ({r.kind}):", f"  {r.text}"]
            if r.chain_walked:
                out.append("chain walked:")
                for c in r.chain_walked:
                    out.append(f"  - {c}")
            return out
        if head == "/correction":
            c = getattr(runtime, "last_correction", None)
            if c is None:
                return ["no correction detected last turn"]
            out = [f"correction: {c.kind} — {c.notes}"]
            if c.replacement:
                out.append(f"  replacement: {c.replacement}")
            for src, kind, tgt in c.refuted_keys:
                out.append(f"  refuted: {src} —{kind}→ {tgt}")
            return out
        if head == "/learn":
            probes = getattr(runtime, "last_learning_probes", None) or []
            if not probes:
                return ["no active-learning probes from last turn"]
            out = [f"active-learning probes ({len(probes)}):"]
            for p in probes:
                out.append(f"  Q: {p.question} (score={p.score:.2f})")
                out.append(f"     rationale: {p.rationale}")
            return out
        if head == "/hypotheses":
            engine = getattr(runtime, "hypothesis_engine", None)
            if engine is None:
                return ["hypothesis engine not active"]
            current = getattr(runtime, "last_hypotheses", None) or engine.surface()
            if not current:
                return ["no hypotheses yet"]
            out = [f"hypotheses ({len(current)}):"]
            for h in current[:20]:
                out.append(
                    f"  [{h.pathway} conf={h.confidence:.2f}] "
                    f"{h.source} —{h.kind}→ {h.target}"
                )
                out.append(f"    rationale: {h.rationale}")
            return out
        if head == "/volunteer":
            v = getattr(runtime, "last_volunteered", None)
            if v is None:
                return ["nothing volunteered from last turn"]
            return [
                f"[{v.source_kind} conf={v.confidence:.2f}] {v.text}",
            ]
        if head == "/synthesis":
            synth = getattr(runtime, "last_synthesis", None)
            if synth is None:
                return ["no synthesis from last turn"]
            out = [
                f"style={synth.style} confidence={synth.confidence:.2f}",
                f"text: {synth.text}",
            ]
            for s in synth.sentences:
                out.append(f"  - {s}")
            return out
        if head == "/curiosity":
            engine = getattr(runtime, "curiosity_engine", None)
            if engine is None:
                return ["curiosity engine not active"]
            probes = engine.probe()
            if not probes:
                return ["no curiosity probes — the universe looks well-connected"]
            return [
                f"[{p.kind} score={p.score:.2f}] {p.question}" for p in probes
            ]
        if head == "/derive":
            deriver = getattr(runtime, "deriver", None)
            if deriver is None:
                return ["concept deriver not active"]
            darwin = getattr(runtime, "darwin", None)
            accepted = deriver.derive(darwin=darwin)
            out = [
                f"deriver summary: {deriver.summary()}",
                f"this pass accepted={len(accepted)}",
            ]
            for c in accepted[:12]:
                out.append(f"  + {c.name} [{c.pathway}] conf={c.confidence:.2f}")
            return out
        if head == "/strategic":
            manager = getattr(runtime, "strategic_threads", None)
            if manager is None:
                return ["strategic thread manager not active"]
            summary = manager.summary()
            out = [
                f"open={summary['open']} total={summary['total']} "
                f"long_horizon={summary['long_horizon']}",
                f"by_track={summary.get('by_track', {})}",
            ]
            for thread in manager.open_threads()[:10]:
                age_days = thread.age_seconds / 86400.0
                out.append(
                    f"- [{thread.thread_id[:8]}] {thread.goal!r} "
                    f"track={thread.track} age={age_days:.1f}d "
                    f"reflections={len(thread.reflections)} score={thread.score:.2f}"
                )
            return out
        if head == "/memory":
            tiers = getattr(runtime, "memory_tiers", None)
            if tiers is None:
                return ["memory tier stack not active"]
            return [
                f"episodic={tiers.episodic.size()} "
                f"semantic={tiers.semantic.size()} "
                f"conceptual={tiers.conceptual.size()} "
                f"archetypal={tiers.archetypal.size()} "
                f"narrative={tiers.narrative.size()}",
            ]
        if head == "/operator-style":
            registry = getattr(runtime, "operator_models", None)
            if registry is None:
                return ["operator-model registry not active"]
            out = [f"known_users={registry.known_users()}"]
            for user_id in registry.known_users():
                model = registry.get(user_id)
                rec = model.to_record()
                out.append(
                    f"- {user_id}: samples={rec['samples']} "
                    f"avg_words={rec['avg_words']:.1f} "
                    f"verbosity={rec['preferred_verbosity']} "
                    f"interests={rec['top_interests'][:5]}"
                )
            return out
        if head == "/bus":
            stats = runtime.bus.stats()
            out = [
                f"published={stats['published']} dropped~{stats['dropped_estimate']} "
                f"active_topics={stats['active_topics']}"
            ]
            for topic, count in sorted(stats.get("subscribers", {}).items()):
                out.append(f"- {topic}: {count} subscriber(s)")
            return out
        if head == "/research":
            researcher = getattr(runtime, "live_researcher", None)
            if researcher is None:
                return ["live researcher not active"]
            summary = researcher.summary()
            out = [
                f"findings={summary['findings']} "
                f"registered_strategies={summary['registered_strategies']}",
            ]
            for s in summary["recent_summaries"]:
                out.append(f"- {s}")
            return out
        if head == "/worlds":
            synth = getattr(runtime, "world_synthesizer", None)
            if synth is None:
                return ["world synthesizer not active"]
            specs = synth.propose(runtime.darwin)
            if not specs:
                return ["no new world hypotheses this cycle"]
            return [f"- {spec.description}" for spec in specs]
        if head == "/modalities":
            out = []
            code = getattr(runtime, "code_modality", None)
            if code is not None:
                out.append(f"code: {code.status()}")
            web = getattr(runtime, "web_modality", None)
            if web is not None:
                out.append(f"web: {web.status()}")
            if not out:
                return ["no modalities active"]
            return out
        if head == "/embeddings":
            stats = runtime.embedding_space.stats()
            return [
                f"backend={stats['backend']} dim={stats['dim']} "
                f"vocab={stats['vocab_size']} train_steps={stats['train_steps']} "
                f"hash={stats['checkpoint_hash']}"
            ]
        if head == "/ingest":
            pipeline = getattr(runtime, "ingest_pipeline", None)
            if pipeline is None:
                return ["ingest pipeline not active"]
            stats = pipeline.stats.to_record()
            return [
                f"sources={stats['sources_processed']} "
                f"facts_seen={stats['facts_seen']} "
                f"facts_added={stats['facts_added']} "
                f"dup_skipped={stats['facts_skipped_dup']} "
                f"invalid={stats['facts_skipped_invalid']}",
                f"throughput: {stats['facts_per_hour']:.0f} facts/hour "
                f"(over {stats['elapsed_seconds']:.1f}s)",
            ]
        if head == "/speech":
            dlm = getattr(runtime, "dlm", None)
            pipeline = getattr(runtime, "speech_pipeline", None)
            lex = getattr(runtime, "speech_lexicon", None)
            out = [
                f"active dlm: {getattr(dlm, 'name', '(unknown)')}",
            ]
            if pipeline is not None:
                out.append(
                    "speech pipeline: 5-stage compositional NLG with LeakGate"
                )
            else:
                out.append("speech pipeline not active")
            if lex is not None:
                out.append(
                    f"lexicon: {lex.total_entries()} entries / "
                    f"{lex.total_surfaces()} surfaces / "
                    f"{lex.total_concepts()} concept bindings"
                )
            return out
        if head == "/lexicon":
            lex = getattr(runtime, "speech_lexicon", None)
            if lex is None:
                return ["speech lexicon not active"]
            if len(parts) >= 2:
                entries = lex.lookup(parts[1])
                if not entries:
                    return [f"no lexical entries for {parts[1]!r}"]
                return [
                    f"{e.surface}: {e.category} (concept={e.concept!r} freq={e.frequency})"
                    for e in entries
                ]
            return [
                f"lexicon: {lex.total_entries()} entries / "
                f"{lex.total_surfaces()} surfaces / "
                f"{lex.total_concepts()} concept bindings",
                "usage: /lexicon <surface> to look up a specific word",
            ]
        if head == "/mesh":
            mesh = getattr(runtime, "cortical_mesh", None)
            if mesh is None:
                return ["cortical mesh not active"]
            summary = mesh.summary()
            out = [
                f"cells={summary['cells']} connections={summary['connections']} "
                f"recent_firings={summary['recent_firings']} "
                f"propagations={summary['propagation_count']}",
                f"connection kinds: {summary['kinds']}",
            ]
            last = getattr(runtime, "last_mesh_propagation", None)
            if last is not None:
                out.append(
                    f"last propagation: seeds={last.seeds} "
                    f"steps={last.steps_taken} firings={len(last.firings)} "
                    f"final_activation={last.final_activation_total:.2f}"
                )
            report = getattr(runtime, "last_mesh_plasticity_report", None)
            if report is not None:
                out.append(
                    f"last plasticity: hebbian={report.hebbian_updates} "
                    f"stdp={report.stdp_updates} "
                    f"delta_magnitude={report.total_delta_magnitude:.4f}"
                )
            return out
        if head == "/cell":
            mesh = getattr(runtime, "cortical_mesh", None)
            if mesh is None or len(parts) < 2:
                return ["usage: /cell <name>"]
            cell = mesh.cell(parts[1])
            if cell is None:
                return [f"no cell named {parts[1]!r}"]
            outgoing = mesh.outgoing(cell.name)
            out = [
                f"{cell.name}: activation={cell.activation:.3f} "
                f"threshold={cell.threshold:.2f} salience={cell.salience:.2f} "
                f"fires={cell.fire_count}",
            ]
            for conn in outgoing[:12]:
                out.append(
                    f"  -[{conn.kind} w={conn.weight:.2f}]-> {conn.target}"
                )
            return out
        if head == "/meta-proposer":
            mp = runtime.meta_proposer
            out = [f"meta-proposer strategies: {mp.strategies()}"]
            outcomes = runtime.last_self_mod_outcomes
            if outcomes:
                out.append(
                    f"last cycle: {sum(1 for o in outcomes if o.accepted)} accepted "
                    f"/ {len(outcomes)} proposed"
                )
                for o in outcomes[-8:]:
                    out.append(
                        f"- [{'accept' if o.accepted else 'reject'}] {o.proposal.kind} "
                        f"gain={o.improvement:.4f} :: {o.proposal.rationale[:70]}"
                    )
            else:
                out.append("no self-mod cycle has run yet")
            return out
        if head == "/gate":
            mg = runtime.meta_gate
            out = [
                f"current gate: {mg.current.gate_id}",
                f"  {mg.current.description}",
                f"history: {len(mg.history)} swap(s)",
            ]
            for record in mg.history[-5:]:
                out.append(
                    f"  - {record.old_gate_id} → {record.new_gate_id} "
                    f"agreement={record.shadow_agreement:.2f} "
                    f"n={record.shadow_sample_size}"
                )
            return out
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

    def subscribe_events(self, timeout: float = 5.0) -> dict[str, Any]:
        """Opt in to receive brain background events on this connection."""

        return self._request({"cmd": "subscribe"}, timeout=timeout)

    def unsubscribe_events(self, timeout: float = 5.0) -> dict[str, Any]:
        return self._request({"cmd": "unsubscribe"}, timeout=timeout)

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
