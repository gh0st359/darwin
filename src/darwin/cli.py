from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from darwin.agent import Darwin
from darwin.dlm import GemmaDLM, StubDLM, SymbolicRealizerDLM, gemma_dlm_available
from darwin.embodiment import RoomSimulationAdapter, UniverseSimulationAdapter
from darwin.generative import ActionSpec, GenerativeUniverse, GenerativeUniverseAdapter, RuleSpec, WorldSpec, WorldSpecGenerator
from darwin.instrumentation import StructuredLogger
from darwin.kernel import ActorScheduler
from darwin.knowledge import CorpusIngestor, KnowledgeGraph
from darwin.research import LiveResearcher
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.server import DEFAULT_HOST, DEFAULT_PORT, DarwinClient, DarwinDaemon, PortInUseError
from darwin.streaming import StreamingSpeaker
from darwin.storage import PersistentStore
from darwin.training_data import TrainingDataCollector
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld, UniverseSimulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darwin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Darwin in the adaptive room world.")
    run_parser.add_argument("--steps", type=int, default=40)
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--exploration", type=float, default=0.25)

    live_parser = subparsers.add_parser("live", help="Start Darwin's always-on interactive CLI.")
    live_parser.add_argument("--seed", type=int, default=7)
    live_parser.add_argument("--exploration", type=float, default=0.20)
    live_parser.add_argument("--memory", type=Path, default=Path("darwin_memory.sqlite3"))
    live_parser.add_argument("--interval", type=float, default=3.0)
    live_parser.add_argument("--no-background", action="store_true")
    live_parser.add_argument("--no-stream", action="store_true")
    live_parser.add_argument("--no-text-stream", action="store_true")
    live_parser.add_argument("--text-delay", type=float, default=0.012)
    live_parser.add_argument(
        "--dlm",
        choices=["stub", "gemma"],
        default="stub",
        help="Language renderer: 'stub' (deterministic composer) or 'gemma' (gemma-3-270m).",
    )
    live_parser.add_argument(
        "--dlm-backend",
        choices=["ollama", "llama-cpp", "transformers"],
        default="ollama",
    )
    live_parser.add_argument("--dlm-model", default="gemma3:270m")

    brain_parser = subparsers.add_parser(
        "brain",
        help="Run Darwin as a 24/7 brain daemon. Chat clients attach via 'darwin connect'.",
    )
    brain_parser.add_argument("--seed", type=int, default=7)
    brain_parser.add_argument("--exploration", type=float, default=0.20)
    brain_parser.add_argument("--memory", type=Path, default=Path("darwin_memory.sqlite3"))
    brain_parser.add_argument("--interval", type=float, default=3.0)
    brain_parser.add_argument("--host", default=DEFAULT_HOST)
    brain_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    brain_parser.add_argument("--kernel", choices=["v3", "v4", "v5"], default="v3")
    brain_parser.add_argument("--workers", default="auto")
    brain_parser.add_argument("--accelerator", default="auto")
    brain_parser.add_argument(
        "--dlm", choices=["stub", "gemma"], default="stub",
    )
    brain_parser.add_argument(
        "--dlm-backend", choices=["ollama", "llama-cpp", "transformers"], default="ollama",
    )
    brain_parser.add_argument("--dlm-model", default="gemma3:270m")
    brain_parser.add_argument("--quiet", action="store_true", help="Suppress local event printing.")

    connect_parser = subparsers.add_parser(
        "connect",
        help="Open a clean chat REPL connected to a running 'darwin brain' daemon.",
    )
    connect_parser.add_argument("--host", default=DEFAULT_HOST)
    connect_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    connect_parser.add_argument(
        "--watch-events", action="store_true",
        help="Mirror the brain's background events into this chat window. "
             "Off by default — keep this terminal a clean conversation and watch "
             "the brain's thinking in the 'darwin brain' terminal instead.",
    )
    connect_parser.add_argument(
        "--text-delay", type=float, default=0.0,
        help="Optional per-word delay when printing responses (0 = instant).",
    )

    export_parser = subparsers.add_parser(
        "export-training",
        help="Export accepted (plan -> rendering) pairs for DLM fine-tuning.",
    )
    export_parser.add_argument(
        "--source",
        type=Path,
        default=Path("training_logs/dlm_training_pairs.jsonl"),
    )
    export_parser.add_argument(
        "--destination",
        type=Path,
        default=Path("training_logs/dlm_training_export.jsonl"),
    )
    export_parser.add_argument("--min-quality", type=float, default=0.7)
    export_parser.add_argument("--renderer", default=None)

    ingest_parser = subparsers.add_parser(
        "ingest-corpus",
        help="Ingest a curated offline corpus into Darwin's v4 knowledge graph.",
    )
    ingest_parser.add_argument("--source", choices=["wikipedia", "wikidata", "wikidump"], required=True)
    ingest_parser.add_argument("--path", type=Path, required=True)
    ingest_parser.add_argument("--memory", type=Path, default=Path("darwin_memory.sqlite3"))

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_room(args.steps, args.seed, args.exploration)
    if args.command == "live":
        return live(
            args.seed,
            args.exploration,
            args.memory,
            args.interval,
            not args.no_background,
            not args.no_stream,
            not args.no_text_stream,
            args.text_delay,
            args.dlm,
            args.dlm_backend,
            args.dlm_model,
        )
    if args.command == "brain":
        return brain(
            args.seed,
            args.exploration,
            args.memory,
            args.interval,
            args.host,
            args.port,
            args.dlm,
            args.dlm_backend,
            args.dlm_model,
            not args.quiet,
            args.kernel,
            args.workers,
            args.accelerator,
        )
    if args.command == "connect":
        return connect(
            args.host,
            args.port,
            args.watch_events,
            args.text_delay,
        )
    if args.command == "export-training":
        return export_training(args.source, args.destination, args.min_quality, args.renderer)
    if args.command == "ingest-corpus":
        return ingest_corpus(args.source, args.path, args.memory)
    return 1


def ingest_corpus(source: str, path: Path, memory_path: Path) -> int:
    from darwin.generative import SandboxedWorldCompiler

    store = PersistentStore(memory_path)
    source_type = "wikipedia" if source == "wikidump" else source
    result = CorpusIngestor(store=store).ingest(path, source_type=source_type)
    graph = KnowledgeGraph.from_store(store)
    specs = WorldSpecGenerator().generate(graph)
    compiler = SandboxedWorldCompiler()
    accepted = 0
    for spec in specs:
        validation = compiler.validate(spec)
        store.record_validation_result(
            target=f"world_spec:{spec.name}",
            valid=validation.valid,
            payload=validation.to_record(),
        )
        if validation.valid:
            store.record_world_spec(spec.to_record(), status="candidate")
            accepted += 1
    store.record_research_event(
        status="ingested",
        url=str(path),
        payload={
            "source_type": source_type,
            "atoms_created": result.atoms_created,
            "atoms_seen": result.atoms_seen,
            "specs_generated": len(specs),
            "specs_accepted": accepted,
        },
    )
    print(
        f"ingested {result.atoms_created} knowledge atoms from {path} "
        f"and generated {accepted}/{len(specs)} sandbox world specs"
    )
    return 0


def export_training(source: Path, destination: Path, min_quality: float, renderer: str | None) -> int:
    collector = TrainingDataCollector(path=source)
    collector.load_existing()
    count = collector.export(destination, min_quality=min_quality, renderer=renderer)
    print(f"exported {count} (plan, rendering) pairs to {destination}")
    return 0


def brain(
    seed: int,
    exploration: float,
    memory_path: Path,
    interval: float,
    host: str,
    port: int,
    dlm_choice: str,
    dlm_backend: str,
    dlm_model: str,
    print_events: bool,
    kernel: str = "v3",
    workers: str = "auto",
    accelerator: str = "auto",
) -> int:
    """Run Darwin as a 24/7 daemon. No stdin loop; clients attach over TCP."""

    store = PersistentStore(memory_path)
    if kernel in {"v4", "v5"}:
        adapter = _build_v4_adapter(store)
    else:
        universe = UniverseSimulation(seed=seed)
        adapter = UniverseSimulationAdapter(universe)
    actions = ensure_chat_action(adapter.possible_actions())
    if kernel == "v5":
        # v5 does not target a hand-coded v3 room/space goal. The goal
        # surface is open: exploration is the goal until the kernel's
        # curriculum scheduler (Phase D) selects a specific learning
        # priority. We pass an empty desired set so nothing leaks into
        # /experiments rationale.
        goal = Goal(
            desired={},
            weights={},
            exploration_weight=0.5,
        )
    else:
        goal = Goal(
            desired={"room.room_bright": True, "room.fuse_intact": True, "space.a.y": 0},
            weights={"room.room_bright": 1.2, "room.fuse_intact": 1.0, "space.a.y": 0.4},
            exploration_weight=0.35,
        )
    darwin = Darwin.from_store(
        actions=actions, store=store, seed=seed, exploration_rate=exploration,
    )
    if kernel == "v5" and dlm_choice == "gemma":
        print(
            "error: --dlm gemma is not supported on --kernel v5.\n"
            "v5 ships a symbolic DiscourseRealizer that owns Darwin's language\n"
            "without any LLM in the inference path. Re-run with --dlm stub or\n"
            "drop --dlm entirely; the realizer is selected automatically."
        )
        return 2
    if kernel == "v5":
        dlm = SymbolicRealizerDLM()
    elif dlm_choice == "gemma":
        if not gemma_dlm_available():
            print("warning: gemma backend requested but no local model detected; DLM will fall back when unreachable.")
        dlm = GemmaDLM(backend=dlm_backend, model=dlm_model)
    else:
        dlm = StubDLM()

    print_lock = threading.RLock()

    def local_sink(event) -> None:
        if not print_events:
            return
        if event.kind == "chat":
            return
        with print_lock:
            label = event.loop if event.loop and event.loop != "main" else event.kind
            print(f"[{label}] {event.content}", flush=True)

    runtime = DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=goal,
        store=store,
        interval=interval,
        event_sink=local_sink,
        logger=StructuredLogger(),
        dlm=dlm,
        training_collector=TrainingDataCollector(),
    )
    if kernel in {"v4", "v5"}:
        runtime.kernel_scheduler = ActorScheduler(workers=workers, accelerator=accelerator)
        runtime.kernel_mode = kernel
    # v5 swaps the runtime startup path so the kernel/driver replaces the
    # five fixed-interval daemons. We monkey-patch the daemon's start hook
    # to call ``start_v5`` instead of ``start``. (The daemon binds the
    # socket FIRST, then calls runtime.start(); we intercept that call.)
    if kernel == "v5":
        original_start = runtime.start
        def _start_v5_path() -> None:
            runtime.start_v5()
        runtime.start = _start_v5_path  # type: ignore[assignment]
        runtime._v3_start = original_start  # keep handle for tests/debug
    # Wire up Darwin's self-awareness with the full runtime context.
    from darwin.self_awareness import (
        REALIZER_KIND_GEMMA,
        REALIZER_KIND_STUB,
        REALIZER_KIND_SYMBOLIC,
        SelfIntrospector,
    )
    if kernel == "v5":
        realizer_kind = REALIZER_KIND_SYMBOLIC
    elif dlm_choice == "gemma":
        realizer_kind = REALIZER_KIND_GEMMA
    else:
        realizer_kind = REALIZER_KIND_STUB
    darwin.introspector = SelfIntrospector(
        darwin,
        runtime=runtime,
        store=store,
        kernel_mode=kernel,
        realizer_kind=realizer_kind,
        realizer_name=dlm.name,
        memory_path=memory_path,
    )
    daemon = DarwinDaemon(runtime, host=host, port=port)

    print("Project Darwin brain")
    print(f"memory={memory_path}")
    print(f"embodiment={adapter.name}")
    print(f"kernel={kernel}")
    print(f"dlm={dlm.name}")
    print(f"listening on {host}:{port}")
    if kernel == "v5":
        print("scheduler=kernel-driven (one thread, priority queue, saturation-aware)")
    else:
        print(f"background loops: {', '.join(runtime.loop_intervals)}")
    print("Attach a client with: darwin connect")
    print("Press Ctrl-C to stop.")
    try:
        daemon.serve_forever()
    except PortInUseError as exc:
        print(f"\nerror: {exc}", flush=True)
        return 2
    print("brain stopped.")
    return 0


def _build_v4_adapter(store: PersistentStore) -> GenerativeUniverseAdapter:
    records = store.load_world_specs()
    specs = [WorldSpec.from_record(record) for record in records]
    if not specs:
        graph = KnowledgeGraph.from_store(store)
        specs = WorldSpecGenerator().generate(graph)
        for spec in specs:
            store.record_world_spec(spec.to_record(), status="candidate")
    if not specs:
        specs = [_bootstrap_v4_world_spec()]
    return GenerativeUniverseAdapter(GenerativeUniverse.from_specs(specs))


def _bootstrap_v4_world_spec() -> WorldSpec:
    return WorldSpec(
        name="generated/curiosity_bootstrap",
        description="Data-only bootstrap world used until a curated corpus is ingested.",
        concepts=["curiosity", "observation", "knowledge"],
        initial_state={"curiosity.knowledge": 0.0, "curiosity.observations": 0},
        actions=[
            ActionSpec(
                name="generated/observe_pattern",
                description="Create one more grounded observation.",
                rules=[
                    RuleSpec(variable="curiosity.knowledge", operation="add", operand=1.0),
                    RuleSpec(variable="curiosity.observations", operation="add", operand=1),
                ],
                vocabulary=["curiosity", "observation", "knowledge"],
                provenance_ids=[],
            )
        ],
        provenance_ids=[],
    )


def connect(host: str, port: int, watch_events: bool, text_delay: float) -> int:
    """Open a clean chat REPL attached to a running 'darwin brain' daemon.

    By default the chat window is a clean conversation: only 'you>' for
    your input and 'darwin>' for Darwin's response. The chat client does
    not even subscribe to the brain's event firehose — those background
    thoughts still happen, they just stream in the 'darwin brain'
    terminal where they belong. Pass --watch-events to mirror them here.
    """

    print_lock = threading.RLock()
    speaker = StreamingSpeaker(enabled=text_delay > 0.0, delay=text_delay)

    def on_event(message: dict) -> None:
        # Without --watch-events, the chat window is silent. Even the
        # 'welcome' frame is suppressed here so it cannot collide with
        # the 'you>' prompt. The connection-confirmation line is printed
        # synchronously in the main thread below, before the REPL starts.
        if not watch_events:
            return
        if message.get("type") != "event":
            return
        if message.get("kind") == "chat":
            return
        loop_name = message.get("loop") or message.get("kind", "event")
        content = message.get("content", "")
        with print_lock:
            sys.stdout.write("\r\x1b[2K")
            print(f"[{loop_name}] {content}")
            sys.stdout.write("you> ")
            sys.stdout.flush()

    client = DarwinClient(host=host, port=port)
    try:
        client.connect(on_event)
    except OSError as exc:
        print(f"could not connect to brain at {host}:{port}: {exc}")
        print("Did you start it with 'darwin brain' first?")
        return 1

    print(f"Connected to brain at {host}:{port}")
    if watch_events:
        try:
            client.subscribe_events()
        except Exception as exc:
            print(f"warning: could not subscribe to events: {exc}")
        print("Watching background events (--watch-events). Use this terminal to chat too.")
    else:
        print("Clean chat window. Background thinking streams in the 'darwin brain' terminal.")
    print("Type your messages, or /help for commands. /exit to leave the chat (brain keeps running).")

    try:
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                break
            if line == "/help":
                _print_remote_help()
                continue
            if line == "/shutdown-brain":
                ack = client.shutdown_brain()
                with print_lock:
                    print(f"darwin> {ack}")
                break
            if line.startswith("/"):
                try:
                    lines = client.command(line)
                except Exception as exc:
                    with print_lock:
                        print(f"darwin> command error: {exc}")
                    continue
                with print_lock:
                    for response_line in lines:
                        print(response_line)
                continue
            try:
                result = client.chat(line)
            except Exception as exc:
                with print_lock:
                    print(f"darwin> chat error: {exc}")
                continue
            with print_lock:
                if speaker.enabled:
                    sys.stdout.write("darwin> ")
                    sys.stdout.flush()
                    speaker.write(result.get("text", ""))
                else:
                    print(f"darwin> {result.get('text', '')}")
    finally:
        client.close()
    return 0


def _print_remote_help() -> None:
    print(
        "\n".join(
            [
                "Chat: type anything (no leading slash) to talk to Darwin.",
                "/status        show Darwin's self-model",
                "/beliefs       show strongest causal beliefs",
                "/beliefs DOMAIN  show beliefs for one universe domain",
                "/universe      show the active embodiment domains",
                "/worlds        show generated v4 world specs",
                "/knowledge Q   query the v4 knowledge graph",
                "/hypotheses    show corpus and causal hypotheses",
                "/mind          show kernel/introspection state",
                "/identity      show Darwin's structural self-image (v5+)",
                "/architecture  show all modules, their roles, and current state",
                "/history N     show last N self-modifications (newest first)",
                "/research status show dormant live research status",
                "/why ID        explain belief or knowledge provenance",
                "/concepts      show concept hierarchy",
                "/experiments   show active experiment proposals",
                "/think         run one cognition cycle now",
                "/dream         consolidate memory and concepts",
                "/simulate      run one mental simulation now",
                "/selfmod       propose+test self-modifications now",
                "/uncertainty   show current per-action uncertainty scan",
                "/loops         show background loop status",
                "/causal-graph  show distilled action->variable graph",
                "/dlm           show DLM info and last render result",
                "/training      show training-data corpus summary",
                "/metrics       show structured-logger metrics snapshot",
                "/thoughts      show last internal thought trace",
                "/retrieved     show memories used for last response",
                "/critic        show self-critique of last response",
                "/trace         show recent runtime events",
                "/exit          disconnect (brain keeps running)",
                "/shutdown-brain  stop the brain daemon and disconnect",
            ]
        )
    )


def run_room(steps: int, seed: int, exploration: float) -> int:
    world = AdaptiveRoomWorld(seed=seed)
    goal = Goal(
        desired={"room_bright": True, "fuse_intact": True},
        weights={"room_bright": 2.0, "fuse_intact": 1.0},
        exploration_weight=0.35,
    )
    darwin = Darwin(
        actions=world.possible_actions(),
        seed=seed,
        exploration_rate=exploration,
    )

    print("Project Darwin causal-adaptive run")
    print(f"seed={seed} steps={steps} exploration={exploration}")
    print(f"initial_state={world.observe()}")
    print()

    transitions = darwin.run(world, goal, steps)
    for transition in transitions[-10:]:
        print(
            f"t={transition.t:02d} action={transition.action:<16} "
            f"reward={transition.reward:>5.2f} after={dict(transition.after)}"
        )

    print()
    print(f"final_state={world.observe()}")
    print()
    print("strongest causal beliefs")
    for belief in darwin.causal_model.beliefs(limit=12):
        print(
            f"- if {belief.condition}: {belief.action} -> {belief.variable} "
            f"{belief.effect} confidence={belief.confidence:.2f} n={belief.samples}"
        )

    print()
    print("salient concepts")
    for concept in darwin.memory.concepts.salient(limit=8):
        print(
            f"- {concept.name} kind={concept.kind} "
            f"support={concept.support} reward_mean={concept.reward_mean:.2f}"
        )

    return 0


def live(
    seed: int,
    exploration: float,
    memory_path: Path,
    interval: float,
    background: bool,
    stream: bool,
    text_stream: bool,
    text_delay: float,
    dlm_choice: str = "stub",
    dlm_backend: str = "ollama",
    dlm_model: str = "gemma3:270m",
) -> int:
    universe = UniverseSimulation(seed=seed)
    adapter = UniverseSimulationAdapter(universe)
    store = PersistentStore(memory_path)
    actions = ensure_chat_action(adapter.possible_actions())
    goal = Goal(
        desired={"room.room_bright": True, "room.fuse_intact": True, "space.a.y": 0},
        weights={"room.room_bright": 1.2, "room.fuse_intact": 1.0, "space.a.y": 0.4},
        exploration_weight=0.35,
    )
    darwin = Darwin.from_store(
        actions=actions,
        store=store,
        seed=seed,
        exploration_rate=exploration,
    )
    print_lock = threading.RLock()

    def stream_event(event) -> None:
        thread_name = threading.current_thread().name
        if not thread_name.startswith("darwin-"):
            return
        if event.kind == "chat":
            return
        with print_lock:
            label = event.loop if event.loop and event.loop != "main" else event.kind
            print(f"\n[{label}] {event.content}")
            print("darwin> ", end="", flush=True)

    if dlm_choice == "gemma":
        if not gemma_dlm_available():
            print("warning: gemma backend requested but no local model detected; the DLM will fall back to the composer when it cannot reach the backend.")
        dlm = GemmaDLM(backend=dlm_backend, model=dlm_model)
    else:
        dlm = StubDLM()

    logger = StructuredLogger()
    collector = TrainingDataCollector()
    runtime = DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=goal,
        store=store,
        interval=interval,
        event_sink=stream_event,
        logger=logger,
        dlm=dlm,
        training_collector=collector,
    )
    runtime.set_streaming(stream)
    speaker = StreamingSpeaker(enabled=text_stream, delay=text_delay)

    print("Project Darwin live")
    print(f"memory={memory_path}")
    print(f"embodiment={adapter.name}")
    print(f"dlm={dlm.name} (backend={dlm_backend if dlm_choice == 'gemma' else 'composer'})")
    print("Type /help for commands. Type /exit to stop.")
    if background:
        runtime.start()
        print(f"background cognition=on (loops={', '.join(runtime.loop_intervals)})")
        print(f"thought stream={'on' if stream else 'off'}")
        print(f"text stream={'on' if speaker.enabled else 'off'}")
    else:
        print("background cognition=off")
        print(f"text stream={'on' if speaker.enabled else 'off'}")

    try:
        while True:
            try:
                message = input("darwin> ").strip()
            except EOFError:
                print()
                break

            if not message:
                continue
            if message.startswith("/"):
                should_continue = _handle_command(message, runtime, goal, speaker)
                if not should_continue:
                    break
            else:
                with print_lock:
                    speaker.write(runtime.chat(message))
    finally:
        runtime.stop()

    return 0


def _handle_command(
    message: str,
    runtime: DarwinRuntime,
    goal: Goal,
    speaker: StreamingSpeaker,
) -> bool:
    parts = message.split()
    command = parts[0].lower()

    if command in {"/exit", "/quit"}:
        return False

    if command == "/help":
        print(
            "\n".join(
                [
                    "/status         show Darwin's self-model",
                    "/beliefs        show strongest causal beliefs",
                    "/beliefs DOMAIN show beliefs for one universe domain",
                    "/universe       show the active embodiment domains",
                    "/concepts       show concept hierarchy",
                    "/semantics      show recent parsed meanings",
                    "/experiments    show active experiment proposals",
                    "/think          run one cognition cycle now",
                    "/dream          consolidate memory and concepts",
                    "/simulate       run one mental simulation now (multi-step causal chain)",
                    "/selfmod        propose+test self-modifications now",
                    "/uncertainty    show current per-action uncertainty scan",
                    "/loops          show background loop status",
                    "/causal-graph   show distilled causal action->variable graph",
                    "/dlm            show DLM info and last render result",
                    "/training       show training-data corpus summary",
                    "/metrics        show structured-logger metrics snapshot",
                    "/run N          run N cognition cycles",
                    "/plan           show the current multi-step plan",
                    "/thoughts       show last internal thought trace",
                    "/reason         show compact reasoning summary",
                    "/retrieved      show memories used for last response",
                    "/critic         show self-critique of last response",
                    "/trace          show recent runtime events",
                    "/stream         inspect or change thought/text streaming",
                    "/exit           shut down cleanly",
                ]
            )
        )
        return True

    if command == "/status":
        for line in runtime.darwin.self_report().lines():
            print(line)
        print(runtime.darwin.world_model.summary(runtime.darwin.causal_model))
        print(runtime.darwin.semantic_memory.summary())
        if runtime.store is not None:
            print(f"storage={runtime.store.counts()}")
        return True

    if command == "/beliefs":
        domain = parts[1].lower() if len(parts) > 1 else None
        for line in _belief_lines(runtime, domain=domain, limit=15):
            print(line)
        return True

    if command == "/universe":
        for line in _universe_lines(runtime):
            print(line)
        return True

    if command == "/concepts":
        for concept in runtime.darwin.memory.concepts.hierarchy(limit=25):
            parents = f" parents={sorted(concept.parents)}" if concept.parents else ""
            print(
                f"- L{concept.level} {concept.kind}: {concept.name} "
                f"support={concept.support} reward_mean={concept.reward_mean:.2f}{parents}"
            )
        return True

    if command == "/semantics":
        frames = runtime.darwin.semantic_memory.recent(limit=10)
        if not frames:
            print("No semantic frames yet.")
            return True
        for frame in frames:
            print(f"- {frame.summary()}")
            if frame.goals:
                print(f"  goals={frame.goals}")
            if frame.values:
                print(f"  values={frame.values}")
            if frame.unknown_terms:
                print(f"  unknown={frame.unknown_terms[:8]}")
        return True

    if command == "/experiments":
        proposals = runtime.darwin.experiment_engine.propose(
            runtime.adapter.observe(),
            runtime.adapter.possible_actions(),
            goal=goal,
            limit=8,
        )
        for proposal in proposals:
            print(f"- {proposal.question} [{proposal.rationale}]")
        return True

    if command == "/think":
        event = runtime.cognition_cycle()
        print(f"{event.kind}: {event.content}")
        return True

    if command == "/dream":
        event = runtime.dream()
        print(event.content)
        return True

    if command == "/run":
        cycles = 1
        if len(parts) > 1:
            try:
                cycles = max(1, int(parts[1]))
            except ValueError:
                print("Usage: /run N")
                return True
        for _ in range(cycles):
            event = runtime.cognition_cycle()
            print(f"{event.kind}: {event.content}")
        return True

    if command == "/plan":
        plan = runtime.darwin.plan(
            runtime.adapter.observe(),
            goal,
            horizon=3,
            actions=runtime.adapter.possible_actions(),
        )
        print(plan.explain())
        for line in plan.trace:
            print(f"- {line}")
        return True

    if command == "/trace":
        for event in runtime.recent_events(limit=12):
            print(f"- {event.kind}: {event.content}")
        return True

    if command == "/thoughts":
        trace = runtime.last_thought_trace
        if trace is None:
            print("No thought trace yet.")
            return True
        print(trace.semantic_summary)
        for step in trace.steps:
            print(f"- {step.label} [{step.confidence:.2f}]: {step.content}")
            for evidence in step.evidence[:3]:
                print(f"  evidence: {evidence}")
        return True

    if command == "/reason":
        trace = runtime.last_thought_trace
        print(trace.compact() if trace is not None else "No reasoning trace yet.")
        return True

    if command == "/retrieved":
        packet = runtime.last_retrieval
        if packet is None:
            print("No retrieval packet yet.")
            return True
        for item in packet.top(12):
            print(f"- {item.kind}:{item.title} score={item.score:.2f}")
            print(f"  {item.content}")
        return True

    if command == "/critic":
        critique = runtime.last_critique
        if critique is None:
            print("No critique yet.")
            return True
        print(f"passed={critique.passed}")
        for issue in critique.issues:
            print(f"- issue: {issue}")
        for revision in critique.revisions:
            print(f"- revision: {revision}")
        return True

    if command == "/stream":
        if len(parts) == 1:
            print(f"thought stream={'on' if runtime.stream_enabled else 'off'}")
            print(f"text stream={'on' if speaker.enabled else 'off'}")
            return True
        if len(parts) >= 3 and parts[1].lower() in {"text", "thought", "thoughts"}:
            target = parts[1].lower()
            value = parts[2].lower()
            if value not in {"on", "off"}:
                print("Usage: /stream text on|off or /stream thoughts on|off")
                return True
            if target == "text":
                speaker.enabled = value == "on"
                print(f"text stream={value}")
            else:
                runtime.set_streaming(value == "on")
                print(f"thought stream={value}")
            return True
        value = parts[1].lower()
        if value not in {"on", "off"}:
            print("Usage: /stream on|off, /stream text on|off, or /stream thoughts on|off")
            return True
        runtime.set_streaming(value == "on")
        print(f"thought stream={value}")
        return True

    if command == "/simulate":
        snapshot = runtime.run_simulation()
        if snapshot is None:
            print("no simulation produced")
        else:
            print(f"chain confidence={snapshot.get('chain_confidence', 0):.3f}")
            print(f"chain uncertainty={snapshot.get('chain_uncertainty', 0):.3f}")
            print(f"total expected reward={snapshot.get('total_expected_reward', 0):.3f}")
            for node in snapshot.get("nodes", [])[:6]:
                print(f"- step {node['step']}: {node['action']} conf={node['confidence']:.2f}")
        return True

    if command == "/selfmod":
        outcomes = runtime.run_self_modification()
        if not outcomes:
            print("no self-modification proposals this cycle")
            return True
        for outcome in outcomes:
            mark = "accepted" if outcome.accepted else "rejected"
            print(
                f"- [{mark}] {outcome.proposal.kind} "
                f"baseline={outcome.baseline_error:.4f} candidate={outcome.candidate_error:.4f} "
                f"gain={outcome.improvement:.4f}"
            )
            print(f"    rationale: {outcome.proposal.rationale}")
            if outcome.proposal.payload:
                print(f"    payload: {outcome.proposal.payload}")
        return True

    if command == "/uncertainty":
        scan = runtime.last_uncertainty_scan
        if scan is None:
            print("no uncertainty scan yet; run the runtime in the background.")
            return True
        for item in scan.get("scan", [])[:10]:
            print(f"- {item['action']:>20} unc={item['uncertainty']:.2f}")
        return True

    if command == "/loops":
        if not runtime.running:
            print("background loops are not running")
            return True
        print("background loops:")
        for name, value in runtime.loop_intervals.items():
            state = runtime._loop_state.get(name, {})
            print(f"- {name:<18} interval={value:.1f}s last={state.get('last_event', 'n/a')}")
        return True

    if command == "/causal-graph":
        graph = runtime.darwin.planner.chain_engine.graph(min_confidence=0.0, limit=80)
        print(
            f"actions={len(graph.actions)} variables={len(graph.variables)} edges={len(graph.edges)}"
        )
        for edge in graph.edges[:20]:
            print(
                f"- {edge.source_action} -> {edge.variable} effect={edge.effect} "
                f"conf={edge.confidence:.2f} n={edge.samples}"
            )
        return True

    if command == "/dlm":
        render = runtime.last_render
        print(f"current DLM: {runtime.dlm.name}")
        if render is None:
            print("no DLM render has happened in this session yet.")
            return True
        print(f"renderer={render.renderer} valid={render.valid} duration={render.duration_ms:.1f}ms")
        for note in render.validation_notes[:5]:
            print(f"- note: {note}")
        return True

    if command == "/training":
        summary = runtime.training_collector.summary()
        print(
            f"training pairs collected={summary['total']} accepted={summary['accepted']} path={summary['path']}"
        )
        for renderer, count in summary["by_renderer"].items():
            print(f"- {renderer}: {count}")
        return True

    if command == "/metrics":
        snapshot = runtime.logger.snapshot()
        print("metrics:")
        for key, value in snapshot["metrics"].items():
            print(f"- {key}: {value}")
        if snapshot["counters"]:
            print("counters:")
            for key, value in snapshot["counters"].items():
                print(f"- {key}: {value}")
        return True

    print(f"Unknown command: {command}. Type /help.")
    return True


def _belief_lines(runtime: DarwinRuntime, domain: str | None = None, limit: int = 15) -> list[str]:
    beliefs = runtime.darwin.causal_model.beliefs(limit=limit * 3)
    if domain:
        beliefs = [
            belief
            for belief in beliefs
            if belief.action.startswith(f"{domain}/") or belief.variable.startswith(f"{domain}.")
        ][:limit]
    else:
        beliefs = beliefs[:limit]
    if not beliefs:
        suffix = f" for {domain}" if domain else ""
        return [f"No grounded causal beliefs yet{suffix}."]
    return [
        (
            f"- if {belief.condition}: {belief.action} -> {belief.variable} "
            f"{belief.effect} confidence={belief.confidence:.2f} n={belief.samples}"
        )
        for belief in beliefs
    ]


def _universe_lines(runtime: DarwinRuntime) -> list[str]:
    adapter = runtime.adapter
    state = adapter.observe()
    actions = adapter.possible_actions()
    domains = sorted(
        {
            str(action.metadata.get("domain", action.name.split("/", 1)[0] if "/" in action.name else "world"))
            for action in actions
        }
    )
    lines = [
        f"embodiment={getattr(adapter, 'name', 'unknown')}",
        "domains=" + ", ".join(domains),
        f"actions={len(actions)} variables={len(state)}",
    ]
    for domain in domains:
        domain_actions = [
            action.name
            for action in actions
            if action.metadata.get("domain") == domain or action.name.startswith(f"{domain}/")
        ]
        domain_variables = [key for key in state if key.startswith(f"{domain}.")]
        lines.append(f"- {domain}: actions={len(domain_actions)} variables={len(domain_variables)}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
