from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from darwin.agent import Darwin
from darwin.dlm import GemmaDLM, StubDLM, gemma_dlm_available
from darwin.embodiment import RoomSimulationAdapter
from darwin.instrumentation import StructuredLogger
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.server import DEFAULT_HOST, DEFAULT_PORT, DarwinClient, DarwinDaemon, PortInUseError
from darwin.streaming import StreamingSpeaker
from darwin.storage import PersistentStore
from darwin.training_data import TrainingDataCollector
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


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
    brain_parser.add_argument(
        "--dlm", choices=["stub", "gemma"], default="stub",
    )
    brain_parser.add_argument(
        "--dlm-backend", choices=["ollama", "llama-cpp", "transformers"], default="ollama",
    )
    brain_parser.add_argument("--dlm-model", default="gemma3:270m")
    brain_parser.add_argument("--quiet", action="store_true", help="Suppress local event printing.")
    brain_parser.add_argument(
        "--world",
        choices=["conceptual", "room"],
        default="conceptual",
        help=(
            "World Darwin lives in. 'conceptual' (default) is Darwin's own "
            "concept universe; 'room' is the legacy v5 AdaptiveRoomWorld."
        ),
    )
    brain_parser.add_argument(
        "--demo-seed",
        action="store_true",
        help=(
            "Opt in to the hardcoded demo concept seed (physics, math, music, "
            "...). Off by default — the universe starts with structural "
            "primitives and grows from chat + derivation."
        ),
    )

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

    chat_parser = subparsers.add_parser(
        "chat",
        help="Alias of 'connect' — clean chat REPL attached to 'darwin brain'.",
    )
    chat_parser.add_argument("--host", default=DEFAULT_HOST)
    chat_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    chat_parser.add_argument("--watch-events", action="store_true")
    chat_parser.add_argument("--text-delay", type=float, default=0.0)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help=(
            "Run a one-shot instrument command against a running brain "
            "(/snapshot, /diff, /divergence, /beliefs, etc.). Prints lines "
            "to stdout for scripting; does not open a chat REPL."
        ),
    )
    inspect_parser.add_argument(
        "instrument",
        help="Instrument slash-command, e.g. '/divergence' or '/diff a b'.",
    )
    inspect_parser.add_argument("--host", default=DEFAULT_HOST)
    inspect_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    inspect_parser.add_argument("--timeout", type=float, default=10.0)

    bench_parser = subparsers.add_parser(
        "bench",
        help="Run the benchmark suite against a fresh runtime and save a scorecard.",
    )
    bench_parser.add_argument(
        "subcommand",
        choices=["run", "compare", "list"],
        help="run = execute the suite; compare = diff two saved scorecards; list = show saved scorecards.",
    )
    bench_parser.add_argument(
        "--label", default="",
        help="A human-readable label for this scorecard (defaults to the timestamp).",
    )
    bench_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the scorecard JSON (default: under DARWIN_DATA_DIR/bench/).",
    )
    bench_parser.add_argument(
        "--earlier", type=Path, default=None,
        help="(compare) path to the earlier scorecard JSON.",
    )
    bench_parser.add_argument(
        "--later", type=Path, default=None,
        help="(compare) path to the later scorecard JSON.",
    )
    bench_parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="(list) directory of scorecards (default: DARWIN_DATA_DIR/bench/).",
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
            world_kind=args.world,
            demo_seed=args.demo_seed,
        )
    if args.command == "connect" or args.command == "chat":
        return connect(
            args.host,
            args.port,
            args.watch_events,
            args.text_delay,
        )
    if args.command == "inspect":
        return inspect(args.host, args.port, args.instrument, args.timeout)
    if args.command == "bench":
        return bench(
            args.subcommand,
            label=args.label,
            out=args.out,
            earlier=args.earlier,
            later=args.later,
            directory=args.dir,
        )
    if args.command == "export-training":
        return export_training(args.source, args.destination, args.min_quality, args.renderer)
    return 1


def bench(
    subcommand: str,
    *,
    label: str = "",
    out: Path | None = None,
    earlier: Path | None = None,
    later: Path | None = None,
    directory: Path | None = None,
) -> int:
    """Benchmark CLI: run / compare / list."""

    import time as _time

    from darwin.bench import (
        BenchmarkRunner,
        build_default_suite,
        compare_scorecards,
        load_scorecard,
        save_scorecard,
    )
    from darwin.paths import data_dir

    bench_dir = directory or (data_dir() / "bench")
    if subcommand == "list":
        bench_dir.mkdir(parents=True, exist_ok=True)
        scorecards = sorted(bench_dir.glob("*.json"))
        if not scorecards:
            print(f"no scorecards in {bench_dir}")
            return 0
        for path in scorecards:
            card = load_scorecard(path)
            if card is None:
                print(f"  {path.name}: (unreadable)")
                continue
            print(
                f"  {path.name}: label={card.label!r} overall={card.overall:.3f} "
                f"completed_at={_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(card.completed_at))}"
            )
        return 0

    if subcommand == "compare":
        if earlier is None or later is None:
            print("compare requires --earlier and --later pointing at scorecard JSON files")
            return 2
        a = load_scorecard(earlier)
        b = load_scorecard(later)
        if a is None or b is None:
            print("could not load one or both scorecards")
            return 2
        cmp = compare_scorecards(a, b)
        print(f"earlier ({cmp.earlier_label}): overall {cmp.earlier_overall:.3f}")
        print(f"later   ({cmp.later_label}):   overall {cmp.later_overall:.3f}")
        sign = "+" if cmp.overall_delta >= 0 else ""
        print(f"delta:  {sign}{cmp.overall_delta:.3f}    winner: {cmp.winner}")
        print()
        print("per-category:")
        for delta in cmp.per_category:
            sign = "+" if delta.delta >= 0 else ""
            print(
                f"  {delta.category:>16}: {delta.earlier:.3f} -> {delta.later:.3f} "
                f"({sign}{delta.delta:.3f})"
            )
        return 0

    # run
    print("Running default benchmark suite against a fresh Darwin runtime ...")
    runtime = _bench_runtime()
    try:
        runner = BenchmarkRunner(build_default_suite())
        card = runner.run(runtime, label=label or _time.strftime("%Y%m%d_%H%M%S"))
    finally:
        try:
            runtime.stop()
        except Exception:
            pass
    if out is None:
        bench_dir.mkdir(parents=True, exist_ok=True)
        out = bench_dir / f"scorecard_{card.scorecard_id}.json"
    save_scorecard(card, out)
    print(f"scorecard saved to {out}")
    print(f"overall: {card.overall:.3f}")
    print("per-category:")
    for cat, score in sorted(card.per_category.items()):
        print(f"  {cat:>16}: {score:.3f}")
    print()
    print("per-task:")
    for result in card.results:
        tag = "PASS" if result.score >= 0.6 else ("PARTIAL" if result.score > 0 else "FAIL")
        print(f"  [{tag}] {result.task_id}: {result.score:.2f}")
        if result.error:
            print(f"    error: {result.error[:200]}")
    return 0


def _bench_runtime():
    """Build a fresh DarwinRuntime suitable for benchmarking."""

    from darwin.agent import Darwin
    from darwin.runtime import DarwinRuntime, ensure_chat_action
    from darwin.types import Goal
    from darwin.universe import ConceptDeriver, ConceptualWorld, build_default_universe

    universe = build_default_universe()
    deriver = ConceptDeriver(universe)
    adapter = ConceptualWorld(universe, deriver=deriver, seed=11)
    actions = ensure_chat_action(adapter.possible_actions())
    darwin = Darwin(actions=actions, seed=11, exploration_rate=0.15)
    goal = Goal(
        desired={"neighbor_domains": 4, "concept_count": 50},
        weights={"neighbor_domains": 1.0, "concept_count": 1.0},
        exploration_weight=0.4,
    )
    runtime = DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=goal,
        interval=0.5,
    )
    return runtime


def inspect(host: str, port: int, instrument: str, timeout: float) -> int:
    """Run a single instrument slash-command against a running brain.

    This is the third lens of the two-terminal architecture: the brain
    terminal streams events, the chat terminal stays clean, and `darwin
    inspect` is the one-shot CLI for scripted queries (e.g. snapshot
    captures, divergence reports, belief dumps).
    """

    instrument = instrument.strip()
    if not instrument.startswith("/"):
        instrument = "/" + instrument
    client = DarwinClient(host=host, port=port)
    try:
        client.connect(lambda _message: None)
    except OSError as exc:
        print(f"could not connect to brain at {host}:{port}: {exc}")
        print("Did you start it with 'darwin brain' first?")
        return 1
    try:
        lines = client.command(instrument, timeout=timeout)
    except Exception as exc:
        print(f"inspect error: {exc}")
        return 2
    finally:
        client.close()
    for line in lines:
        print(line)
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
    *,
    world_kind: str = "conceptual",
    demo_seed: bool = False,
) -> int:
    """Run Darwin as a 24/7 daemon. No stdin loop; clients attach over TCP."""

    store = PersistentStore(memory_path)
    if world_kind == "room":
        room = AdaptiveRoomWorld(seed=seed)
        adapter = RoomSimulationAdapter(room)
        actions = ensure_chat_action(adapter.possible_actions())
        goal = Goal(
            desired={"room_bright": True, "fuse_intact": True},
            weights={"room_bright": 2.0, "fuse_intact": 1.0},
            exploration_weight=0.35,
        )
    else:
        # Default: Darwin lives in its own concept universe.
        from darwin.universe import (
            ConceptDeriver,
            ConceptualWorld,
            build_default_universe,
        )

        universe = build_default_universe()
        if demo_seed:
            from darwin.universe.demo_universe import demo_seed_universe

            demo_seed_universe(universe)
        deriver = ConceptDeriver(universe)
        adapter = ConceptualWorld(universe, deriver=deriver, seed=seed)
        actions = ensure_chat_action(adapter.possible_actions())
        goal = Goal(
            desired={
                "neighbor_domains": 4,
                "concept_count": 50,
            },
            weights={"neighbor_domains": 2.0, "concept_count": 1.0},
            exploration_weight=0.40,
        )
    darwin = Darwin.from_store(
        actions=actions, store=store, seed=seed, exploration_rate=exploration,
    )
    if dlm_choice == "gemma":
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
    # If the operator opted into the hardcoded encyclopedic demo seed,
    # apply it to the runtime's actual universe instance.
    if world_kind != "room" and demo_seed:
        from darwin.universe.demo_universe import demo_seed_universe

        demo_seed_universe(runtime.universe)
        runtime.grounder.refresh()
    daemon = DarwinDaemon(runtime, host=host, port=port)

    print("Project Darwin brain")
    print(f"memory={memory_path}")
    print(f"dlm={dlm.name}")
    print(f"world={world_kind}{' +demo_seed' if demo_seed else ''}")
    print(f"universe: {runtime.universe.summary()['concepts']} concepts, "
          f"{runtime.universe.summary()['relations']} relations")
    print(f"listening on {host}:{port}")
    print(f"background loops: {', '.join(runtime.loop_intervals)}")
    print("Attach a client with: darwin chat (or darwin connect)")
    print("Press Ctrl-C to stop.")
    try:
        daemon.serve_forever()
    except PortInUseError as exc:
        print(f"\nerror: {exc}", flush=True)
        return 2
    print("brain stopped.")
    return 0


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
    world = AdaptiveRoomWorld(seed=seed)
    adapter = RoomSimulationAdapter(world)
    store = PersistentStore(memory_path)
    actions = ensure_chat_action(adapter.possible_actions())
    goal = Goal(
        desired={"room_bright": True, "fuse_intact": True},
        weights={"room_bright": 2.0, "fuse_intact": 1.0},
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
        beliefs = runtime.darwin.causal_model.beliefs(limit=15)
        if not beliefs:
            print("No grounded causal beliefs yet.")
        for belief in beliefs:
            print(
                f"- if {belief.condition}: {belief.action} -> {belief.variable} "
                f"{belief.effect} confidence={belief.confidence:.2f} n={belief.samples}"
            )
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


if __name__ == "__main__":
    raise SystemExit(main())
