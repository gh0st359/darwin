from __future__ import annotations

import argparse
import threading
from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
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
        )
    return 1


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
        if threading.current_thread().name != "darwin-runtime":
            return
        if event.kind == "chat":
            return
        with print_lock:
            print(f"\n[{event.kind}] {event.content}")
            print("darwin> ", end="", flush=True)

    runtime = DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=goal,
        store=store,
        interval=interval,
        event_sink=stream_event,
    )
    runtime.set_streaming(stream)

    print("Project Darwin live")
    print(f"memory={memory_path}")
    print("Type /help for commands. Type /exit to stop.")
    if background:
        runtime.start()
        print(f"background cognition=on interval={interval:.1f}s")
        print(f"thought stream={'on' if stream else 'off'}")
    else:
        print("background cognition=off")

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
                should_continue = _handle_command(message, runtime, goal)
                if not should_continue:
                    break
            else:
                with print_lock:
                    print(runtime.chat(message))
    finally:
        runtime.stop()

    return 0


def _handle_command(message: str, runtime: DarwinRuntime, goal: Goal) -> bool:
    parts = message.split()
    command = parts[0].lower()

    if command in {"/exit", "/quit"}:
        return False

    if command == "/help":
        print(
            "\n".join(
                [
                    "/status       show Darwin's self-model",
                    "/beliefs      show strongest causal beliefs",
                    "/concepts     show concept hierarchy",
                    "/semantics    show recent parsed meanings",
                    "/experiments  show active experiment proposals",
                    "/think        run one cognition cycle now",
                    "/dream        consolidate memory and concepts",
                    "/run N        run N cognition cycles",
                    "/plan         show the current multi-step plan",
                    "/trace        show recent runtime events",
                    "/stream on|off show or hide live background thoughts",
                    "/exit         shut down cleanly",
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

    if command == "/stream":
        if len(parts) == 1:
            print(f"thought stream={'on' if runtime.stream_enabled else 'off'}")
            return True
        value = parts[1].lower()
        if value not in {"on", "off"}:
            print("Usage: /stream on|off")
            return True
        runtime.set_streaming(value == "on")
        print(f"thought stream={value}")
        return True

    print(f"Unknown command: {command}. Type /help.")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
