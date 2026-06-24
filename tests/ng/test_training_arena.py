from __future__ import annotations

import tempfile
from pathlib import Path

from darwin.agent import Darwin
from darwin.embodiment import RoomSimulationAdapter
from darwin.ng.training_arena import NGTrainingArena
from darwin.runtime import DarwinRuntime, ensure_chat_action
from darwin.storage import PersistentStore
from darwin.types import Goal
from darwin.worlds import AdaptiveRoomWorld


def _runtime(tmpdir: Path) -> DarwinRuntime:
    world = AdaptiveRoomWorld(seed=177)
    adapter = RoomSimulationAdapter(world)
    store = PersistentStore(tmpdir / "memory.sqlite3")
    darwin = Darwin(
        actions=ensure_chat_action(adapter.possible_actions()),
        store=store,
        seed=177,
        exploration_rate=0.05,
    )
    return DarwinRuntime(
        darwin=darwin,
        adapter=adapter,
        goal=Goal(desired={"room_bright": True}),
        interval=100.0,
        state_path=False,
    )


def test_training_arena_runs_cycles_and_writes_visual_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmpdir = Path(directory)
        runtime = _runtime(tmpdir)
        arena = NGTrainingArena(runtime, out_dir=tmpdir / "arena")

        session = arena.run(cycles=3, activate_every=2, visual=False)

        assert session.session_id
        assert len(session.frames) == 3
        assert session.frames[-1].frontier_score >= 0.0
        assert session.frames[-1].awareness_index > 0.0
        assert session.frames[-1].strategic_power_index > 0.0
        assert session.trace_path.exists()
        assert session.html_path.exists()
        html = session.html_path.read_text(encoding="utf-8")
        assert "Darwin NG Training Arena" in html
        assert "awareness" in html
        assert "strategic" in html
        assert "living" in html


def test_training_arena_frame_contains_visible_thought_state() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmpdir = Path(directory)
        runtime = _runtime(tmpdir)
        arena = NGTrainingArena(runtime, out_dir=tmpdir / "arena")

        session = arena.run(cycles=1, visual=False)
        frame = session.frames[0]
        rendered = frame.render_ansi(width=72)

        assert "Darwin NG :: cycle 1" in rendered
        assert "reply:" in rendered
        assert "dominant need:" in rendered
        assert "top objective:" in rendered
        assert "curriculum:" in rendered


def test_ng_train_cli_runs_and_reports_artifacts(capsys) -> None:
    from darwin.cli import main

    with tempfile.TemporaryDirectory() as directory:
        code = main([
            "ng-train",
            "--cycles", "1",
            "--out", str(Path(directory) / "out"),
            "--no-visual",
        ])

    output = capsys.readouterr().out
    assert code == 0
    assert "Darwin NG training session:" in output
    assert "trace:" in output
    assert "html:" in output
